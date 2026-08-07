"""
AI Agent for ICD-10 Medical Coding.
Uses Databricks Foundation Models (Llama) with tool calling.

Improvements over v1:
- LLM client cached at module level (one secret fetch per process)
- search_icd10_codes includes sf=code,name + exponential backoff/retry
- semantic_search_icd10 uses Postgres FTS on icd10_lookup table
- retrieve_similar_notes does cosine-similarity KNN over clinical_notes
  embeddings stored in workspace.medical_coding.clinical_notes (Delta)
- get_session_history wired into TOOLS
- Forced save step retained for reliability
"""

import json, os, time, requests
import numpy as np
import lakebase

NLM_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

# ── Cached singletons ─────────────────────────────────────────────────────────

_llm_client     = None
_embed_model    = None   # sentence-transformers model (lazy-loaded)
_clinical_notes = None   # list of {note_id, specialty, note_text, embedding}

def get_llm_client():
    global _llm_client
    if _llm_client:
        return _llm_client
    import openai, base64
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    secret = w.secrets.get_secret(scope="database", key="databricks-token")
    token  = base64.b64decode(secret.value).decode("utf-8")
    _llm_client = openai.OpenAI(
        api_key=token,
        base_url=f"{w.config.host}/serving-endpoints"
    )
    return _llm_client

def _get_embed_model():
    """Lazy-load sentence-transformer model (same model used in notebook 03)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def _load_clinical_notes():
    """
    Fetch clinical note embeddings from Delta via Databricks SQL warehouse.
    Cached at module level — loaded once per process.
    """
    global _clinical_notes
    if _clinical_notes is not None:
        return _clinical_notes
    try:
        import base64
        from databricks.sdk import WorkspaceClient
        from databricks import sql as dbsql
        w   = WorkspaceClient()
        sec = w.secrets.get_secret(scope="database", key="databricks-token")
        tok = base64.b64decode(sec.value).decode("utf-8")
        warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
        if not warehouse_id:
            _clinical_notes = []
            return _clinical_notes
        with dbsql.connect(
            server_hostname = w.config.host.replace("https://", ""),
            http_path       = f"/sql/1.0/warehouses/{warehouse_id}",
            access_token    = tok,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT specialty, note_text, embedding
                    FROM workspace.medical_coding.clinical_notes
                """)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                _clinical_notes = [dict(zip(cols, r)) for r in rows]
    except Exception:
        _clinical_notes = []
    return _clinical_notes

LLM_MODEL = os.environ.get("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")

# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_backoff(fn, max_retries=3, base_delay=1.0):
    """Exponential backoff retry — handles 429 and transient network errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = base_delay * (2 ** attempt)
                time.sleep(wait)
            elif attempt == max_retries - 1:
                raise
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout):
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    return None

# ── Tool implementations ─────────────────────────────────────────────────────

def search_icd10_codes(query: str, max_results: int = 8) -> list[dict]:
    """Search ICD-10-CM codes via NLM API with retry/backoff."""
    def _call():
        r = requests.get(
            NLM_URL,
            params={"terms": query, "maxList": max_results,
                    "df": "code,name", "sf": "code,name"},
            timeout=10
        )
        r.raise_for_status()
        data  = r.json()
        items = data[3] if len(data) > 3 else []
        return [{"code": item[0], "description": item[1]} for item in items]
    try:
        return _with_backoff(_call) or []
    except Exception as e:
        return [{"error": str(e)}]

def retrieve_similar_notes(note_text: str, top_k: int = 3) -> list[dict]:
    """
    KNN retrieval over workspace.medical_coding.clinical_notes embeddings (Delta).
    Embeds the query with all-MiniLM-L6-v2 (same model used at ingest time),
    computes cosine similarity against all stored 384-dim embeddings in memory,
    and returns the top-K most similar clinical cases with their specialties.

    Used to ground ICD-10 suggestions in historically similar cases.
    See notebooks/06_knn_clinical_notes.py for the Spark version of this logic.
    """
    try:
        notes = _load_clinical_notes()
        if not notes:
            return [{"info": "clinical_notes not available (set DATABRICKS_WAREHOUSE_ID)"}]

        model          = _get_embed_model()
        query_emb      = model.encode(note_text, normalize_embeddings=True)

        scored = []
        for n in notes:
            emb = n.get("embedding")
            if not emb:
                continue
            v    = np.array(emb, dtype=np.float32)
            v   /= (np.linalg.norm(v) or 1.0)
            sim  = float(np.dot(query_emb, v))
            scored.append((sim, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "similarity":   round(sim, 4),
                "specialty":    n["specialty"],
                "note_preview": (n["note_text"] or "")[:300],
            }
            for sim, n in scored[:top_k]
        ]
    except Exception as e:
        return [{"error": str(e)}]

def semantic_search_icd10(query: str, max_results: int = 8) -> list[dict]:
    """
    Full-text semantic search over ICD-10 codes stored in Lakebase.
    Uses Postgres tsvector/tsquery — complements NLM API especially for
    clinical phrases that NLM's prefix search misses.
    """
    try:
        rows = lakebase.run_query("""
            SELECT code, description,
                   ts_rank(to_tsvector('english', description),
                           plainto_tsquery('english', %s)) AS rank
            FROM icd10_lookup
            WHERE to_tsvector('english', description) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, max_results))
        if not rows:
            # Fallback: ILIKE partial match when FTS finds nothing
            rows = lakebase.run_query("""
                SELECT code, description, 0.5 AS rank
                FROM icd10_lookup
                WHERE description ILIKE %s
                ORDER BY code
                LIMIT %s
            """, (f"%{query}%", max_results))
        return [{"code": r["code"], "description": r["description"],
                 "rank": float(r["rank"])} for r in rows]
    except Exception as e:
        return [{"error": str(e)}]

def get_session_history(user_email: str, limit: int = 5) -> list[dict]:
    """Get past accepted codes for a user — helps avoid duplicate suggestions."""
    rows = lakebase.run_query("""
        SELECT s.session_id, s.specialty, s.created_at::text,
               ARRAY_AGG(sg.icd10_code ORDER BY sg.confidence DESC) AS codes
        FROM coding_sessions s
        LEFT JOIN code_suggestions sg
               ON sg.session_id = s.session_id AND sg.accepted = true
        WHERE s.user_email = %s
        GROUP BY s.session_id, s.specialty, s.created_at
        ORDER BY s.created_at DESC
        LIMIT %s
    """, (user_email, limit))
    return [dict(r) for r in rows]

def save_suggestions(session_id: int, suggestions: list[dict]) -> dict:
    """Save code suggestions to Lakebase."""
    for s in suggestions:
        lakebase.run_write("""
            INSERT INTO code_suggestions
                (session_id, icd10_code, description, confidence, explanation)
            VALUES (%s, %s, %s, %s, %s)
        """, (session_id, s["code"], s["description"],
              round(float(s.get("confidence", 0.8)), 3),
              s.get("explanation", "")))
    return {"saved": len(suggestions)}

def log_tool_call(session_id: int, tool_name: str,
                  tool_input: dict, tool_output) -> None:
    """Audit trail — every agent tool call is logged."""
    lakebase.run_write("""
        INSERT INTO agent_tool_calls (session_id, tool_name, tool_input, tool_output)
        VALUES (%s, %s, %s::jsonb, %s::jsonb)
    """, (session_id, tool_name,
          json.dumps(tool_input), json.dumps(tool_output)))

# ── Tool definitions for LLM ─────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_similar_notes",
            "description": (
                "KNN retrieval: finds the most similar clinical cases in the "
                "workspace.medical_coding.clinical_notes Delta table using "
                "384-dim sentence-transformer embeddings and cosine similarity. "
                "Call FIRST to ground suggestions in historically similar cases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note_text": {"type": "string",
                                  "description": "The clinical note text to find similar cases for"},
                    "top_k":    {"type": "integer", "default": 3}
                },
                "required": ["note_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_icd10_codes",
            "description": "Search ICD-10-CM codes via NLM API (with retry/backoff). Use for specific medical terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Clinical term or diagnosis to search for"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search_icd10",
            "description": (
                "Full-text search over 12,316 ICD-10-CM codes in Lakebase (Postgres tsvector). "
                "Use when search_icd10_codes returns no results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Clinical phrase or description to search"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_history",
            "description": "Retrieve past coding sessions for a user to avoid duplicate suggestions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_email": {"type": "string"},
                    "limit":      {"type": "integer", "default": 5}
                },
                "required": ["user_email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_suggestions",
            "description": "Save ALL final ICD-10 code suggestions to the database. MUST be called once at the end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "suggestions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code":        {"type": "string"},
                                "description": {"type": "string"},
                                "confidence":  {"type": "number"},
                                "explanation": {"type": "string"}
                            },
                            "required": ["code", "description", "confidence", "explanation"]
                        }
                    }
                },
                "required": ["session_id", "suggestions"]
            }
        }
    }
]

TOOL_MAP = {
    "retrieve_similar_notes": retrieve_similar_notes,
    "search_icd10_codes":     search_icd10_codes,
    "semantic_search_icd10":  semantic_search_icd10,
    "get_session_history":    get_session_history,
    "save_suggestions":       save_suggestions,
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a certified professional medical coder (CPC) specializing in ICD-10-CM diagnosis coding.
Your job is to assign accurate, billable ICD-10-CM codes following the Official Guidelines for Coding and Reporting (CMS/NCHS).

WORKFLOW — follow this exact order:
1. Call retrieve_similar_notes with the full note text to find historically similar cases.
   Study their specialties and coding patterns before proceeding.
2. Identify the PRINCIPAL DIAGNOSIS — the condition established after study to be chiefly
   responsible for this encounter. This drives all other coding decisions.
3. Identify ADDITIONAL diagnoses ONLY if they meet ALL of these criteria (Section III guidelines):
   - Documented by the provider (not assumed or inferred)
   - Required clinical evaluation, therapeutic treatment, or diagnostic procedures
     during THIS encounter — OR — affected nursing care/monitoring
   - Actively managed, not simply noted in history
   DO NOT CODE:
   - Conditions described as "possible", "probable", "suspected", "rule out", or "?"
   - Signs/symptoms that are integral to a confirmed diagnosis (e.g. chest pain when
     MI is confirmed — code the MI, not the chest pain)
   - Incidental findings that did not affect patient management (e.g. an unrelated
     finding noted during a procedure but left untreated)
   - Historical conditions with no current relevance to this encounter
   - Conditions from a previous encounter that are fully resolved
4. For each condition to code, call search_icd10_codes. If fewer than 3 results,
   also call semantic_search_icd10.
5. Select the most specific code:
   - Use 7th character extensions where required (fractures: A=initial, D=subsequent,
     S=sequela; injuries; obstetric codes) — omitting the 7th character is a coding error
   - Laterality is built into the code — match left/right/bilateral exactly as documented
   - Use etiology + manifestation pairs (e.g. diabetic retinopathy: E11.3- + H36)
   - Use combination codes when available (e.g. diabetes with CKD = E11.65 not two codes)
   - Avoid "unspecified" codes when the documentation supports specificity
6. Assign confidence scores:
   - 0.95–1.0: explicitly stated, definitive diagnosis with direct code match
   - 0.80–0.94: documented condition, best available code (minor specificity gap)
   - 0.60–0.79: inferred from context or symptom-level coding required
7. Call save_suggestions. Each explanation must quote the exact phrase from the note
   that supports the code, and state why additional conditions qualify under Section III.

SEQUENCING: Principal diagnosis first → complications → comorbidities managed this visit
NEVER return plain text without calling save_suggestions first."""

# ── Agent runner ──────────────────────────────────────────────────────────────

def run_agent(session_id: int, note_text: str,
              specialty: str = None, user_email: str = None) -> list[dict]:
    """
    Run the AI coding agent on a clinical note.
    Returns suggestions saved to the database.
    """
    client = get_llm_client()

    user_msg = "Please code this clinical note"
    if specialty:
        user_msg += f" (specialty: {specialty})"
    user_msg += f":\n\n{note_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg}
    ]

    for _ in range(10):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2000
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if fn_name == "save_suggestions":
                fn_args["session_id"] = session_id

            fn     = TOOL_MAP.get(fn_name)
            result = fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}

            try:
                log_tool_call(session_id, fn_name, fn_args, result)
            except Exception:
                pass

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result)
            })

    # Check if suggestions were saved
    suggestions = lakebase.run_query(
        "SELECT * FROM code_suggestions WHERE session_id = %s ORDER BY confidence DESC",
        (session_id,)
    )

    # Force save if agent forgot
    if not suggestions:
        messages.append({
            "role":    "user",
            "content": "You MUST now call save_suggestions with all codes found. Do not write text."
        })
        force_resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "save_suggestions"}},
            max_tokens=2000
        )
        force_msg = force_resp.choices[0].message
        if force_msg.tool_calls:
            for tc in force_msg.tool_calls:
                fn_args = json.loads(tc.function.arguments)
                fn_args["session_id"] = session_id
                result = save_suggestions(**fn_args)
                try:
                    log_tool_call(session_id, "save_suggestions_forced", fn_args, result)
                except Exception:
                    pass

        suggestions = lakebase.run_query(
            "SELECT * FROM code_suggestions WHERE session_id = %s ORDER BY confidence DESC",
            (session_id,)
        )

    return suggestions
