"""
AI Agent for ICD-10 Medical Coding.
Uses Databricks Foundation Models (Llama) with tool calling.

Improvements over v1:
- LLM client cached at module level (one secret fetch per process)
- search_icd10_codes includes sf=code,name for full description search
- semantic_search_icd10 tool uses Postgres full-text search over icd10_lookup table
- get_session_history wired into TOOLS so agent can reference past sessions
- Forced save step retained for reliability
"""

import json, os, requests
import lakebase

NLM_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

# ── Cached LLM client (created once per process) ─────────────────────────────

_llm_client = None

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

LLM_MODEL = os.environ.get("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")

# ── Tool implementations ─────────────────────────────────────────────────────

def search_icd10_codes(query: str, max_results: int = 8) -> list[dict]:
    """Search ICD-10-CM codes via NLM API (searches both code and description)."""
    try:
        r = requests.get(
            NLM_URL,
            params={
                "terms":   query,
                "maxList": max_results,
                "df":      "code,name",
                "sf":      "code,name",   # search in description too
            },
            timeout=10
        )
        r.raise_for_status()
        data  = r.json()
        items = data[3] if len(data) > 3 else []
        return [{"code": item[0], "description": item[1]} for item in items]
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
            "name": "search_icd10_codes",
            "description": "Search ICD-10-CM codes via NLM API. Use for specific medical terms.",
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
                "Semantic full-text search over all ICD-10-CM codes using Postgres. "
                "Use this when search_icd10_codes returns no results or when you need "
                "broader clinical phrase matching."
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
    "search_icd10_codes":    search_icd10_codes,
    "semantic_search_icd10": semantic_search_icd10,
    "get_session_history":   get_session_history,
    "save_suggestions":      save_suggestions,
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert medical coder specializing in ICD-10-CM coding.
Your job is to read clinical documentation and assign the most accurate diagnosis codes.

When given a clinical note:
1. Identify ALL significant diagnoses, conditions, and complications mentioned
2. For each condition, call search_icd10_codes. If it returns no results, call semantic_search_icd10
3. Select the most specific code available (avoid unspecified codes when specificity exists)
4. After finding all codes, call save_suggestions with ALL codes and your reasoning
5. Follow coding guidelines: code the principal diagnosis first, then secondary diagnoses

Apply these ICD-10 coding principles:
- Code to the highest degree of specificity
- Do not code signs/symptoms when a definitive diagnosis is documented
- Code chronic conditions that are actively managed
- Include laterality when documented (left, right, bilateral)
- ALWAYS end by calling save_suggestions — do not return text without saving"""

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
