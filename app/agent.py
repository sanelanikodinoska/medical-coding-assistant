"""
AI Agent for ICD-10 Medical Coding
Uses Databricks Foundation Models (Llama) with tool calling.
"""

import json, os, requests
import lakebase

NLM_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

# ── Tool implementations ─────────────────────────────────────────────────────

def search_icd10_codes(query: str, max_results: int = 8) -> list[dict]:
    """Search ICD-10-CM codes using the NLM API."""
    try:
        r = requests.get(NLM_URL, params={"terms": query, "maxList": max_results, "df": "code,name"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data[3] if len(data) > 3 else []
        return [{"code": item[0], "description": item[1]} for item in items]
    except Exception as e:
        return [{"error": str(e)}]

def get_session_history(user_email: str, limit: int = 5) -> list[dict]:
    """Get past coding sessions for a user."""
    rows = lakebase.run_query("""
        SELECT s.session_id, s.specialty, s.created_at,
               ARRAY_AGG(sg.icd10_code ORDER BY sg.confidence DESC) AS codes
        FROM coding_sessions s
        LEFT JOIN code_suggestions sg ON sg.session_id = s.session_id AND sg.accepted = true
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
            INSERT INTO code_suggestions (session_id, icd10_code, description, confidence, explanation)
            VALUES (%s, %s, %s, %s, %s)
        """, (session_id, s["code"], s["description"],
              round(float(s.get("confidence", 0.8)), 3), s.get("explanation", "")))
    return {"saved": len(suggestions)}

def log_tool_call(session_id: int, tool_name: str, tool_input: dict, tool_output) -> None:
    """Log every agent tool call for the audit trail."""
    lakebase.run_write("""
        INSERT INTO agent_tool_calls (session_id, tool_name, tool_input, tool_output)
        VALUES (%s, %s, %s::jsonb, %s::jsonb)
    """, (session_id, tool_name, json.dumps(tool_input), json.dumps(tool_output)))

# ── Tool definitions for LLM ─────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_icd10_codes",
            "description": "Search ICD-10-CM diagnosis codes matching a clinical concept or symptom",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Clinical term, symptom, or diagnosis to search for"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_suggestions",
            "description": "Save the final ICD-10 code suggestions to the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "suggestions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code":        {"type": "string", "description": "ICD-10-CM code e.g. I21.9"},
                                "description": {"type": "string", "description": "Official code description"},
                                "confidence":  {"type": "number", "description": "0.0 to 1.0"},
                                "explanation": {"type": "string", "description": "Why this code applies"}
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
    "search_icd10_codes": search_icd10_codes,
    "save_suggestions":   save_suggestions,
}

# ── Agent runner ─────────────────────────────────────────────────────────────

def get_llm_client():
    try:
        import openai
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        return openai.OpenAI(
            api_key=w.config.token,
            base_url=f"{w.config.host}/serving-endpoints"
        )
    except Exception as e:
        raise RuntimeError(f"Could not create LLM client: {e}")

LLM_MODEL = os.environ.get("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")

SYSTEM_PROMPT = """You are an expert medical coder specializing in ICD-10-CM coding.
Your job is to read clinical documentation and assign the most accurate diagnosis codes.

When given a clinical note:
1. Identify ALL significant diagnoses, conditions, and complications mentioned
2. For each condition, call search_icd10_codes to find the correct ICD-10-CM code
3. Select the most specific code available (avoid unspecified codes when specificity exists)
4. After finding all codes, call save_suggestions with ALL codes and your reasoning
5. Follow coding guidelines: code the principal diagnosis first, then secondary diagnoses

Apply these ICD-10 coding principles:
- Code to the highest degree of specificity
- Do not code signs/symptoms when a definitive diagnosis is documented
- Code chronic conditions that are actively managed
- Include laterality when documented (left, right, bilateral)"""

def run_agent(session_id: int, note_text: str, specialty: str = None) -> list[dict]:
    """
    Run the AI coding agent on a clinical note.
    Returns the list of code suggestions saved to the database.
    """
    client = get_llm_client()

    user_msg = f"Please code this clinical note"
    if specialty:
        user_msg += f" (specialty: {specialty})"
    user_msg += f":\n\n{note_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg}
    ]

    # Agentic loop — keep going until no more tool calls
    for iteration in range(10):
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

        # Execute each tool call
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            # Inject session_id for save_suggestions
            if fn_name == "save_suggestions":
                fn_args["session_id"] = session_id

            fn = TOOL_MAP.get(fn_name)
            if fn:
                result = fn(**fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            # Log the tool call
            try:
                log_tool_call(session_id, fn_name, fn_args, result)
            except Exception:
                pass

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })

    # Return saved suggestions from DB
    return lakebase.run_query(
        "SELECT * FROM code_suggestions WHERE session_id = %s ORDER BY confidence DESC",
        (session_id,)
    )
