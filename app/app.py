"""
Medical Coding Assistant — Flask application.

Improvements over v1:
- Global JSON error handler (no more HTML 500 pages)
- ValueError guard on limit parameter
- /api/analytics endpoint reading from Delta via Databricks SQL
- Blueprint-ready structure (single file for now)
"""

import json, os
import lakebase
from flask import Flask, jsonify, render_template, request, abort
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# ── Global JSON error handler (API routes only) ───────────────────────────────

@app.errorhandler(HTTPException)
def handle_http_error(e):
    # Only return JSON for /api/* routes; let HTML pages propagate normally
    if request.path.startswith("/api/"):
        return jsonify(error=e.description), e.code
    return e

@app.errorhandler(500)
def handle_500(e):
    return jsonify(error="Internal server error"), 500

# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/healthz")
def health():
    return jsonify(status="ok")

# ── Stats ─────────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    rows = lakebase.run_query("""
        SELECT
            (SELECT COUNT(*) FROM coding_sessions)                                    AS total_sessions,
            (SELECT COUNT(*) FROM code_suggestions)                                   AS total_suggestions,
            (SELECT COUNT(*) FROM code_suggestions WHERE accepted = TRUE)             AS accepted,
            (SELECT COUNT(*) FROM code_suggestions WHERE accepted = FALSE)            AS rejected,
            (SELECT COUNT(DISTINCT specialty) FROM coding_sessions WHERE specialty IS NOT NULL) AS specialties
    """)
    return jsonify(dict(rows[0])) if rows else jsonify({})

# ── Sessions ──────────────────────────────────────────────────────────────────

@app.route("/api/sessions")
def sessions():
    try:
        limit = int(request.args.get("limit", 50))
        limit = max(1, min(limit, 500))          # clamp to [1, 500]
    except (ValueError, TypeError):
        limit = 50

    rows = lakebase.run_query("""
        SELECT s.session_id, s.specialty, s.user_email,
               s.created_at::text,
               COUNT(sg.suggestion_id)                       AS code_count,
               COUNT(sg.suggestion_id) FILTER
                   (WHERE sg.accepted = TRUE)                AS accepted_count
        FROM coding_sessions s
        LEFT JOIN code_suggestions sg USING (session_id)
        GROUP BY s.session_id, s.specialty, s.user_email, s.created_at
        ORDER BY s.created_at DESC
        LIMIT %s
    """, (limit,))
    return jsonify([dict(r) for r in rows])

@app.route("/api/sessions/<int:session_id>")
def get_session(session_id):
    rows = lakebase.run_query(
        "SELECT * FROM coding_sessions WHERE session_id = %s", (session_id,)
    )
    if not rows:
        abort(404)
    session = dict(rows[0])

    suggestions = lakebase.run_query(
        "SELECT * FROM code_suggestions WHERE session_id = %s ORDER BY confidence DESC",
        (session_id,)
    )
    session["suggestions"] = [dict(r) for r in suggestions]

    tool_calls = lakebase.run_query(
        "SELECT * FROM agent_tool_calls WHERE session_id = %s ORDER BY created_at",
        (session_id,)
    )
    session["tool_calls"] = [dict(r) for r in tool_calls]

    return jsonify(session)

@app.route("/api/sessions/<int:session_id>", methods=["DELETE"])
def delete_session(session_id):
    lakebase.run_write(
        "DELETE FROM code_suggestions WHERE session_id = %s", (session_id,)
    )
    lakebase.run_write(
        "DELETE FROM agent_tool_calls WHERE session_id = %s", (session_id,)
    )
    lakebase.run_write(
        "DELETE FROM coding_sessions WHERE session_id = %s", (session_id,)
    )
    return jsonify(deleted=True)

# ── Code a note ───────────────────────────────────────────────────────────────

@app.route("/api/code", methods=["POST"])
def code_note():
    import agent
    data      = request.get_json(force=True)
    note_text = (data.get("note_text") or "").strip()
    specialty = data.get("specialty")
    email     = data.get("user_email", "anonymous@app")

    if not note_text:
        abort(400, description="note_text is required")

    # Create session (must commit before agent references session_id)
    session = lakebase.run_write_returning("""
        INSERT INTO coding_sessions (specialty, user_email, note_text)
        VALUES (%s, %s, %s) RETURNING *
    """, (specialty, email, note_text))
    session_id = session["session_id"]

    try:
        suggestions = agent.run_agent(session_id, note_text, specialty, email)
        session["suggestions"] = suggestions
        return jsonify(session)
    except Exception as e:
        return jsonify({"error": str(e), "session": session}), 500

# ── Suggestion accept / reject ────────────────────────────────────────────────

@app.route("/api/suggestions/<int:suggestion_id>/accept", methods=["POST"])
def accept(suggestion_id):
    lakebase.run_write(
        "UPDATE code_suggestions SET accepted = TRUE  WHERE suggestion_id = %s",
        (suggestion_id,)
    )
    return jsonify(updated=True)

@app.route("/api/suggestions/<int:suggestion_id>/reject", methods=["POST"])
def reject(suggestion_id):
    lakebase.run_write(
        "UPDATE code_suggestions SET accepted = FALSE WHERE suggestion_id = %s",
        (suggestion_id,)
    )
    return jsonify(updated=True)

# ── ICD-10 search (proxy / fallback) ─────────────────────────────────────────

@app.route("/api/search")
def search():
    import requests as req
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    try:
        limit = int(request.args.get("limit", 10))
        limit = max(1, min(limit, 50))
    except (ValueError, TypeError):
        limit = 10

    r = req.get(
        "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
        params={"terms": query, "maxList": limit, "df": "code,name", "sf": "code,name"},
        timeout=8
    )
    r.raise_for_status()
    data  = r.json()
    items = data[3] if len(data) > 3 else []
    return jsonify([{"code": i[0], "description": i[1]} for i in items])

# ── Semantic search (Lakebase FTS) ────────────────────────────────────────────

@app.route("/api/semantic-search")
def semantic_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    try:
        limit = int(request.args.get("limit", 10))
        limit = max(1, min(limit, 50))
    except (ValueError, TypeError):
        limit = 10

    rows = lakebase.run_query("""
        SELECT code, description,
               ts_rank(to_tsvector('english', description),
                       plainto_tsquery('english', %s)) AS rank
        FROM icd10_lookup
        WHERE to_tsvector('english', description) @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
    """, (query, query, limit))

    if not rows:
        rows = lakebase.run_query("""
            SELECT code, description, 0.0 AS rank
            FROM icd10_lookup
            WHERE description ILIKE %s
            ORDER BY code
            LIMIT %s
        """, (f"%{query}%", limit))

    return jsonify([{"code": r["code"], "description": r["description"]} for r in rows])

# ── Analytics (reads from Delta via Databricks SQL) ───────────────────────────

@app.route("/api/analytics")
def analytics():
    """
    Returns aggregated trends from the Delta analytics table (written by notebook 04).
    Falls back to Lakebase live counts if Delta isn't available.
    """
    try:
        from databricks import sql as dbsql
        import base64
        from databricks.sdk import WorkspaceClient
        w   = WorkspaceClient()
        sec = w.secrets.get_secret(scope="database", key="databricks-token")
        tok = base64.b64decode(sec.value).decode("utf-8")

        warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
        if not warehouse_id:
            raise ValueError("DATABRICKS_WAREHOUSE_ID not set")

        with dbsql.connect(
            server_hostname = w.config.host.replace("https://", ""),
            http_path       = f"/sql/1.0/warehouses/{warehouse_id}",
            access_token    = tok,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT specialty,
                           COUNT(*) AS session_count,
                           AVG(code_count) AS avg_codes_per_session
                    FROM workspace.medical_coding.sessions_history
                    GROUP BY specialty
                    ORDER BY session_count DESC
                    LIMIT 20
                """)
                rows = [dict(zip([d[0] for d in cur.description], row))
                        for row in cur.fetchall()]
        return jsonify({"source": "delta", "data": rows})

    except Exception as e:
        # Graceful fallback to Lakebase live data
        rows = lakebase.run_query("""
            SELECT specialty,
                   COUNT(*) AS session_count,
                   ROUND(AVG(code_count), 1) AS avg_codes_per_session
            FROM (
                SELECT s.session_id, s.specialty,
                       COUNT(sg.suggestion_id) AS code_count
                FROM coding_sessions s
                LEFT JOIN code_suggestions sg USING (session_id)
                GROUP BY s.session_id, s.specialty
            ) t
            GROUP BY specialty
            ORDER BY session_count DESC
            LIMIT 20
        """)
        return jsonify({"source": "lakebase", "data": [dict(r) for r in rows]})

# ── Frontend ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
