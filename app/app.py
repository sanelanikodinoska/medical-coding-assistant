from flask import Flask, request, jsonify, render_template
import lakebase, agent as ag

app = Flask(__name__)

# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/healthz")
def healthz():
    return {"status": "ok"}

# ── Stats ─────────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    rows = lakebase.run_query("""
        SELECT
            COUNT(DISTINCT s.session_id)                                          AS total_sessions,
            COUNT(sg.suggestion_id)                                               AS total_suggestions,
            COUNT(sg.suggestion_id) FILTER (WHERE sg.accepted = true)             AS accepted,
            COUNT(sg.suggestion_id) FILTER (WHERE sg.accepted = false)            AS rejected,
            COUNT(DISTINCT s.specialty)                                           AS specialties
        FROM coding_sessions s
        LEFT JOIN code_suggestions sg ON sg.session_id = s.session_id
    """)
    return jsonify(rows[0] if rows else {})

# ── Sessions ──────────────────────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    specialty = request.args.get("specialty")
    limit     = int(request.args.get("limit", 50))

    if specialty:
        rows = lakebase.run_query("""
            SELECT s.session_id, s.specialty, s.user_email, s.created_at,
                   LEFT(s.note_text, 120) AS note_preview,
                   COUNT(sg.suggestion_id) AS suggestion_count
            FROM coding_sessions s
            LEFT JOIN code_suggestions sg ON sg.session_id = s.session_id
            WHERE s.specialty = %s
            GROUP BY s.session_id, s.specialty, s.user_email, s.created_at, s.note_text
            ORDER BY s.created_at DESC LIMIT %s
        """, (specialty, limit))
    else:
        rows = lakebase.run_query("""
            SELECT s.session_id, s.specialty, s.user_email, s.created_at,
                   LEFT(s.note_text, 120) AS note_preview,
                   COUNT(sg.suggestion_id) AS suggestion_count
            FROM coding_sessions s
            LEFT JOIN code_suggestions sg ON sg.session_id = s.session_id
            GROUP BY s.session_id, s.specialty, s.user_email, s.created_at, s.note_text
            ORDER BY s.created_at DESC LIMIT %s
        """, (limit,))
    return jsonify(rows)

@app.route("/api/sessions/<int:sid>", methods=["GET"])
def get_session(sid):
    sessions = lakebase.run_query(
        "SELECT * FROM coding_sessions WHERE session_id = %s", (sid,))
    if not sessions:
        return jsonify({"error": "Not found"}), 404
    session = sessions[0]
    session["suggestions"] = lakebase.run_query(
        "SELECT * FROM code_suggestions WHERE session_id = %s ORDER BY confidence DESC", (sid,))
    session["tool_calls"] = lakebase.run_query(
        "SELECT tool_name, created_at FROM agent_tool_calls WHERE session_id = %s ORDER BY created_at", (sid,))
    return jsonify(session)

@app.route("/api/sessions/<int:sid>", methods=["DELETE"])
def delete_session(sid):
    lakebase.run_write("DELETE FROM coding_sessions WHERE session_id = %s", (sid,))
    return jsonify({"deleted": sid})

# ── Code a note (main AI action) ──────────────────────────────────────────────

@app.route("/api/code", methods=["POST"])
def code_note():
    data      = request.get_json(force=True)
    note_text = (data.get("note_text") or "").strip()
    specialty = (data.get("specialty") or "").strip() or None
    email     = (data.get("user_email") or "").strip() or None

    if not note_text:
        return jsonify({"error": "note_text is required"}), 400
    if len(note_text) < 20:
        return jsonify({"error": "Note is too short to code"}), 400

    # Create session
    session = lakebase.run_write_returning("""
        INSERT INTO coding_sessions (note_text, specialty, user_email)
        VALUES (%s, %s, %s) RETURNING *
    """, (note_text, specialty, email))

    # Run AI agent
    try:
        suggestions = ag.run_agent(session["session_id"], note_text, specialty)
        session["suggestions"] = suggestions
        return jsonify(session)
    except Exception as e:
        return jsonify({"error": str(e), "session": session}), 500

# ── Quick ICD-10 search (no AI, instant) ─────────────────────────────────────

@app.route("/api/search")
def search_codes():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify([])
    results = ag.search_icd10_codes(q, max_results=10)
    return jsonify(results)

# ── Accept / reject a suggestion ─────────────────────────────────────────────

@app.route("/api/suggestions/<int:sid>/accept", methods=["POST"])
def accept_suggestion(sid):
    lakebase.run_write(
        "UPDATE code_suggestions SET accepted = true  WHERE suggestion_id = %s", (sid,))
    return jsonify({"accepted": sid})

@app.route("/api/suggestions/<int:sid>/reject", methods=["POST"])
def reject_suggestion(sid):
    lakebase.run_write(
        "UPDATE code_suggestions SET accepted = false WHERE suggestion_id = %s", (sid,))
    return jsonify({"rejected": sid})

# ── Frontend ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
