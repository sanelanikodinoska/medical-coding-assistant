# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Lakebase Schema
# MAGIC Creates all operational tables in Lakebase. Run this ONCE before anything else.

# COMMAND ----------

import base64, psycopg2
from databricks.sdk import WorkspaceClient

def get_conn():
    w = WorkspaceClient()
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    url = base64.b64decode(secret.value).decode("utf-8")
    return psycopg2.connect(url)

# COMMAND ----------

DDL = """
CREATE TABLE IF NOT EXISTS coding_sessions (
    session_id  SERIAL PRIMARY KEY,
    note_text   TEXT NOT NULL,
    specialty   TEXT,
    user_email  TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS code_suggestions (
    suggestion_id SERIAL PRIMARY KEY,
    session_id    INTEGER REFERENCES coding_sessions(session_id) ON DELETE CASCADE,
    icd10_code    TEXT NOT NULL,
    description   TEXT NOT NULL,
    confidence    NUMERIC(4,3),
    explanation   TEXT,
    accepted      BOOLEAN DEFAULT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    call_id     SERIAL PRIMARY KEY,
    session_id  INTEGER REFERENCES coding_sessions(session_id) ON DELETE CASCADE,
    tool_name   TEXT NOT NULL,
    tool_input  JSONB,
    tool_output JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE coding_sessions  REPLICA IDENTITY FULL;
ALTER TABLE code_suggestions REPLICA IDENTITY FULL;
"""

conn = get_conn()
cur  = conn.cursor()
cur.execute(DDL)
conn.commit()
cur.close()
conn.close()
print("Tables created")

# COMMAND ----------

# Verify tables exist
conn = get_conn()
cur  = conn.cursor()
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public' ORDER BY table_name
""")
print("Tables in Lakebase:", [r[0] for r in cur.fetchall()])
cur.close()
conn.close()
