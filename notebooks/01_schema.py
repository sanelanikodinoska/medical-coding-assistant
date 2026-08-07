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
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    call_id     SERIAL PRIMARY KEY,
    session_id  INTEGER REFERENCES coding_sessions(session_id) ON DELETE CASCADE,
    tool_name   TEXT NOT NULL,
    tool_input  JSONB,
    tool_output JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- ICD-10 lookup table for Postgres full-text search (populated by notebook 05)
CREATE TABLE IF NOT EXISTS icd10_lookup (
    code        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    tsvec       TSVECTOR GENERATED ALWAYS AS
                    (to_tsvector('english', description)) STORED
);

-- Add updated_at to existing tables FIRST (before indexes reference it)
ALTER TABLE coding_sessions  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
ALTER TABLE code_suggestions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();

CREATE INDEX IF NOT EXISTS icd10_lookup_tsvec_idx ON icd10_lookup USING GIN (tsvec);
CREATE INDEX IF NOT EXISTS icd10_lookup_desc_idx  ON icd10_lookup (description text_pattern_ops);

-- Indexes for CDC watermark queries (updated_at columns)
CREATE INDEX IF NOT EXISTS sessions_updated_at_idx    ON coding_sessions  (updated_at);
CREATE INDEX IF NOT EXISTS suggestions_updated_at_idx ON code_suggestions (updated_at);
CREATE INDEX IF NOT EXISTS sessions_session_id_idx    ON code_suggestions (session_id);
CREATE INDEX IF NOT EXISTS tool_calls_session_id_idx  ON agent_tool_calls (session_id);

-- Auto-update updated_at on row change
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$;

DROP TRIGGER IF EXISTS sessions_updated_at    ON coding_sessions;
DROP TRIGGER IF EXISTS suggestions_updated_at ON code_suggestions;

CREATE TRIGGER sessions_updated_at
    BEFORE UPDATE ON coding_sessions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER suggestions_updated_at
    BEFORE UPDATE ON code_suggestions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

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
