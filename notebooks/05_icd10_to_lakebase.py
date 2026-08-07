# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05 — ICD-10 Codes → Lakebase (Full-Text Search)
# MAGIC
# MAGIC Reads ICD-10 codes from Delta (`workspace.medical_coding.icd10_codes`)
# MAGIC and loads them into a Postgres `icd10_lookup` table in Lakebase.
# MAGIC Enables `tsvector`-based full-text search so the agent can use
# MAGIC `semantic_search_icd10` as a complement to the NLM API.
# MAGIC
# MAGIC Run once after notebook 02 has populated the Delta table.
# MAGIC Safe to re-run: uses TRUNCATE + INSERT for idempotency.

# COMMAND ----------
import base64, psycopg2
from psycopg2.extras import execute_batch
from databricks.sdk import WorkspaceClient

# ── Lakebase connection ───────────────────────────────────────────────────────
w      = WorkspaceClient()
secret = w.secrets.get_secret(scope="database", key="lakebase-url")
db_url = base64.b64decode(secret.value).decode("utf-8")

conn = psycopg2.connect(db_url, connect_timeout=30)
conn.autocommit = False
cur  = conn.cursor()

# COMMAND ----------
# MAGIC %md ### 1 — Create table with FTS index (idempotent)

# COMMAND ----------
cur.execute("""
    CREATE TABLE IF NOT EXISTS icd10_lookup (
        code        TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        category    TEXT,
        tsvec       TSVECTOR GENERATED ALWAYS AS
                        (to_tsvector('english', description)) STORED
    )
""")

# GIN index for fast full-text search
cur.execute("""
    CREATE INDEX IF NOT EXISTS icd10_lookup_tsvec_idx
    ON icd10_lookup USING GIN (tsvec)
""")

conn.commit()
print("Table and index ready.")

# COMMAND ----------
# MAGIC %md ### 2 — Load ICD-10 codes from Delta

# COMMAND ----------
df = spark.table("workspace.medical_coding.icd10_codes") \
          .select("code", "description") \
          .dropDuplicates(["code"])

rows   = df.collect()
total  = len(rows)
print(f"Loaded {total:,} codes from Delta.")

# COMMAND ----------
# MAGIC %md ### 3 — Upsert into Lakebase

# COMMAND ----------
BATCH_SIZE = 1000

data = [(r["code"], r["description"] or "")
        for r in rows]

# Truncate first for clean reload (safe because Delta is the source of truth)
cur.execute("TRUNCATE TABLE icd10_lookup")

for i in range(0, total, BATCH_SIZE):
    batch = data[i : i + BATCH_SIZE]
    execute_batch(cur, """
        INSERT INTO icd10_lookup (code, description)
        VALUES (%s, %s)
        ON CONFLICT (code) DO UPDATE
            SET description = EXCLUDED.description
    """, batch)
    conn.commit()
    print(f"  Inserted {min(i + BATCH_SIZE, total):,}/{total:,}")

# COMMAND ----------
# MAGIC %md ### 4 — Verify

# COMMAND ----------
cur.execute("SELECT COUNT(*) FROM icd10_lookup")
count = cur.fetchone()[0]
print(f"icd10_lookup rows: {count:,}")

# Quick FTS test
cur.execute("""
    SELECT code, description
    FROM icd10_lookup
    WHERE tsvec @@ plainto_tsquery('english', 'pneumonia')
    ORDER BY ts_rank(tsvec, plainto_tsquery('english', 'pneumonia')) DESC
    LIMIT 5
""")
print("\nTop 5 FTS results for 'pneumonia':")
for row in cur.fetchall():
    print(f"  {row[0]}  {row[1]}")

cur.close()
conn.close()
