# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04 — Lakebase → Delta (Incremental CDC)
# MAGIC
# MAGIC Implements a true incremental change-capture pipeline:
# MAGIC
# MAGIC 1. Reads only rows *newer than* the last watermark from Lakebase
# MAGIC 2. MERGEs new/changed rows into Delta target tables
# MAGIC 3. Delta Change Data Feed (CDF) is enabled so downstream consumers
# MAGIC    can read only changed rows with `readChangeFeed`
# MAGIC 4. A watermark table persists the high-water timestamp between runs
# MAGIC
# MAGIC Run on a schedule (e.g., every 15 min via Databricks Workflows).

# COMMAND ----------
import base64, psycopg2
from psycopg2.extras import RealDictCursor
from pyspark.sql import functions as F
from databricks.sdk import WorkspaceClient

CATALOG = "workspace"
SCHEMA  = "medical_coding"

# ── Lakebase connection ───────────────────────────────────────────────────────
w      = WorkspaceClient()
secret = w.secrets.get_secret(scope="database", key="lakebase-url")
db_url = base64.b64decode(secret.value).decode("utf-8")

conn = psycopg2.connect(db_url, connect_timeout=30)
cur  = conn.cursor(cursor_factory=RealDictCursor)

# COMMAND ----------
# MAGIC %md ### 1 — Enable Delta CDF on target tables (idempotent)

# COMMAND ----------
for table in ("sessions_history", "suggestions_history"):
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.{table}
        USING DELTA
        TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)

# sessions_history
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.sessions_history
    (
        session_id   BIGINT,
        specialty    STRING,
        user_email   STRING,
        note_text    STRING,
        created_at   TIMESTAMP,
        updated_at   TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

# suggestions_history
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.suggestions_history
    (
        suggestion_id BIGINT,
        session_id    BIGINT,
        icd10_code    STRING,
        description   STRING,
        confidence    DOUBLE,
        explanation   STRING,
        accepted      BOOLEAN,
        created_at    TIMESTAMP,
        updated_at    TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

# Watermark table — persists last-seen timestamp per source table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.cdc_watermarks
    (
        source_table STRING,
        watermark    TIMESTAMP
    )
    USING DELTA
""")

print("Delta tables ready.")

# COMMAND ----------
# MAGIC %md ### 2 — Helper: get/set watermark

# COMMAND ----------
from datetime import datetime, timezone

def get_watermark(source_table: str) -> str:
    rows = spark.sql(f"""
        SELECT watermark FROM {CATALOG}.{SCHEMA}.cdc_watermarks
        WHERE source_table = '{source_table}'
    """).collect()
    if rows:
        return rows[0]["watermark"].isoformat()
    return "1970-01-01T00:00:00+00:00"   # first run: get everything

def set_watermark(source_table: str, ts: str):
    spark.sql(f"""
        MERGE INTO {CATALOG}.{SCHEMA}.cdc_watermarks AS t
        USING (SELECT '{source_table}' AS source_table,
                      CAST('{ts}' AS TIMESTAMP) AS watermark) AS s
        ON t.source_table = s.source_table
        WHEN MATCHED THEN UPDATE SET t.watermark = s.watermark
        WHEN NOT MATCHED THEN INSERT (source_table, watermark)
                              VALUES (s.source_table, s.watermark)
    """)

# COMMAND ----------
# MAGIC %md ### 3 — Incremental sync: coding_sessions → sessions_history

# COMMAND ----------
wm_sessions = get_watermark("coding_sessions")
print(f"Sessions watermark: {wm_sessions}")

cur.execute("""
    SELECT session_id, specialty, user_email, note_text,
           created_at AT TIME ZONE 'UTC' AS created_at,
           COALESCE(updated_at, created_at) AT TIME ZONE 'UTC' AS updated_at
    FROM coding_sessions
    WHERE COALESCE(updated_at, created_at) > %s
    ORDER BY updated_at
""", (wm_sessions,))
new_sessions = cur.fetchall()
print(f"  New/changed rows: {len(new_sessions)}")

if new_sessions:
    df_sessions = spark.createDataFrame(
        [dict(r) for r in new_sessions],
        schema="session_id BIGINT, specialty STRING, user_email STRING, "
               "note_text STRING, created_at TIMESTAMP, updated_at TIMESTAMP"
    )
    df_sessions.createOrReplaceTempView("_new_sessions")

    spark.sql(f"""
        MERGE INTO {CATALOG}.{SCHEMA}.sessions_history AS t
        USING _new_sessions AS s ON t.session_id = s.session_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT  *
    """)

    max_ts = max(r["updated_at"] for r in new_sessions)
    set_watermark("coding_sessions", max_ts.isoformat())
    print(f"  Watermark updated to {max_ts}")
else:
    print("  Nothing new — skipping.")

# COMMAND ----------
# MAGIC %md ### 4 — Incremental sync: code_suggestions → suggestions_history

# COMMAND ----------
wm_sugg = get_watermark("code_suggestions")
print(f"Suggestions watermark: {wm_sugg}")

cur.execute("""
    SELECT suggestion_id, session_id, icd10_code, description,
           confidence, explanation, accepted,
           created_at AT TIME ZONE 'UTC' AS created_at,
           COALESCE(updated_at, created_at) AT TIME ZONE 'UTC' AS updated_at
    FROM code_suggestions
    WHERE COALESCE(updated_at, created_at) > %s
    ORDER BY updated_at
""", (wm_sugg,))
new_sugg = cur.fetchall()
print(f"  New/changed rows: {len(new_sugg)}")

if new_sugg:
    df_sugg = spark.createDataFrame(
        [dict(r) for r in new_sugg],
        schema="suggestion_id BIGINT, session_id BIGINT, icd10_code STRING, "
               "description STRING, confidence DOUBLE, explanation STRING, "
               "accepted BOOLEAN, created_at TIMESTAMP, updated_at TIMESTAMP"
    )
    df_sugg.createOrReplaceTempView("_new_sugg")

    spark.sql(f"""
        MERGE INTO {CATALOG}.{SCHEMA}.suggestions_history AS t
        USING _new_sugg AS s ON t.suggestion_id = s.suggestion_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT  *
    """)

    max_ts = max(r["updated_at"] for r in new_sugg)
    set_watermark("code_suggestions", max_ts.isoformat())
    print(f"  Watermark updated to {max_ts}")
else:
    print("  Nothing new — skipping.")

cur.close()
conn.close()

# COMMAND ----------
# MAGIC %md ### 5 — Analytics query using Change Data Feed

# COMMAND ----------
# Read only rows that changed since the last CDC version
# (demonstrates CDF — downstream jobs can use this pattern)
from delta.tables import DeltaTable

dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.sessions_history")
latest_version = dt.history(1).collect()[0]["version"]

if latest_version >= 1:
    changes = spark.read \
        .format("delta") \
        .option("readChangeFeed", "true") \
        .option("startingVersion", max(0, latest_version - 5)) \
        .table(f"{CATALOG}.{SCHEMA}.sessions_history")

    print(f"CDF — rows changed in last 5 versions: {changes.count()}")
    changes.select("session_id", "specialty", "_change_type", "_commit_timestamp") \
           .show(20, truncate=False)

# Analytics summary
spark.sql(f"""
    SELECT specialty,
           COUNT(*)                       AS total_sessions,
           COUNT(DISTINCT user_email)     AS unique_users,
           ROUND(AVG(code_count), 1)      AS avg_codes
    FROM (
        SELECT s.session_id, s.specialty, s.user_email,
               COUNT(sg.suggestion_id) AS code_count
        FROM {CATALOG}.{SCHEMA}.sessions_history s
        LEFT JOIN {CATALOG}.{SCHEMA}.suggestions_history sg
               ON sg.session_id = s.session_id
        GROUP BY s.session_id, s.specialty, s.user_email
    )
    GROUP BY specialty
    ORDER BY total_sessions DESC
""").show(20, truncate=False)
