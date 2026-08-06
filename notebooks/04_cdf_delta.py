# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — CDF: Lakebase → Delta
# MAGIC Reads coding sessions from Lakebase and writes them to a Delta table for analytics.
# MAGIC This satisfies the "Change Data Feed from Lakebase into a Delta table" requirement.
# MAGIC
# MAGIC Run this notebook after the app has some sessions, or schedule it to run hourly.

# COMMAND ----------

import base64, psycopg2, json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md ## Step 1 — Read current data from Lakebase

# COMMAND ----------

from databricks.sdk import WorkspaceClient

def get_conn():
    w = WorkspaceClient()
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    url = base64.b64decode(secret.value).decode("utf-8")
    return psycopg2.connect(url)

conn = get_conn()
cur  = conn.cursor()

# Read all sessions with their suggestions
cur.execute("""
    SELECT
        s.session_id,
        s.note_text,
        s.specialty,
        s.user_email,
        s.created_at,
        s.updated_at,
        COUNT(sg.suggestion_id)                                    AS total_suggestions,
        COUNT(sg.suggestion_id) FILTER (WHERE sg.accepted = true)  AS accepted_count,
        COUNT(sg.suggestion_id) FILTER (WHERE sg.accepted = false) AS rejected_count,
        ARRAY_AGG(sg.icd10_code ORDER BY sg.confidence DESC)       AS suggested_codes
    FROM coding_sessions s
    LEFT JOIN code_suggestions sg ON sg.session_id = s.session_id
    GROUP BY s.session_id, s.note_text, s.specialty, s.user_email, s.created_at, s.updated_at
    ORDER BY s.created_at DESC
""")
sessions = cur.fetchall()
cols = [d[0] for d in cur.description]

# Read tool call audit log
cur.execute("""
    SELECT call_id, session_id, tool_name, tool_input::text, tool_output::text, created_at
    FROM agent_tool_calls
    ORDER BY created_at DESC
""")
tool_calls = cur.fetchall()
tool_cols  = [d[0] for d in cur.description]

cur.close()
conn.close()

print(f"Sessions: {len(sessions)}")
print(f"Tool calls: {len(tool_calls)}")

# COMMAND ----------

# MAGIC %md ## Step 2 — Write sessions history to Delta

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS main.medical_coding")

if sessions:
    sessions_data = [dict(zip(cols, row)) for row in sessions]
    # Convert timestamps and arrays for Spark
    for row in sessions_data:
        row["created_at"] = str(row["created_at"])
        row["updated_at"] = str(row["updated_at"])
        row["suggested_codes"] = row["suggested_codes"] or []
        row["total_suggestions"] = int(row["total_suggestions"] or 0)
        row["accepted_count"]    = int(row["accepted_count"] or 0)
        row["rejected_count"]    = int(row["rejected_count"] or 0)

    df_sessions = spark.createDataFrame(sessions_data)

    (df_sessions.write
       .format("delta")
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable("main.medical_coding.sessions_history"))

    print(f"Written {len(sessions)} sessions to main.medical_coding.sessions_history")
else:
    print("No sessions yet — run the app first to create some sessions, then re-run this notebook")

# COMMAND ----------

# MAGIC %md ## Step 3 — Write tool call audit log to Delta

# COMMAND ----------

if tool_calls:
    tool_data = [dict(zip(tool_cols, row)) for row in tool_calls]
    for row in tool_data:
        row["created_at"] = str(row["created_at"])

    df_tools = spark.createDataFrame(tool_data)

    (df_tools.write
       .format("delta")
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable("main.medical_coding.agent_tool_calls_history"))

    print(f"Written {len(tool_calls)} tool calls to main.medical_coding.agent_tool_calls_history")

# COMMAND ----------

# MAGIC %md ## Step 4 — Analytics summary

# COMMAND ----------

# This is what the Delta table looks like for analytics
spark.sql("""
    SELECT
        specialty,
        COUNT(*) AS total_sessions,
        SUM(total_suggestions) AS total_codes_suggested,
        SUM(accepted_count) AS total_accepted,
        ROUND(SUM(accepted_count) * 100.0 / NULLIF(SUM(total_suggestions), 0), 1) AS acceptance_rate_pct
    FROM main.medical_coding.sessions_history
    GROUP BY specialty
    ORDER BY total_sessions DESC
""").show() if sessions else print("No data yet")
