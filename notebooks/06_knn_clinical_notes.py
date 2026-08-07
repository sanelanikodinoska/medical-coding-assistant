# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 06 — KNN Retrieval over Clinical Notes Embeddings
# MAGIC
# MAGIC Demonstrates approximate nearest-neighbour search over the
# MAGIC `workspace.medical_coding.clinical_notes` Delta table using
# MAGIC the 384-dim sentence-transformer embeddings stored at pipeline time.
# MAGIC
# MAGIC This pattern is also used by the agent's `retrieve_similar_notes` tool:
# MAGIC the agent calls this logic (via the Databricks SQL warehouse) to ground
# MAGIC ICD-10 suggestions in historically similar clinical cases.

# COMMAND ----------
# MAGIC %pip install sentence-transformers --quiet

# COMMAND ----------
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from sentence_transformers import SentenceTransformer

CATALOG = "workspace"
SCHEMA  = "medical_coding"
MODEL   = "all-MiniLM-L6-v2"   # same model used in notebook 03

# COMMAND ----------
# MAGIC %md ### 1 — Load clinical notes with embeddings from Delta

# COMMAND ----------
df = spark.table(f"{CATALOG}.{SCHEMA}.clinical_notes") \
          .select("specialty", "note_text", "embedding")

print(f"Clinical notes: {df.count()}")
df.select("specialty").show(truncate=False)

# COMMAND ----------
# MAGIC %md ### 2 — Embed the query note

# COMMAND ----------
# Sample query — replace with any clinical note text
QUERY_NOTE = """
CHIEF COMPLAINT: Chest pain and shortness of breath.
HISTORY: 67-year-old male with hypertension and type 2 diabetes presenting
with substernal chest pain radiating to the left arm, onset 2 hours ago.
ASSESSMENT: 1. Acute inferior STEMI  2. Hypertension  3. Type 2 diabetes mellitus
"""

model = SentenceTransformer(MODEL)
query_embedding = model.encode(QUERY_NOTE).tolist()
print(f"Query embedding shape: {len(query_embedding)}-dim")

# COMMAND ----------
# MAGIC %md ### 3 — Compute cosine similarity with Spark UDF

# COMMAND ----------
# Broadcast query so every executor gets a copy
query_bc = sc.broadcast(query_embedding)

@F.udf(FloatType())
def cosine_similarity(stored_embedding):
    """Cosine similarity between query and stored embedding."""
    if stored_embedding is None:
        return 0.0
    q = np.array(query_bc.value, dtype=np.float32)
    v = np.array(stored_embedding, dtype=np.float32)
    denom = np.linalg.norm(q) * np.linalg.norm(v)
    return float(np.dot(q, v) / denom) if denom > 0 else 0.0

# COMMAND ----------
# MAGIC %md ### 4 — Retrieve top-K most similar notes (KNN)

# COMMAND ----------
K = 5

results = (
    df.withColumn("similarity", cosine_similarity(F.col("embedding")))
      .orderBy(F.col("similarity").desc())
      .limit(K)
      .select("specialty", "similarity",
              F.expr("LEFT(note_text, 200)").alias("note_preview"))
)

print(f"\nTop-{K} most similar clinical notes to query:")
results.show(K, truncate=False)

# COMMAND ----------
# MAGIC %md ### 5 — How the agent uses this
# MAGIC
# MAGIC In `app/agent.py`, the `retrieve_similar_notes` tool:
# MAGIC 1. Calls the Databricks SQL warehouse to fetch stored embeddings
# MAGIC 2. Generates a query embedding using the same `all-MiniLM-L6-v2` model
# MAGIC 3. Computes cosine similarity in NumPy (only ~20 notes → fast in memory)
# MAGIC 4. Returns the top-K notes with their specialties so the LLM can ground
# MAGIC    ICD-10 suggestions in historically similar cases

# COMMAND ----------
# MAGIC %md ### 6 — Validate: embeddings are 384-dim and non-zero

# COMMAND ----------
spark.sql(f"""
    SELECT
        COUNT(*)             AS total_notes,
        COUNT(embedding)     AS with_embedding,
        SIZE(embedding)      AS embedding_dim
    FROM {CATALOG}.{SCHEMA}.clinical_notes
""").show()
