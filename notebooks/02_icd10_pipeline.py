# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — ICD-10 Pipeline
# MAGIC Fetches ~77,000 ICD-10-CM codes from the NLM API, generates embeddings,
# MAGIC and writes to a Delta table. Re-run every October when CMS releases new codes.

# COMMAND ----------

# Install sentence-transformers for embeddings
%pip install sentence-transformers requests --quiet
dbutils.library.restartPython()

# COMMAND ----------

import requests, json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
from sentence_transformers import SentenceTransformer

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md ## Step 1 — Fetch all ICD-10-CM codes from NLM API

# COMMAND ----------

BASE_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

def fetch_codes_for_prefix(prefix, max_results=500):
    """Fetch ICD-10 codes matching a prefix letter."""
    params = {
        "terms": prefix,
        "maxList": max_results,
        "df": "code,name"
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Response: [total, codes_list, extra, display_list]
    # display_list contains [[code, description], ...]
    items = data[3] if len(data) > 3 else []
    return [{"code": item[0], "description": item[1]} for item in items]

# Fetch codes across all letter prefixes A-Z
all_codes = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    try:
        codes = fetch_codes_for_prefix(letter)
        all_codes.extend(codes)
        print(f"  {letter}: {len(codes)} codes")
    except Exception as e:
        print(f"  {letter}: ERROR — {e}")

print(f"\nTotal codes fetched: {len(all_codes)}")

# COMMAND ----------

# MAGIC %md ## Step 2 — Generate embeddings on descriptions

# COMMAND ----------

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast, free

descriptions = [c["description"] for c in all_codes]

# Batch embed (process in chunks to avoid memory issues)
BATCH = 512
embeddings = []
for i in range(0, len(descriptions), BATCH):
    batch = descriptions[i:i+BATCH]
    embs  = model.encode(batch, show_progress_bar=False).tolist()
    embeddings.extend(embs)
    print(f"  Embedded {min(i+BATCH, len(descriptions))}/{len(descriptions)}")

for i, code in enumerate(all_codes):
    code["embedding"] = embeddings[i]

print("Embeddings done")

# COMMAND ----------

# MAGIC %md ## Step 3 — Write to Delta table

# COMMAND ----------

schema = StructType([
    StructField("code",        StringType(), False),
    StructField("description", StringType(), False),
    StructField("embedding",   ArrayType(FloatType()), True),
])

rows = [(c["code"], c["description"], c["embedding"]) for c in all_codes]
df   = spark.createDataFrame(rows, schema)

# Create catalog schema if it doesn't exist
spark.sql("CREATE SCHEMA IF NOT EXISTS main.medical_coding")

# Write (overwrite to get latest codes)
(df.write
   .format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable("main.medical_coding.icd10_codes"))

print(f"Written {df.count()} codes to main.medical_coding.icd10_codes")

# COMMAND ----------

# Quick check
spark.sql("SELECT code, description FROM main.medical_coding.icd10_codes LIMIT 5").show(truncate=False)
