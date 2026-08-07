# AI Medical Coding Assistant

**[🚀 Open Live App](https://medical-coding-assistant-7474643859693768.aws.databricksapps.com/)**

## Demo

![Demo](AI_Medical_Coding_Assistant_Demo-ezgif.com-optimize.gif)

An AI-powered ICD-10-CM coding assistant that reads free-form clinical notes and suggests the correct diagnosis codes with explanations. Built by a certified medical coder using Databricks, Lakebase, and Llama 3.

---

## Capstone Requirements

| # | Requirement | Implementation | Evidence |
|---|---|---|---|
| 1 | ✅ Spark data pipeline | `notebooks/02_icd10_pipeline.py` — NLM API → 12,316 ICD-10 codes → Delta; `03_mtsamples_pipeline.py` — 20 clinical notes + 384-dim embeddings → Delta | Delta tables in `workspace.medical_coding` |
| 2 | ✅ Third-party API | [NLM Clinical Tables API](https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search) — free, no key; `sf=code,name` searches descriptions; retry/backoff with 429 handling | `agent.py: search_icd10_codes`, `_with_backoff` |
| 3 | ✅ Unstructured data + embeddings + retrieval | `03_mtsamples_pipeline.py` generates 384-dim embeddings; `06_knn_clinical_notes.py` demonstrates KNN with cosine similarity UDF over the Delta embeddings table; `agent.py: retrieve_similar_notes` queries the Delta table via SQL warehouse and computes cosine similarity in NumPy at request time | See screenshot below |
| 4 | ✅ Databricks App | Flask + Alpine.js + Tailwind CSS, live at link above | Live URL |
| 5 | ✅ AI agent with tools (read + write) | Llama 3.3 70B with 5 tools: `retrieve_similar_notes` (KNN read), `search_icd10_codes` (NLM read), `semantic_search_icd10` (Postgres FTS read), `get_session_history` (read), `save_suggestions` (write) + full audit trail in `agent_tool_calls` | `agent.py` |
| 6 | ✅ CDF → Delta (incremental) | `04_cdf_delta.py` — watermark-based incremental MERGE from Lakebase into Delta; `delta.enableChangeDataFeed=true` on both target tables; `cdc_watermarks` table persists high-water timestamp between runs | See screenshot below |

---

## Architecture

```
NLM ICD-10 API          MTSamples Clinical Notes
      │                          │
      └──────── Spark Pipeline ──┘
                     │
         Delta Tables (workspace.medical_coding)
         ├── icd10_codes          (12,316 codes + 384-dim embeddings)
         ├── clinical_notes       (20 notes + 384-dim embeddings)  ← KNN source
         ├── sessions_history     (CDF enabled, watermark incremental MERGE)
         ├── suggestions_history  (CDF enabled, watermark incremental MERGE)
         └── cdc_watermarks       (high-water timestamp per source table)
                     │
         Lakebase (Postgres — databricks_ai_bootcamp_postgres)
         ├── coding_sessions      (REPLICA IDENTITY FULL, updated_at trigger)
         ├── code_suggestions     (REPLICA IDENTITY FULL, updated_at trigger)
         ├── agent_tool_calls     (full audit trail)
         └── icd10_lookup         (12,316 codes + GIN tsvector index for FTS)
                     │
         Databricks App (Flask, port 8000)
                     │
         AI Agent (Llama 3.3 70B — Databricks Foundation Models)
         Tools:
         ├── retrieve_similar_notes  → KNN over clinical_notes Delta embeddings
         ├── search_icd10_codes      → NLM API (retry/backoff)
         ├── semantic_search_icd10   → Postgres FTS over icd10_lookup
         ├── get_session_history     → past accepted codes per user
         └── save_suggestions        → persist to Lakebase + audit log
```

---

## Screenshots

### App — Code a Note & Suggestions
![Code suggestions](images/Code_suggestions.png)
![7th character warning](images/7th_char_warn.png)
![Session details](images/Session_details.png)

### App — Session Detail & Agent Audit Trail
![Agent tool calling](images/Agent_tool_calling.png)
![Audit trail](images/Audit_trails_agent_calls.png)
![Session app calling LLM](images/Session_app_calling_llm.png)

### App — Semantic Search & Save Suggestions
![Semantic search tool](images/Semantic_search_tool.png)
![Semantic search and save](images/Semantic_search_tool_and_save_suggestion.png)

### Code Lookup
![Search code result](images/Search_code_result.png)
![AI search by note](images/AI_search_by_note.png)
![AI search by note result](images/AI_search_by_note_result.png)

### KNN Retrieval over Clinical Notes Embeddings (Notebook 06)
![KNN output](images/KNN_output.png)

### ICD-10 Lakebase Full-Text Search
![icd10_lookup FTS result](<images/icd10_lookup FTS result.png>)

### CDF / Incremental CDC (Notebook 04)
![CDF enabled — DESCRIBE HISTORY](images/CDF_enabled_describe_history.png)
![CDF incremental output](images/CDF_incremental.png)
![CDF improvement](images/CDF_improvment.png)
![CDF Delta tables](images/CDF_delta_tables.png)

### Unity Catalog — workspace.medical_coding Tables
![Show tables in catalog](images/Show_tables_in_catalog.png)

### Coding Sessions
![Coding sessions](images/Coding_sessions.png)

---

## Key Implementation Details

### Catalog and Schema
All Delta tables use **`workspace.medical_coding`** — this is the only catalog available in this workspace (verified: `workspace`, `dbacademy`, `samples`, `system`). Every notebook uses `CATALOG = "workspace"`.

Validation query:
```sql
SHOW TABLES IN workspace.medical_coding
-- icd10_codes, clinical_notes, sessions_history, suggestions_history, cdc_watermarks
```

### Secrets — base64 encoding
Databricks Secret API always returns values **base64-encoded**, regardless of how they were stored. Storing with `string_value` is correct; the SDK wraps the value in base64 on retrieval. Both `lakebase.py` and `agent.py` decode correctly:

```python
# Stored:
w.secrets.put_secret(scope="database", key="lakebase-url", string_value="postgresql://...")

# Retrieved (SDK returns base64-encoded bytes):
secret = w.secrets.get_secret(scope="database", key="lakebase-url")
url = base64.b64decode(secret.value).decode("utf-8")   # ← correct
```

### CDF / Incremental CDC
`04_cdf_delta.py` implements true incremental change capture:
- `delta.enableChangeDataFeed = true` on `sessions_history` and `suggestions_history`
- Watermark stored in `cdc_watermarks` Delta table — only rows newer than the last run are fetched from Lakebase
- Uses `MERGE` (not overwrite) to upsert changed rows
- Reads CDF with `readChangeFeed` to show changed rows per run

```sql
-- CDF enabled (run DESCRIBE HISTORY workspace.medical_coding.sessions_history to verify):
CREATE OR REPLACE TABLE workspace.medical_coding.sessions_history (...)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
```

### KNN Retrieval over Clinical Notes Embeddings
The agent's `retrieve_similar_notes` tool performs cosine-similarity KNN over the 384-dim embeddings stored in `workspace.medical_coding.clinical_notes`:

1. Query embeddings from Delta via Databricks SQL warehouse
2. Embed the incoming note with `all-MiniLM-L6-v2` (same model used at ingest)
3. Compute cosine similarity with NumPy in memory
4. Return top-K similar cases to ground the agent's coding decisions

The standalone Spark version is in `notebooks/06_knn_clinical_notes.py` (cosine similarity UDF over the full Delta table).

---

## Project Structure

```
medical-coding-assistant/
├── notebooks/
│   ├── 01_schema.py              # Lakebase tables, indexes, updated_at triggers
│   ├── 02_icd10_pipeline.py      # ICD-10 codes → Delta (Spark + embeddings)
│   ├── 03_mtsamples_pipeline.py  # Clinical notes + embeddings → Delta
│   ├── 04_cdf_delta.py           # Incremental CDC: Lakebase → Delta (CDF enabled)
│   ├── 05_icd10_to_lakebase.py   # ICD-10 codes → Lakebase icd10_lookup (FTS index)
│   └── 06_knn_clinical_notes.py  # KNN demo: cosine similarity UDF over embeddings
└── app/
    ├── app.py                    # Flask routes + JSON error handlers
    ├── agent.py                  # AI agent: 5 tools, retry/backoff, KNN retrieval
    ├── lakebase.py               # Lakebase connection (URL cached at process level)
    ├── app.yaml                  # Databricks App config
    ├── requirements.txt          # incl. sentence-transformers, numpy, databricks-sql-connector
    └── templates/index.html      # Single-page UI (Alpine.js + Tailwind)
```

---

## Setup & Deployment

### 1. Store secrets
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
# Lakebase connection URL
w.secrets.put_secret(scope="database", key="lakebase-url",
                     string_value="postgresql://username:password@host:5432/dbname")
# Databricks PAT for LLM access (generate with all scopes including model-serving)
w.secrets.put_secret(scope="database", key="databricks-token",
                     string_value="dapi...")
# SQL warehouse ID (for KNN retrieval from Delta)
# Find in: SQL Warehouses → your warehouse → Connection details → HTTP path
# Extract the warehouse ID from: /sql/1.0/warehouses/<WAREHOUSE_ID>
```

Set `DATABRICKS_WAREHOUSE_ID` in `app.yaml` env section for KNN retrieval to work.

### 2. Run notebooks in order
1. `01_schema.py` — Lakebase tables + indexes + triggers
2. `02_icd10_pipeline.py` — ICD-10 → Delta
3. `03_mtsamples_pipeline.py` — Clinical notes → Delta
4. `05_icd10_to_lakebase.py` — ICD-10 → Lakebase FTS index (12,316 rows)
5. `04_cdf_delta.py` — after app has sessions; schedule every 15 min in Workflows
6. `06_knn_clinical_notes.py` — optional demo/validation of KNN

### 3. Deploy Databricks App
- Source: this GitHub repo, branch `main`, path `app/`
- Resources: Database (`databricks_ai_bootcamp_postgres`) + Secret (`database/lakebase-url`)

---

## Tech Stack

| Layer | Technology |
|---|---|
| App runtime | Databricks Apps (Flask, port 8000) |
| Operational DB | Lakebase — Managed Postgres (`databricks_ai_bootcamp_postgres`) |
| Analytics store | Delta Lake — Unity Catalog `workspace.medical_coding` |
| LLM | Llama 3.3 70B via Databricks Foundation Models |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim) |
| Code search | NLM Clinical Tables API (free, no key) |
| FTS | Postgres `tsvector` / `plainto_tsquery` on `icd10_lookup` |
| Frontend | Alpine.js + Tailwind CSS (CDN, no build step) |

---

## Security

All credentials are stored in Databricks Secret Scope (`database`). No secrets appear in code or are committed to GitHub. The app accesses them via `WorkspaceClient().secrets.get_secret()` at runtime.
