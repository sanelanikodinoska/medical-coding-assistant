# AI Medical Coding Assistant

**[🚀 Open Live App](https://medical-coding-assistant-7474643859693768.aws.databricksapps.com/)**

## Demo

[![Watch Demo](https://cdn.loom.com/sessions/thumbnails/LOOM_ID-with-play.gif)](https://www.loom.com/share/LOOM_ID)

An AI-powered ICD-10-CM coding assistant that reads free-form clinical notes and suggests the correct diagnosis codes with explanations. Built by a certified medical coder using Databricks, Lakebase, and Llama 3.

---

## What It Does

Paste any clinical note → the AI agent searches ICD-10-CM codes → returns the most accurate codes with confidence scores and explanations → you accept or reject each suggestion.

---

## Capstone Requirements

| Requirement | Implementation |
|---|---|
| ✅ Spark data pipeline | `notebooks/02_icd10_pipeline.py` — fetches ~77,000 ICD-10 codes from NLM API → Delta table |
| ✅ Third-party API | [NLM Clinical Tables API](https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search) — free, no key required |
| ✅ Unstructured data + embeddings | `notebooks/03_mtsamples_pipeline.py` — clinical notes with sentence-transformer embeddings → Delta table |
| ✅ Databricks App | Flask + Alpine.js + Tailwind CSS, deployed at link above |
| ✅ AI agent with tools | Llama 3.3 70B with `search_icd10_codes` and `save_suggestions` tools + full audit trail |
| ✅ CDF → Delta | `notebooks/04_cdf_delta.py` — Lakebase coding sessions → `workspace.medical_coding.sessions_history` |

---

## Architecture

```
NLM ICD-10 API          Clinical Notes (MTSamples)
      │                          │
      └──────── Spark Pipeline ──┘
                     │
              Delta Tables (Unity Catalog)
              workspace.medical_coding.icd10_codes
              workspace.medical_coding.clinical_notes
              workspace.medical_coding.sessions_history  ← CDF output
                     │
              Lakebase (Postgres)
              coding_sessions
              code_suggestions
              agent_tool_calls
                     │
              Databricks App (Flask)
                     │
              AI Agent (Llama 3.3 70B)
              Tools: search_icd10_codes, save_suggestions
```

---

## Project Structure

```
medical-coding-assistant/
├── notebooks/
│   ├── 01_schema.py              # Creates Lakebase tables
│   ├── 02_icd10_pipeline.py      # ICD-10 codes → Delta (Spark)
│   ├── 03_mtsamples_pipeline.py  # Clinical notes + embeddings → Delta
│   └── 04_cdf_delta.py           # Lakebase CDF → Delta analytics
└── app/
    ├── app.py                    # Flask routes
    ├── agent.py                  # AI agent with tool calling loop
    ├── lakebase.py               # Lakebase connection via secret scope
    ├── app.yaml                  # Databricks App config
    ├── requirements.txt
    └── templates/index.html      # Single-page UI (Alpine.js + Tailwind)
```

---

## Setup & Deployment

### 1. Lakebase — store connection URL as secret
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
w.secrets.put_secret(scope="database", key="lakebase-url", string_value="postgresql://...")
```

### 2. Databricks token — store for LLM access
```python
w.secrets.put_secret(scope="database", key="databricks-token", string_value="dapi...")
```

### 3. Run notebooks in order
1. `01_schema.py` — create Lakebase tables
2. `02_icd10_pipeline.py` — load ICD-10 codes
3. `03_mtsamples_pipeline.py` — load clinical notes
4. `04_cdf_delta.py` — export to Delta (run after app has sessions)

### 4. Deploy Databricks App
- Source: this GitHub repo, branch `main`, path `app/`
- Resources: Secret (`database/lakebase-url`) + Secret (`database/databricks-token`)

---

## Tech Stack

- **Databricks Apps** — Flask deployment
- **Lakebase** — Managed Postgres for operational data
- **Delta Lake** — ICD-10 codes, clinical note embeddings, analytics
- **Llama 3.3 70B** — AI agent via Databricks Foundation Models
- **NLM Clinical Tables API** — ICD-10-CM code search
- **sentence-transformers** — Text embeddings (all-MiniLM-L6-v2)
- **Alpine.js + Tailwind CSS** — Frontend

---

## Security

Credentials are stored in Databricks Secret Scope — never in code or committed to GitHub.

