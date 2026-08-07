# Capstone Reflection — AI Medical Coding Assistant
Sanela Nikodinoska

## 1. What was the most difficult part?

The most difficult part was getting the AI agent to reliably call the `save_suggestions` tool after searching for codes. The Llama model would correctly search ICD-10 codes using the NLM API but sometimes returned its findings as plain text instead of calling the save tool. I resolved this by adding a forced tool-call step at the end of the agent loop — if no suggestions were saved after the main loop, the agent is explicitly instructed to call `save_suggestions` with a constrained `tool_choice` parameter.

A second unexpected challenge was the Change Data Feed pipeline. The initial implementation was a full snapshot overwrite, which is not true CDC. The improved version uses an `updated_at` watermark stored in a `cdc_watermarks` Delta table, merges only new/changed rows from Lakebase into Delta targets, and enables `delta.enableChangeDataFeed = true` so downstream consumers can use `readChangeFeed`. Getting the schema, column types, and CDF version range right required several iterations — notably, CDF data is not recorded at version 0 (table creation), so the pipeline starts reading from version 1 onwards.

## 2. How is Lakebase different from storing data in a traditional analytics table?

Lakebase is a transactional Postgres database optimized for real-time reads and writes, making it ideal for the app's operational data like coding sessions and suggestions. A traditional Delta table is optimized for analytical queries over large datasets but is not suitable for frequent row-level inserts, updates, and deletes that a live application requires. In this project, I used both: Lakebase for the app's live data, and Delta tables for analytics — the Change Data Feed pipeline (notebook 04) bridges the two by incrementally exporting changed Lakebase rows to Delta using a watermark-based MERGE, with CDF enabled on the Delta targets for downstream consumers.

Beyond the original pipeline, I also added an `icd10_lookup` table in Lakebase (notebook 05) to enable Postgres full-text search (`tsvector`/`plainto_tsquery`) over all 12,316 ICD-10 codes. This powers a `semantic_search_icd10` agent tool that complements the NLM API — when the API returns no results, the agent falls back to ranked FTS results from Lakebase.

## 3. What feature would you add next?

I would add a true vector similarity search using the embeddings already stored in `workspace.medical_coding.icd10_codes`. The embeddings (all-MiniLM-L6-v2, 384-dim) exist but are not yet used at query time. Wiring them into a Databricks Vector Search index and exposing it as an agent tool would allow the model to find semantically related codes even when neither the NLM API nor keyword FTS returns results — for example, matching "difficulty swallowing" to dysphagia codes without an exact keyword overlap.

I would also add a batch mode where a coder can upload multiple notes at once and receive codes for all of them in a single session, and a trends tab in the UI that surfaces analytics from the Delta CDF tables — showing which specialties generate the most sessions, which codes are most frequently suggested, and how acceptance rates vary by coder over time.