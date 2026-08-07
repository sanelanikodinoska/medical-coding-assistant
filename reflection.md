# Capstone Reflection — AI Medical Coding Assistant
Sanela Nikodinoska

##1. What was the most difficult part?
The most difficult part was getting the AI agent to reliably call the save_suggestions tool after searching for codes. The Llama model would correctly search ICD-10 codes using the NLM API but sometimes returned its findings as plain text instead of calling the save tool. I resolved this by adding a forced tool-call step at the end of the agent loop — if no suggestions were saved after the main loop, the agent is explicitly instructed to call save_suggestions with a constrained tool_choice parameter.

##2. How is Lakebase different from storing data in a traditional analytics table?
Lakebase is a transactional Postgres database optimized for real-time reads and writes, making it ideal for the app's operational data like coding sessions and suggestions. A traditional Delta table is optimized for analytical queries over large datasets but is not suitable for frequent row-level inserts, updates, and deletes that a live application requires. In this project, I used both: Lakebase for the app's live data, and Delta tables for analytics — the Change Data Feed pipeline (notebook 04) bridges the two by exporting Lakebase sessions to Delta for reporting.

##3. What feature would you add next?
I would add a feedback loop where accepted and rejected code suggestions are used to fine-tune the agent's prompting — over time, the system would learn which codes a specific coder tends to accept for which specialties, personalizing suggestions based on individual coding patterns. I would also add a batch mode where a coder can upload multiple notes at once and receive codes for all of them in a single session.