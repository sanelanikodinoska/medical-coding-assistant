# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Clinical Notes Pipeline (MTSamples)
# MAGIC Loads de-identified clinical notes (unstructured text), generates embeddings,
# MAGIC and writes to a Delta table. This is the "unstructured data" requirement.

# COMMAND ----------

%pip install sentence-transformers --quiet
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
from sentence_transformers import SentenceTransformer

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md ## Step 1 — Load clinical notes
# MAGIC
# MAGIC These are sample de-identified medical transcriptions in the style of MTSamples.com.
# MAGIC The full MTSamples dataset (4,000+ notes) is available at https://mtsamples.com

# COMMAND ----------

SAMPLE_NOTES = [
    {
        "specialty": "Cardiology",
        "note_text": "CHIEF COMPLAINT: Chest pain and shortness of breath.\n\nHISTORY: 67-year-old male with a history of hypertension and type 2 diabetes mellitus presenting with substernal chest pain radiating to the left arm, onset 2 hours ago, associated with diaphoresis and nausea. Pain rated 8/10. No relief with rest.\n\nASSESSMENT: Acute STEMI. Patient transferred to cath lab emergently.\n\nPLAN: Percutaneous coronary intervention, aspirin 325mg, heparin drip, cardiology consult."
    },
    {
        "specialty": "Orthopedics",
        "note_text": "CHIEF COMPLAINT: Right knee pain after fall.\n\nHISTORY: 45-year-old female fell while running. Reports immediate pain and swelling of the right knee. Unable to bear weight. No prior knee surgeries.\n\nEXAM: Significant swelling, positive Lachman test, positive anterior drawer sign.\n\nASSESSMENT: Anterior cruciate ligament (ACL) tear, right knee.\n\nPLAN: MRI knee right, orthopedic surgery referral, immobilization, ice, elevation."
    },
    {
        "specialty": "Pulmonology",
        "note_text": "CHIEF COMPLAINT: Worsening shortness of breath and productive cough.\n\nHISTORY: 72-year-old male with 40 pack-year smoking history presenting with progressive dyspnea, chronic productive cough, and recent 15 lb weight loss over 3 months. Hemoptysis noted 1 week ago.\n\nASSESSMENT: Suspected lung malignancy, right upper lobe mass on CXR. COPD exacerbation.\n\nPLAN: CT chest with contrast, pulmonology consult, bronchoscopy, smoking cessation counseling."
    },
    {
        "specialty": "Neurology",
        "note_text": "CHIEF COMPLAINT: Sudden onset right-sided weakness and speech difficulty.\n\nHISTORY: 58-year-old hypertensive female with atrial fibrillation presents with acute onset right arm weakness and expressive aphasia beginning 90 minutes ago. Last known well 2 hours ago.\n\nNIH Stroke Scale: 12. CT head: no hemorrhage.\n\nASSESSMENT: Acute ischemic stroke, left MCA territory.\n\nPLAN: IV tPA administration, stroke neurology consult, MRI brain, ICU admission, anticoagulation management."
    },
    {
        "specialty": "Endocrinology",
        "note_text": "CHIEF COMPLAINT: Poorly controlled diabetes and fatigue.\n\nHISTORY: 55-year-old female with type 2 diabetes mellitus x 12 years. HbA1c 10.2%. Reports polydipsia, polyuria, and blurred vision. Current medications: metformin 1000mg BID.\n\nASSESSMENT: Uncontrolled type 2 diabetes mellitus with hyperglycemia. Early diabetic retinopathy.\n\nPLAN: Add insulin glargine 10 units at bedtime, ophthalmology referral, diabetic nutrition education, repeat HbA1c in 3 months."
    },
    {
        "specialty": "Gastroenterology",
        "note_text": "CHIEF COMPLAINT: Abdominal pain and rectal bleeding.\n\nHISTORY: 62-year-old male with 2-month history of change in bowel habits, intermittent rectal bleeding, and unintentional 20 lb weight loss. Family history of colon cancer.\n\nEXAM: Palpable left lower quadrant mass. Fecal occult blood positive.\n\nASSESSMENT: Suspected colorectal malignancy.\n\nPLAN: Urgent colonoscopy, CT abdomen/pelvis, surgical oncology consult, CEA level."
    },
    {
        "specialty": "Psychiatry",
        "note_text": "CHIEF COMPLAINT: Worsening depression and suicidal ideation.\n\nHISTORY: 35-year-old female with major depressive disorder presenting with 6-week history of worsening depressed mood, anhedonia, insomnia, poor concentration, and passive suicidal ideation without plan or intent. PHQ-9 score: 22 (severe).\n\nASSESSMENT: Major depressive disorder, severe episode. No psychotic features.\n\nPLAN: Increase sertraline to 150mg daily, intensive outpatient program referral, safety planning, follow-up in 1 week."
    },
    {
        "specialty": "Nephrology",
        "note_text": "CHIEF COMPLAINT: Leg swelling and decreased urine output.\n\nHISTORY: 48-year-old male with long-standing hypertension and diabetes presenting with bilateral lower extremity edema, decreased urine output, and fatigue x 2 weeks. Creatinine 4.2 (baseline 1.1 six months ago). BUN 78.\n\nASSESSMENT: Acute on chronic kidney disease, stage 4. Diabetic nephropathy. Hypertensive nephrosclerosis.\n\nPLAN: Nephrology consult, fluid restriction, low-protein diet, fistula creation planning, consider dialysis initiation."
    },
    {
        "specialty": "Hematology",
        "note_text": "CHIEF COMPLAINT: Fatigue and pallor.\n\nHISTORY: 28-year-old female with fatigue, pallor, and palpitations. Heavy menstrual periods x 6 months. Hemoglobin 7.2, MCV 68, ferritin 4.\n\nASSESSMENT: Iron deficiency anemia secondary to menorrhagia.\n\nPLAN: IV iron infusion, gynecology referral for menorrhagia workup, repeat CBC in 4 weeks, dietary iron counseling."
    },
    {
        "specialty": "Dermatology",
        "note_text": "CHIEF COMPLAINT: Changing mole on back.\n\nHISTORY: 52-year-old male with history of sun exposure presents with a mole on the upper back that has been growing and changing color over 6 months. ABCDE criteria: asymmetric, irregular border, multiple colors (brown and black), diameter 8mm, evolving.\n\nASSESSMENT: Suspicious melanocytic lesion. Rule out malignant melanoma.\n\nPLAN: Excisional biopsy with 1-2mm margins, dermatopathology, sentinel lymph node biopsy if confirmed melanoma."
    },
    {
        "specialty": "Rheumatology",
        "note_text": "CHIEF COMPLAINT: Joint pain and morning stiffness.\n\nHISTORY: 42-year-old female with 8-month history of symmetric polyarthritis affecting MCPs, PIPs, and wrists bilaterally. Morning stiffness lasting 2 hours. RF positive, anti-CCP positive. CRP elevated.\n\nASSESSMENT: Seropositive rheumatoid arthritis.\n\nPLAN: Methotrexate 15mg weekly, folic acid 1mg daily, hydroxychloroquine 200mg BID, rheumatology follow-up in 6 weeks, hepatitis B and TB screening before biologics."
    },
    {
        "specialty": "Urology",
        "note_text": "CHIEF COMPLAINT: Difficulty urinating and urinary frequency.\n\nHISTORY: 70-year-old male with 1-year history of urinary hesitancy, weak stream, nocturia x4, and incomplete bladder emptying. PSA 6.8. DRE: enlarged prostate, no nodules.\n\nASSESSMENT: Benign prostatic hyperplasia with lower urinary tract symptoms. Elevated PSA requiring further evaluation.\n\nPLAN: Tamsulosin 0.4mg daily, transrectal ultrasound-guided prostate biopsy, urology consult, post-void residual measurement."
    },
    {
        "specialty": "Infectious Disease",
        "note_text": "CHIEF COMPLAINT: Fever, neck stiffness, and altered mental status.\n\nHISTORY: 19-year-old male presenting with sudden onset fever 39.8C, severe headache, photophobia, and neck stiffness. Kernig and Brudzinski signs positive. Petechial rash noted.\n\nASSESSMENT: Bacterial meningitis, suspected Neisseria meningitidis.\n\nPLAN: Immediate blood cultures, LP after CT head, IV ceftriaxone 2g q12h, dexamethasone, isolation, contact tracing, prophylactic treatment for close contacts."
    },
    {
        "specialty": "Obstetrics",
        "note_text": "CHIEF COMPLAINT: Prenatal visit, 32 weeks gestation.\n\nHISTORY: 29-year-old G2P1 at 32 weeks gestation with gestational diabetes mellitus diagnosed at 26 weeks. Blood glucose well-controlled on diet. Blood pressure 148/96 today, baseline 110/70. 2+ proteinuria. Headache and visual changes denied.\n\nASSESSMENT: Preeclampsia without severe features, gestational diabetes mellitus.\n\nPLAN: Antenatal steroids, magnesium sulfate consideration, BP monitoring every 2 days, NST twice weekly, delivery planning at 37 weeks if stable."
    },
    {
        "specialty": "Emergency Medicine",
        "note_text": "CHIEF COMPLAINT: Severe allergic reaction.\n\nHISTORY: 24-year-old female brought by EMS after bee sting. Generalized urticaria, angioedema of lips and tongue, bronchospasm, and hypotension BP 80/50.\n\nASSESSMENT: Anaphylaxis, severe.\n\nPLAN: Epinephrine 0.3mg IM thigh immediately, IV access x2, diphenhydramine 50mg IV, methylprednisolone 125mg IV, albuterol nebulization, 1L NS bolus, observation x6 hours, EpiPen prescription, allergy referral."
    },
    {
        "specialty": "Oncology",
        "note_text": "CHIEF COMPLAINT: Follow-up for breast cancer.\n\nHISTORY: 49-year-old female with stage IIB invasive ductal carcinoma, ER+/PR+/HER2-, s/p lumpectomy and sentinel lymph node biopsy. Two positive lymph nodes. Starting adjuvant chemotherapy.\n\nASSESSMENT: Breast cancer, ER-positive, node-positive, stage IIB.\n\nPLAN: TC chemotherapy x4 cycles, then letrozole for 5 years, radiation oncology consult, genetic counseling for BRCA testing, bone density monitoring."
    },
    {
        "specialty": "Ophthalmology",
        "note_text": "CHIEF COMPLAINT: Sudden loss of vision in left eye.\n\nHISTORY: 68-year-old hypertensive male with sudden, painless loss of vision in the left eye, described as a 'curtain coming down'. No floaters or flashes prior.\n\nEXAM: Visual acuity: right 20/20, left hand motion only. Fundus: elevated retina with fluid beneath in all quadrants.\n\nASSESSMENT: Rhegmatogenous retinal detachment, left eye, macula off.\n\nPLAN: Urgent vitreoretinal surgery consult, NPO, eye shield, avoid Valsalva maneuvers."
    },
    {
        "specialty": "Pediatrics",
        "note_text": "CHIEF COMPLAINT: Ear pain and fever in a 3-year-old.\n\nHISTORY: Mother brings 3-year-old male with 2-day history of right ear pain, tugging at ear, fever to 39.2C, and decreased appetite. Recent upper respiratory infection 1 week ago. No antibiotic allergies.\n\nEXAM: Right TM erythematous, bulging, landmarks obscured, decreased mobility on pneumatic otoscopy.\n\nASSESSMENT: Acute otitis media, right ear.\n\nPLAN: Amoxicillin 90mg/kg/day divided BID x10 days, acetaminophen for pain/fever, follow-up if no improvement in 48-72 hours."
    },
    {
        "specialty": "Cardiology",
        "note_text": "CHIEF COMPLAINT: Palpitations and dizziness.\n\nHISTORY: 55-year-old male with history of hypertension presenting with episodic palpitations, lightheadedness, and one syncopal episode last week. Duration of episodes 2-5 minutes.\n\nEKG: Atrial fibrillation with rapid ventricular response, rate 142.\n\nASSESSMENT: New onset atrial fibrillation with rapid ventricular response.\n\nPLAN: Rate control with metoprolol, anticoagulation with apixaban after CHADS2-VASc scoring (score 2), cardiology consult, echocardiogram, thyroid function tests, electrolyte panel."
    },
    {
        "specialty": "Pulmonology",
        "note_text": "CHIEF COMPLAINT: Cough and wheezing.\n\nHISTORY: 22-year-old female with history of childhood asthma, returning with worsening dyspnea, wheezing, and cough triggered by exercise and cold air. Using albuterol inhaler daily. Waking from sleep 3 nights per week with symptoms.\n\nASSESSMENT: Asthma, moderate persistent, poorly controlled.\n\nPLAN: Add fluticasone/salmeterol ICS/LABA inhaler, continue albuterol PRN, allergy testing, asthma action plan, peak flow monitoring."
    }
]

print(f"Loaded {len(SAMPLE_NOTES)} clinical notes")

# COMMAND ----------

# MAGIC %md ## Step 2 — Generate embeddings

# COMMAND ----------

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [n["note_text"] for n in SAMPLE_NOTES]
embeddings = model.encode(texts, show_progress_bar=True).tolist()

for i, note in enumerate(SAMPLE_NOTES):
    note["embedding"] = embeddings[i]

print("Embeddings generated")

# COMMAND ----------

# MAGIC %md ## Step 3 — Write to Delta table

# COMMAND ----------

schema_spark = """
    specialty STRING,
    note_text STRING,
    embedding ARRAY<FLOAT>
"""

from pyspark.sql import Row
rows = [Row(specialty=n["specialty"], note_text=n["note_text"], embedding=n["embedding"])
        for n in SAMPLE_NOTES]

df = spark.createDataFrame(rows)

spark.sql("CREATE SCHEMA IF NOT EXISTS main.medical_coding")

(df.write
   .format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable("main.medical_coding.clinical_notes"))

print(f"Written {df.count()} notes to main.medical_coding.clinical_notes")

# COMMAND ----------

spark.sql("SELECT specialty, LEFT(note_text, 80) as preview FROM main.medical_coding.clinical_notes LIMIT 5").show(truncate=False)
