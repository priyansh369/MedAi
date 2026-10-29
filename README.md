# Medoryx AI — Clinical Medical Report Analyzer & NLP Disease Classifier

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/Tests-33%20Passed%20(100%25)-brightgreen?logo=pytest)](https://docs.pytest.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Multi--stage%20Container-2496ED?logo=docker)](https://www.docker.com/)
[![Vercel Ready](https://img.shields.io/badge/Deploy-Vercel%20Serverless-black?logo=vercel)](https://vercel.com)
[![Multilingual](https://img.shields.io/badge/Multilingual-English%20%7C%20Hinglish%20%7C%20Gujlish%20%7C%20Hindi%20%7C%20Gujarati-blue)](#multilingual-support)

A full-stack, enterprise-grade healthcare Natural Language Processing (NLP) system designed for clinical laboratory report interpretation, abdominal ultrasound analysis, and biomedical abstract classification with zero-error reliability.

---

## 🌟 Key Capabilities

### 1. Structured & Conversational Medical Report Analyzer
- **Laboratory Pathology Interpretation**: Automatically extracts numerical test values (e.g., Hemoglobin, Fasting Glucose, HbA1c, Serum Creatinine, Platelet count, WBC, Bilirubin, Lipid profile, Urinalysis dipstick protein) from digital PDF lab printouts, scanned OCR text, or user text entries.
- **Automated Clinical Unit Conversions**:
  - Blood Glucose & Lipids: `mmol/L` $\to$ `mg/dL` ($1\text{ mmol/L} \approx 18.02\text{ mg/dL}$ glucose; $\approx 38.67\text{ mg/dL}$ lipids).
  - Hemoglobin: `g/L` $\to$ `g/dL` ($135\text{ g/L} \to 13.5\text{ g/dL}$).
  - Bounded inequalities extraction ($<0.05$, $>100$).
- **Reference Range Gauges**: Visual horizontal interval meters show patient values mapped against healthy physiological bounds.
- **Direct Disease Identification Engine**: Instantly flags acute conditions from serology and markers:
  - **Dengue Fever / Acute Viral Thrombocytopenia** (NS1 serology or Platelets $< 100,000$ with leukopenia)
  - **Malaria Parasite Infection** (*Plasmodium vivax / falciparum*, smear / antigen positive)
  - **Typhoid Fever** (Widal test positive, Salmonella Typhi reactive)
  - **Type 2 Diabetes Mellitus** (HbA1c $\ge 6.5\%$, Fasting Blood Sugar $\ge 126$ mg/dL)
  - **Severe Anemia** (Hemoglobin $< 8.0$ g/dL)
  - **Clinical Jaundice / Liver Dysfunction** (Bilirubin $\ge 2.0$ mg/dL)
  - **Renal Strain / Kidney Impairment** (Creatinine $\ge 1.5$ mg/dL)
- **All-Normal Reassurance**: If all parameters are within healthy ranges, generates an explicit high-contrast reassurance banner confirming **no doctor consultation is needed**, strictly mutually exclusive with pathology detections.
- **Abdominal Ultrasound / Sonography Explainer**: Interprets radiological organ evaluations (Gallbladder Cholelithiasis, Diffuse Hepatic Steatosis / Fatty Liver, Nephrolithiasis / Kidney Stones, Hydronephrosis) with clinical negation handling (*"no calculi seen"*, *"no stone or hydronephrosis"*).

### 2. Biomedical Abstract Classifier (NLP Engine)
- Trained on the peer-reviewed **Medical Text Classification (Ohsumed)** corpus.
- Employs a **10,000-feature TF-IDF Vectorizer** (with unigram and bigram extraction) paired with a **Multinomial Naive Bayes** and **Logistic Regression** model.
- Accurately classifies clinical notes into 5 disease classes:
  1. *Neoplasms / Oncology*
  2. *Digestive system diseases*
  3. *Nervous system diseases*
  4. *Cardiovascular diseases*
  5. *General pathological conditions*
- **Biomedical Domain Guardrails**: Automatic Out-of-Distribution (OOD) term density validator that warns users when classifying non-clinical or ambiguous queries.

### 3. Multilingual Regional Language Support
- Automatically detects and processes text written in:
  - **English**
  - **Hinglish** (Hindi in Latin script)
  - **Gujlish** (Gujarati in Latin script)
  - **Native Hindi** (Devanagari Unicode: `\u0900-\u097F`)
  - **Native Gujarati Script** (`\u0A80-\u0AFF`)
- Translates colloquial clinical symptom phrasing into standardized medical vocabulary for classification.
- Generates localized patient-friendly diagnostic explanations in the patient's language.

### 4. Enterprise Production Hardening
- **Observability**: Live `/healthz` and `/api/health` probes returning uptime, model load states, and ISO timestamps.
- **Security**: Magic-byte signature validation (`%PDF-`, `PNG`, `JPEG`), strict CORS middleware, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`.
- **Client UX**: Toast notification system, one-click PDF print/export stylesheet, diagnostic summary clipboard copy, and live word/character counters.
- **Continuous Integration**: GitHub Actions automated pipeline executing 33 pytest unit and integration tests on push.

---

## 🏗️ Architecture

```
User Input (PDF / OCR / Text)
       │
       ▼
[Multilingual Language Detector] ───► English / Hinglish / Gujlish / Native Hindi / Gujarati
       │
       ├──► 1. Report Analyzer Engine:
       │       ├── Standard Lab Dictionary Matching (with unit & range extraction)
       │       ├── Automatic Unit Normalization (mmol/L -> mg/dL, g/L -> g/dL)
       │       ├── Clinical Negation Detection (NegEx pattern)
       │       ├── Direct Disease Detector (Dengue, Malaria, Typhoid, Diabetes, Jaundice)
       │       └── All-Normal Reassurance Generator (Mutually Exclusive)
       │
       └──► 2. NLP Classifier Engine:
               ├── Clinical Symptom Normalizer & Multilingual Mapping
               ├── Biomedical Domain Guardrail (OOD Density Check)
               ├── 10,000-Feature TF-IDF Vectorizer
               └── Multinomial Naive Bayes / Logistic Regression (Top-5 Probabilities)
```

---

## 🚀 Getting Started

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/priyansh369/MedAi.git
   cd MedAi
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Run test suite (33 tests):**
   ```bash
   pytest -v
   ```

5. **Run the web application:**
   ```bash
   uvicorn app:app --reload --host 127.0.0.1 --port 8000
   ```

6. Open your browser: `http://127.0.0.1:8000`

---

## 🐳 Docker Containerization

Run the platform with Docker or Docker Compose in an isolated, production-grade container:

```bash
# Using Docker Compose
docker-compose up --build -d

# Check health
curl http://localhost:8000/healthz
```

---

## ☁️ Deploying on Vercel

This repository is optimized for **Vercel Serverless Functions** (fits well under Vercel's 500 MB Lambda size limit):

1. Import this repository in [Vercel Dashboard](https://vercel.com/new).
2. Set **Framework Preset** to `Other`.
3. The included `vercel.json` and `.vercelignore` automatically configure the deployment and exclude offline training datasets (`data/`).
4. Click **Deploy**.

---

## 📂 Project Structure

```
├── app.py                     # Production FastAPI application & middleware
├── Dockerfile                 # Multi-stage container definition
├── docker-compose.yml         # Container orchestration configuration
├── vercel.json                # Vercel serverless routing configuration
├── .vercelignore              # Excludes datasets and offline scripts from lambda bundle
├── .env.example               # Template for environment configuration
├── requirements.txt           # Lightweight production serverless dependencies
├── requirements-dev.txt       # Full local development / test dependencies
├── pytest.ini                 # Pytest test suite configuration
├── tests/                     # 33 comprehensive unit and integration tests
│   ├── test_api.py            # Endpoints, security headers & upload tests
│   ├── test_classifier.py     # NLP model and vectorizer vocabulary tests
│   ├── test_multilingual.py   # Hinglish, Gujlish & Native Hindi/Gujarati script tests
│   └── test_report_analyzer.py# Unit conversions, bounded operators & diseases tests
├── .github/workflows/
│   └── ci.yml                 # Automated GitHub Actions test pipeline
├── train_models.py            # Model training script for TF-IDF + Naive Bayes / Logistic Regression
├── evaluate_models.py         # Evaluation metrics and confusion matrix generator
├── medical_classifier.py      # Core inference engine for text categorization
├── src/
│   ├── multilingual.py        # Language detection & regional translation engine
│   ├── preprocess.py          # Clinical text normalization and regex tokenizer
│   └── report_analyzer.py     # Lab report parser, unit converter & sonography explainer
├── static/
│   ├── index.html             # Clean responsive dashboard (Zero inline styles)
│   ├── app.js                 # Frontend application logic, toasts, copy & PDF print
│   └── style.css              # Consolidated stylesheet with print export media queries
├── models/                    # Serialized TF-IDF vectorizer and trained Scikit-learn models
├── data/                      # Medical text classification datasets (offline training)
└── reports/                   # Performance benchmarks and confusion matrices
```

---

## ⚖️ Academic Research Disclaimer
*This system is developed as an educational project for natural language processing and medical decision-support experimentation. It does not replace certified professional medical judgment or physician consultation.*
