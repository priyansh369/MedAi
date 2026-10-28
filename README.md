# MedReport AI — Clinical Medical Report Analyzer & NLP Disease Classifier

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Multilingual](https://img.shields.io/badge/Multilingual-English%20%7C%20Hinglish%20%7C%20Gujlish-blue)](#multilingual-support)

A full-stack healthcare Natural Language Processing (NLP) system designed for clinical laboratory report interpretation, abdominal ultrasound analysis, and biomedical abstract classification.

---

## 🌟 Key Capabilities

### 1. Structured & Conversational Medical Report Analyzer
- **Laboratory Pathology Interpretation**: Automatically extracts numerical test values (e.g., Hemoglobin, Fasting Glucose, HbA1c, Serum Creatinine, Platelet count, WBC, Bilirubin, Lipid profile) from digital PDF lab printouts, scanned OCR text, or user text entries.
- **Reference Range Gauges**: Visual horizontal interval meters show patient values mapped against healthy physiological bounds.
- **Direct Disease Identification Engine**: Instantly flags acute conditions from serology and markers:
  - **Dengue Fever / Acute Viral Thrombocytopenia** (NS1 serology or Platelets $< 100,000$ with leukopenia)
  - **Malaria Parasite Infection** (*Plasmodium vivax / falciparum*, smear / antigen positive)
  - **Typhoid Fever** (Widal test positive, Salmonella Typhi reactive)
  - **Type 2 Diabetes Mellitus** (HbA1c $\ge 6.5\%$, Fasting Blood Sugar $\ge 126$ mg/dL)
  - **Severe Anemia** (Hemoglobin $< 8.0$ g/dL)
  - **Clinical Jaundice / Liver Dysfunction** (Bilirubin $\ge 2.0$ mg/dL)
  - **Renal Strain / Kidney Impairment** (Creatinine $\ge 1.5$ mg/dL)
- **All-Normal Reassurance**: If all parameters are within healthy ranges, generates an explicit high-contrast reassurance banner confirming **no doctor consultation is needed**.
- **Abdominal Ultrasound / Sonography Explainer**: Interprets radiological organ evaluations (Gallbladder Cholelithiasis, Diffuse Hepatic Steatosis / Fatty Liver, Nephrolithiasis / Kidney Stones, Hydronephrosis) with clinical negation handling (*"no calculi seen"*, *"no stone or hydronephrosis"*).

### 2. Biomedical Abstract Classifier (NLP Engine)
- Trained on the peer-reviewed **Medical Text Classification (Ohsumed)** corpus.
- Employs a **10,000-feature TF-IDF Vectorizer** (with unigram and bigram extraction) paired with a **Multinomial Naive Bayes** model.
- Accurately classifies clinical notes into 5 disease classes:
  1. *Neoplasms / Oncology*
  2. *Digestive system diseases*
  3. *Nervous system diseases*
  4. *Cardiovascular diseases*
  5. *General pathological conditions*

### 3. Multilingual Regional Language Support
- Automatically detects and processes text written in **English**, **Hinglish** (Hindi in Latin script), and **Gujlish** (Gujarati in Latin script).
- Translates colloquial clinical symptom phrasing into standardized medical vocabulary for classification.
- Generates localized patient-friendly diagnostic explanations in the patient's language.

---

## 🏗️ Architecture

```
User Input (PDF / OCR / Text)
       │
       ▼
[Multilingual Language Detector] ───► English / Hinglish / Gujlish
       │
       ├──► 1. Report Analyzer Engine:
       │       ├── Standard Lab Dictionary Matching (with unit & range extraction)
       │       ├── Clinical Negation Detection (NegEx pattern)
       │       ├── Direct Disease Detector (Dengue, Malaria, Typhoid, Diabetes, Jaundice)
       │       └── All-Normal Reassurance Generator
       │
       └──► 2. NLP Classifier Engine:
               ├── Clinical Symptom Normalizer
               ├── 10,000-Feature TF-IDF Vectorizer
               └── Multinomial Naive Bayes (Top-5 Class Probabilities)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

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
   ```

4. **Run the web application:**
   ```bash
   uvicorn app:app --reload --host 127.0.0.1 --port 8000
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

---

## 📂 Project Structure

```
├── app.py                     # FastAPI application endpoints & sample reports
├── requirements.txt           # Python package dependencies
├── train_models.py            # Model training script for TF-IDF + Naive Bayes / Logistic Regression
├── evaluate_models.py         # Evaluation metrics and confusion matrix generator
├── medical_classifier.py      # Core inference engine for text categorization
├── src/
│   ├── multilingual.py        # Language detection & Hinglish/Gujlish translation engine
│   ├── preprocess.py          # Clinical text normalization and regex tokenizer
│   └── report_analyzer.py     # Lab report parser, negation detection & sonography explainer
├── static/
│   ├── index.html             # Single-page web dashboard interface
│   ├── app.js                 # Frontend application logic, clipboard paste & dynamic gauges
│   └── style.css              # Responsive custom styles
├── models/                    # Serialized TF-IDF vectorizer and trained Scikit-learn models
├── data/                      # Medical text classification datasets
└── reports/                   # Performance benchmarks and confusion matrices
```

---

## ⚖️ Academic Research Disclaimer
*This system is developed as an educational mini-project for natural language processing and medical decision-support experimentation. It does not replace certified professional medical judgment or physician consultation.*
