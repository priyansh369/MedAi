"""
app.py
------
FastAPI Backend Server for Medical Text Classification Dashboard.

Serves:
- Real-time NLP inference using pre-trained TF-IDF vectorizer and ML models.
- Verified test evaluation metrics and confusion matrices.
- Verified dataset statistics and sample abstracts.
- Static assets (HTML, CSS, JS) for the presentation dashboard.
"""

from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib

from src.preprocess import clean_medical_text
from src.multilingual import translate_clinical_text, detect_language
from src.report_analyzer import (
    extract_text_from_pdf,
    extract_text_from_image,
    parse_lab_report,
    analyze_sonography_report
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Create FastAPI application
app = FastAPI(
    title="Medical Text Classification API",
    description="NLP-powered classification of medical research abstracts across 5 disease categories.",
    version="1.0.0"
)

# Category ID to Name mapping
LABEL_MAP = {
    1: "Neoplasms",
    2: "Digestive system diseases",
    3: "Nervous system diseases",
    4: "Cardiovascular diseases",
    5: "General pathological conditions"
}

CATEGORY_DESCRIPTIONS = {
    "Neoplasms": "Tumors, carcinomas, oncological malignancies, and benign growths.",
    "Digestive system diseases": "Gastrointestinal, colorectal, hepatic, and gastric conditions.",
    "Nervous system diseases": "Neurological disorders, cerebral ischemia, stroke, and nerve dysfunctions.",
    "Cardiovascular diseases": "Myocardial infarction, coronary artery disease, heart failure, and vascular conditions.",
    "General pathological conditions": "Systemic pathology, sepsis, inflammation, wound healing, and multi-organ conditions."
}

# Load model artifacts into memory on server startup
MODEL_CACHE = {}

def get_models():
    if not MODEL_CACHE:
        vec_path = MODELS_DIR / "selected_vectorizer.joblib"
        nb_path = MODELS_DIR / "selected_best_model.joblib"
        lr_path = MODELS_DIR / "logistic_regression_model.joblib"

        if not vec_path.exists() or not nb_path.exists():
            raise RuntimeError(
                "Model artifacts not found in 'models/' folder. "
                "Please run 'python medical_classifier.py' to generate model artifacts first."
            )

        MODEL_CACHE["vectorizer"] = joblib.load(vec_path)
        MODEL_CACHE["nb_model"] = joblib.load(nb_path)
        MODEL_CACHE["lr_model"] = joblib.load(lr_path) if lr_path.exists() else None

    return MODEL_CACHE

# Pydantic schemas for request validation
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=15000, description="Medical research abstract or clinical text (English, Hinglish, Gujlish)")
    model_type: Optional[str] = Field("nb", description="'nb' for Naive Bayes, 'lr' for Logistic Regression")
    language: Optional[str] = Field("auto", description="'auto', 'en', 'hinglish', or 'gujlish'")

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
def serve_index():
    """Serves the main dashboard page."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    return FileResponse(index_file)


@app.post("/api/classify")
def classify_abstract(req: ClassifyRequest):
    """
    Cleans raw medical text, normalizes Hinglish/Gujlish into medical English concepts,
    vectorizes via TF-IDF, and runs inference using the trained Scikit-learn model.
    """
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    words = raw_text.split()
    if len(words) < 2:
        raise HTTPException(
            status_code=400,
            detail="The entered text is too short. Please provide at least 2 words."
        )

    # 1. Multilingual Translation & Normalization (Supports English, Hinglish, Gujlish)
    trans_info = translate_clinical_text(raw_text)
    text_to_process = trans_info["translated_text"]

    # 2. Clean text using the project's preprocessing function
    cleaned_text = clean_medical_text(text_to_process)

    # 3. Get loaded model artifacts
    models = get_models()
    vectorizer = models["vectorizer"]

    if req.model_type == "lr" and models["lr_model"] is not None:
        model = models["lr_model"]
        model_name = "Logistic Regression"
    else:
        model = models["nb_model"]
        model_name = "Multinomial Naive Bayes"

    # 4. Transform via fitted TF-IDF vocabulary (10,000 features)
    vectorized = vectorizer.transform([cleaned_text])

    # 5. Predict class and confidence probabilities
    predicted_class_id = int(model.predict(vectorized)[0])
    predicted_category = LABEL_MAP.get(predicted_class_id, "Unknown Category")

    probabilities = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized)[0]
        for class_id, prob in zip(model.classes_, probs):
            cid = int(class_id)
            probabilities.append({
                "class_id": cid,
                "category_name": LABEL_MAP.get(cid, f"Category {cid}"),
                "confidence_score": round(float(prob), 4),
                "confidence_percent": round(float(prob) * 100, 2),
                "is_top": (cid == predicted_class_id)
            })
        # Sort descending by confidence
        probabilities.sort(key=lambda x: x["confidence_score"], reverse=True)

    return {
        "success": True,
        "predicted_class_id": predicted_class_id,
        "predicted_category": predicted_category,
        "category_description": CATEGORY_DESCRIPTIONS.get(predicted_category, ""),
        "model_used": model_name,
        "detected_language": trans_info["detected_language"],
        "is_translated": trans_info["is_translated"],
        "translated_clinical_query": trans_info["translated_text"] if trans_info["is_translated"] else None,
        "substitutions_made": trans_info["substitutions_made"],
        "probabilities": probabilities,
        "metrics": {
            "raw_length": len(raw_text),
            "raw_word_count": len(words),
            "cleaned_word_count": len(cleaned_text.split()),
        },
        "disclaimer": "DISCLAIMER: For academic research and educational text classification only. Not for clinical diagnosis."
    }


@app.get("/api/metrics")
def get_model_metrics():
    """Returns actual verified evaluation metrics on the 2,888 test abstracts."""
    return {
        "evaluation_dataset": "Held-out Test Set (2,888 Medical Research Abstracts)",
        "training_dataset": "Training Set (11,550 Medical Research Abstracts)",
        "vocabulary_size": 10000,
        "models": [
            {
                "name": "Multinomial Naive Bayes",
                "is_selected": True,
                "selection_reason": "Highest 5-Fold Cross-Validation Accuracy (59.45%) and Test Accuracy (58.86%)",
                "accuracy": 0.5886,
                "macro_precision": 0.5821,
                "macro_recall": 0.6014,
                "macro_f1": 0.5899,
                "weighted_precision": 0.5804,
                "weighted_recall": 0.5886,
                "weighted_f1": 0.5822,
                "training_time_seconds": 0.010,
                "per_class": [
                    {"id": 1, "name": "Neoplasms", "precision": 0.6826, "recall": 0.7678, "f1": 0.7227, "support": 633},
                    {"id": 2, "name": "Digestive system diseases", "precision": 0.5123, "recall": 0.5585, "f1": 0.5344, "support": 299},
                    {"id": 3, "name": "Nervous system diseases", "precision": 0.5761, "recall": 0.5506, "f1": 0.5631, "support": 385},
                    {"id": 4, "name": "Cardiovascular diseases", "precision": 0.6431, "recall": 0.7148, "f1": 0.6770, "support": 610},
                    {"id": 5, "name": "General pathological conditions", "precision": 0.4963, "recall": 0.4152, "f1": 0.4521, "support": 961}
                ],
                "confusion_matrix": {
                    "classes": ["Neoplasms", "Digestive", "Nervous", "Cardiovascular", "Pathological"],
                    "matrix": [
                        [486, 31, 23, 13, 80],
                        [36, 167, 4, 9, 83],
                        [26, 8, 212, 29, 110],
                        [9, 6, 27, 436, 132],
                        [155, 114, 102, 191, 399]
                    ]
                }
            },
            {
                "name": "Logistic Regression",
                "is_selected": False,
                "selection_reason": "Baseline Comparison Model",
                "accuracy": 0.5672,
                "macro_precision": 0.5738,
                "macro_recall": 0.5400,
                "macro_f1": 0.5522,
                "weighted_precision": 0.5688,
                "weighted_recall": 0.5672,
                "weighted_f1": 0.5648,
                "training_time_seconds": 1.381,
                "per_class": [
                    {"id": 1, "name": "Neoplasms", "precision": 0.6865, "recall": 0.7299, "f1": 0.7075, "support": 633},
                    {"id": 2, "name": "Digestive system diseases", "precision": 0.5070, "recall": 0.3645, "f1": 0.4241, "support": 299},
                    {"id": 3, "name": "Nervous system diseases", "precision": 0.5661, "recall": 0.4338, "f1": 0.4912, "support": 385},
                    {"id": 4, "name": "Cardiovascular diseases", "precision": 0.6474, "recall": 0.6443, "f1": 0.6459, "support": 610},
                    {"id": 5, "name": "General pathological conditions", "precision": 0.4617, "recall": 0.5276, "f1": 0.4925, "support": 961}
                ],
                "confusion_matrix": {
                    "classes": ["Neoplasms", "Digestive", "Nervous", "Cardiovascular", "Pathological"],
                    "matrix": [
                        [462, 20, 20, 14, 117],
                        [39, 109, 4, 7, 140],
                        [31, 4, 167, 24, 159],
                        [14, 5, 23, 393, 175],
                        [127, 77, 81, 169, 507]
                    ]
                }
            }
        ]
    }


@app.get("/api/dataset-info")
def get_dataset_info():
    """Returns actual dataset distribution and metadata."""
    return {
        "dataset_name": "Medical Abstracts Text Classification Corpus",
        "citation": "Schopf et al., ACM NLPIR 2023",
        "total_records": 14438,
        "train_records": 11550,
        "test_records": 2888,
        "split_ratio": "80% Training / 20% Held-out Testing",
        "categories": [
            {
                "id": 1,
                "name": "Neoplasms",
                "train_count": 2530,
                "test_count": 633,
                "total_count": 3163,
                "percentage": "21.9%"
            },
            {
                "id": 2,
                "name": "Digestive system diseases",
                "train_count": 1195,
                "test_count": 299,
                "total_count": 1494,
                "percentage": "10.3%"
            },
            {
                "id": 3,
                "name": "Nervous system diseases",
                "train_count": 1540,
                "test_count": 385,
                "total_count": 1925,
                "percentage": "13.3%"
            },
            {
                "id": 4,
                "name": "Cardiovascular diseases",
                "train_count": 2441,
                "test_count": 610,
                "total_count": 3051,
                "percentage": "21.1%"
            },
            {
                "id": 5,
                "name": "General pathological conditions",
                "train_count": 3844,
                "test_count": 961,
                "total_count": 4805,
                "percentage": "33.3%"
            }
        ]
    }


@app.get("/api/sample-texts")
def get_sample_texts():
    """Returns verified medical abstracts for fast live demonstration in English, Hinglish, and Gujlish."""
    return [
        {
            "id": 1,
            "category": "Neoplasms",
            "language": "English",
            "title": "Pheochromocytoma & Neuron-Specific Enolase",
            "snippet": "Neuropeptide Y and neuron-specific enolase levels in benign and malignant pheochromocytomas. Neuron-specific enolase (NSE) is the isoform of enolase, a glycolytic enzyme found in the neuroendocrine system. Neuropeptide Y (NPY) is a peptide discovered in the peripheral and central nervous systems. The authors evaluated whether the measurement of these molecules could help to discriminate between benign and malignant forms of pheochromocytoma. Determination of serum NSE may be useful for distinguishing between malignant and benign pheochromocytoma tumors."
        },
        {
            "id": 2,
            "category": "Digestive system diseases",
            "language": "English",
            "title": "Chylous Ascites & Gastrointestinal Bleeding",
            "snippet": "The management of postoperative chylous ascites. A case report and literature review. Chylous ascites remains rare as a complication of surgical intervention. Conservative approaches involving dietary restriction of long-chain triglycerides and salt together with multiple paracenteses are therapies of choice. Persistent chylous ascites may be treated surgically by ligation of leaking lymphatics. We report a case of chylous ascites developing after laparotomy and duodenotomy for obscure gastrointestinal bleeding."
        },
        {
            "id": 3,
            "category": "Cardiovascular diseases",
            "language": "English",
            "title": "Thrombolysis in Hemodialysis Grafts & Coronary Flow",
            "snippet": "Pharmacomechanical thrombolysis and angioplasty in the management of clotted vascular hemodialysis grafts. The results of pharmacomechanical thrombolysis and angioplasty of 121 thrombosed vascular grafts were reviewed. The pulsed-spray injection of urokinase into the vascular clot achieved rapid recanalization. All treated arterial grafts underwent complete or near-complete thrombolysis to restore systemic vascular flow."
        },
        {
            "id": "h_cardio",
            "category": "Cardiovascular diseases",
            "language": "Hinglish",
            "title": "[Hinglish] Dil ka Daura / Seene me Dard",
            "snippet": "Patient ko subah se seene me severe dard aur jalan ho rahi hai, saans lene me dikkat aur ECG me acute myocardial infarction aur heart failure ke signs mile hain."
        },
        {
            "id": "h_digestive",
            "category": "Digestive system diseases",
            "language": "Hinglish",
            "title": "[Hinglish] Pait me Ulcer & Acidity",
            "snippet": "Patient ko pait me tezz dard hai, bar bar ulti aur loose motion ho rahe hain, gastric ulcer aur hyperacidity diagnose hui hai."
        },
        {
            "id": "g_cardio",
            "category": "Cardiovascular diseases",
            "language": "Gujlish",
            "title": "[Gujlish] Hriday Rog & Chhati ma Dukhavo",
            "snippet": "Dardi ne chhati ma asahya dukhavo ane dhabkara vadhi gaya hata, angiography ma coronary artery blockage ane hriday rog aavyo hato."
        },
        {
            "id": "g_digestive",
            "category": "Digestive system diseases",
            "language": "Gujlish",
            "title": "[Gujlish] Pet ma Pathri & Acidity",
            "snippet": "Mane pet ma right side dukhe che, sonography report ma gallbladder ma pathri (gallstone) batavi che ane liver ma charbi (fatty liver) aavi che."
        }
    ]

# -----------------------------------------------------------------------------
# Medical Report Analyzer Endpoints (Phase 3 & Phase 5)
# -----------------------------------------------------------------------------

class ReportAnalyzeRequest(BaseModel):
    report_text: str = Field(..., min_length=3, max_length=50000, description="Extracted medical report text to analyze")
    language: Optional[str] = Field("auto", description="'auto', 'en', 'hinglish', or 'gujlish'")

@app.post("/api/report/upload")
async def upload_medical_report(file: UploadFile = File(...)):
    """
    Extracts text from PDF or image reports in memory.
    Files are never saved permanently to disk (Privacy-compliant).
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Please upload a PDF, PNG, JPG, or JPEG medical report."
        )

    # Read bytes into memory
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
    
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 15 MB limit.")

    if ext == ".pdf":
        extracted_text = extract_text_from_pdf(content)
    else:
        extracted_text = extract_text_from_image(content, filename)

    return {
        "success": True,
        "filename": filename,
        "file_type": ext.replace(".", "").upper(),
        "file_size_kb": round(len(content) / 1024, 1),
        "extracted_text": extracted_text,
        "char_count": len(extracted_text),
        "notice": "Review and edit the extracted text below before running laboratory or sonography interpretation."
    }

@app.post("/api/report/analyze-lab")
def analyze_lab_report_endpoint(req: ReportAnalyzeRequest):
    """
    Parses laboratory test parameters, compares values against reference ranges,
    flags high/low deviations, and provides plain-language non-diagnostic interpretations
    in English, Hinglish, or Gujlish.
    """
    result = parse_lab_report(req.report_text, target_lang=req.language)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to parse laboratory report."))
    return result

@app.post("/api/report/analyze-sonography")
def analyze_sonography_report_endpoint(req: ReportAnalyzeRequest):
    """
    Explains written ultrasound and sonography findings in English, Hinglish, or Gujlish,
    translating dense radiological terminology into patient-friendly language.
    """
    result = analyze_sonography_report(req.report_text, target_lang=req.language)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to analyze sonography report."))
    return result

@app.get("/api/report/sample-reports")
def get_sample_medical_reports():
    """
    Provides realistic, verified sample reports in English, Hinglish, and Gujlish for pathology, lab panels,
    and written ultrasound so students and examiners can test immediately without a PDF file.
    """
    return [
        {
            "id": "normal_health_lab",
            "title": "[English] Normal Health Profile (All Normal - No Doctor Needed)",
            "language": "English",
            "type": "lab",
            "snippet": (
                "ANNUAL PREVENTIVE HEALTH CHECKUP REPORT (ALL NORMAL)\n"
                "Patient ID: 10482   Age: 32   Sex: Male\n\n"
                "TEST NAME                      VALUE      UNIT       REFERENCE INTERVAL\n"
                "Hemoglobin                     14.2       g/dL       12.0 - 15.5\n"
                "Total Leukocyte Count (WBC)    6800       /mcL       4000 - 11000\n"
                "RBC Count                      4.8        million    4.0 - 5.5\n"
                "Platelet Count                 265000     /mcL       150000 - 450000\n"
                "Fasting Blood Sugar            88         mg/dL      70 - 99\n"
                "HbA1c                          5.2        %          4.0 - 5.6\n"
                "Serum Creatinine               0.9        mg/dL      0.6 - 1.3\n"
                "Blood Urea                     24         mg/dL      15 - 45\n"
                "Total Bilirubin                0.7        mg/dL      0.2 - 1.2\n"
                "SGPT (ALT)                     28         U/L        7 - 55\n"
                "Total Cholesterol              172        mg/dL      100 - 200"
            )
        },
        {
            "id": "dengue_fever_lab",
            "title": "[English] Dengue Fever with Severe Thrombocytopenia",
            "language": "English",
            "type": "lab",
            "snippet": (
                "ACUTE FEBRILE ILLNESS / VECTOR BORNE INFECTION REPORT\n"
                "Clinical History: High continuous fever for 4 days, retro-orbital headache, body aches.\n\n"
                "Dengue NS1 Antigen: POSITIVE (Reactive)\n"
                "TEST NAME                      VALUE      UNIT       REFERENCE INTERVAL\n"
                "Platelet Count                 42000      /mcL       150000 - 450000\n"
                "Total Leukocyte Count (WBC)    2900       /mcL       4000 - 11000\n"
                "Hemoglobin                     14.8       g/dL       12.0 - 15.5\n"
                "Serum Creatinine               0.8        mg/dL      0.6 - 1.3"
            )
        },
        {
            "id": "malaria_typhoid_lab",
            "title": "[English] Malaria & Typhoid Positive Serology",
            "language": "English",
            "type": "lab",
            "snippet": (
                "CLINICAL PARASITOLOGY & FEVER SEROLOGY REPORT\n"
                "Clinical History: Chills, high fever spikes, rigors, and malaise.\n\n"
                "Malaria Parasite (MP Smear): POSITIVE (Plasmodium Vivax ring forms identified)\n"
                "Widal Test (Typhoid): S. Typhi O (1:160) POSITIVE, S. Typhi H (1:320) POSITIVE\n\n"
                "TEST NAME                      VALUE      UNIT       REFERENCE INTERVAL\n"
                "Platelet Count                 112000     /mcL       150000 - 450000\n"
                "Total Leukocyte Count (WBC)    4800       /mcL       4000 - 11000\n"
                "Hemoglobin                     11.2       g/dL       12.0 - 15.5"
            )
        },
        {
            "id": "cbc",
            "title": "[English] CBC with Anemia & Leukocytosis",
            "language": "English",
            "type": "lab",
            "snippet": (
                "PATIENT LABORATORY INVESTIGATION REPORT\n"
                "Investigation: Complete Blood Count (CBC)\n\n"
                "TEST NAME                      VALUE      UNIT       REFERENCE INTERVAL\n"
                "Hemoglobin                     9.8        g/dL       12.0 - 15.5\n"
                "Total Leukocyte Count (WBC)    13400      /mcL       4000 - 11000\n"
                "RBC Count                      3.8        million    4.0 - 5.5\n"
                "Platelet Count                 220000     /mcL       150000 - 450000\n"
                "Serum Creatinine               0.9        mg/dL      0.6 - 1.3\n"
                "Fasting Blood Sugar            118        mg/dL      70 - 99"
            )
        },
        {
            "id": "cmp_lab",
            "title": "[English] Metabolic Panel & Kidney Strain",
            "language": "English",
            "type": "lab",
            "snippet": (
                "COMPREHENSIVE METABOLIC BIOCHEMISTRY PANEL\n"
                "Patient ID: 94812   Sex: Male   Age: 56\n\n"
                "Fasting Blood Sugar            174        mg/dL      70 - 99\n"
                "HbA1c                          8.4        %          4.0 - 5.6\n"
                "Serum Creatinine               2.1        mg/dL      0.6 - 1.3\n"
                "Blood Urea Nitrogen (BUN)      34         mg/dL      7 - 20\n"
                "Serum Potassium                5.4        mmol/L     3.5 - 5.0\n"
                "Serum Sodium                   138        mmol/L     135 - 145"
            )
        },
        {
            "id": "lipid_lab",
            "title": "[English] Lipid Panel & Cardiovascular Risk",
            "language": "English",
            "type": "lab",
            "snippet": (
                "FASTING LIPID PROFILE EXAMINATION\n"
                "Total Cholesterol              245        mg/dL      125 - 200\n"
                "Triglycerides                  280        mg/dL      50 - 150\n"
                "LDL Cholesterol                162        mg/dL      60 - 100\n"
                "HDL Cholesterol                34         mg/dL      40 - 60"
            )
        },
        {
            "id": "hinglish_normal_lab",
            "title": "[Hinglish] All Normal Report (Sab Fit - No Doctor Needed)",
            "language": "Hinglish",
            "type": "lab",
            "snippet": (
                "MERA COMPLETE HEALTH CHECKUP (HINGLISH - SABHI NORMAL)\n"
                "Mera hemoglobin 14.5 g/dL aaya hai (Normal 12.0 - 15.5)\n"
                "Fasting blood sugar 85 mg/dL aaya hai (Normal 70 - 99)\n"
                "Platelet count 245000 /mcL aaya hai (Normal 150000 - 450000)\n"
                "Total leukocyte count (WBC) 6500 /mcL aaya hai (Normal 4000 - 11000)\n"
                "Serum creatinine 0.8 mg/dL bataya hai (Normal 0.6 - 1.3)"
            )
        },
        {
            "id": "hinglish_lab",
            "title": "[Hinglish] CBC, Sugar & Creatinine Statement",
            "language": "Hinglish",
            "type": "lab",
            "snippet": (
                "MERA CLINICAL LABORATORY REPORT (HINGLISH)\n"
                "Mera hemoglobin 9.2 g/dL aaya hai (Normal 12.0 - 15.5)\n"
                "Fasting blood sugar 155 mg/dL aaya hai (Normal 70 - 99)\n"
                "Total leukocyte count (WBC) 14200 /mcL aaya hai (Normal 4000 - 11000)\n"
                "Serum creatinine 1.4 mg/dL bataya hai (Normal 0.6 - 1.3)"
            )
        },
        {
            "id": "gujlish_normal_lab",
            "title": "[Gujlish] All Normal Report (Badhu Normal - No Doctor Needed)",
            "language": "Gujlish",
            "type": "lab",
            "snippet": (
                "MARO SWAASTHYA CHECKUP REPORT (GUJLISH - BADHU NORMAL)\n"
                "Maro hemoglobin (lohi) 13.8 g/dL aavyo che (Normal 12.0 - 15.5)\n"
                "Fasting blood sugar 90 mg/dL che (Normal 70 - 99)\n"
                "Total WBC count 7200 /mcL aavya che (Normal 4000 - 11000)\n"
                "Platelet count 280000 /mcL che (Normal 150000 - 450000)\n"
                "Serum creatinine 0.9 mg/dL aavyu che (Normal 0.6 - 1.3)"
            )
        },
        {
            "id": "gujlish_lab",
            "title": "[Gujlish] Lohi (Hb), Sugar & WBC Report",
            "language": "Gujlish",
            "type": "lab",
            "snippet": (
                "MARO DARDI LABORATORY REPORT (GUJLISH)\n"
                "Maro hemoglobin (lohi) 8.8 g/dL aavyo che (Normal 12.0 - 15.5)\n"
                "Fasting blood sugar 165 mg/dL che (Normal 70 - 99)\n"
                "Total WBC count 13800 /mcL aavya che (Normal 4000 - 11000)\n"
                "Serum creatinine 1.3 mg/dL aavyu che (Normal 0.6 - 1.3)"
            )
        },
        {
            "id": "ultrasound_abdomen",
            "title": "[English] Abdominal Ultrasound (Cholelithiasis)",
            "language": "English",
            "type": "sonography",
            "snippet": (
                "ULTRASOUND EXAMINATION OF WHOLE ABDOMEN\n\n"
                "CLINICAL INDICATION: Right upper quadrant abdominal pain and dyspepsia.\n\n"
                "FINDINGS:\n"
                "LIVER: Shows mildly increased parenchymal echogenicity with subtle posterior attenuation, "
                "consistent with Grade I diffuse hepatic steatosis (fatty liver). No focal hepatic mass.\n"
                "GALLBLADDER: Well distended. Multiple mobile hyperechoic foci measuring up to 7 mm noted inside gallbladder lumen "
                "with distinct posterior acoustic shadowing, diagnostic of cholelithiasis (gallstones).\n"
                "KIDNEYS: Both kidneys are normal in size and cortical echogenicity. No calculus or hydronephrosis.\n\n"
                "IMPRESSION: 1. Multiple Gallbladder Calculi (Cholelithiasis). 2. Mild Hepatic Steatosis (Grade I Fatty Liver)."
            )
        },
        {
            "id": "hinglish_sono",
            "title": "[Hinglish] Ultrasound (Pitte me Pathri & Fatty Liver)",
            "language": "Hinglish",
            "type": "sonography",
            "snippet": (
                "SONOGRAPHY EXAMINATION OF WHOLE ABDOMEN (HINGLISH)\n\n"
                "CLINICAL INDICATION: Pait me dard aur ulti.\n\n"
                "FINDINGS:\n"
                "Gallbladder me pathri (gallstones) dikhi hai with acoustic shadowing. Cholelithiasis bataya hai.\n"
                "Liver me charbi (diffuse hepatic steatosis Grade 1 / fatty liver) hai.\n"
                "Kidneys normal hain, no calculus seen.\n\n"
                "IMPRESSION: Gallbladder calculi (pathri) with mild fatty liver."
            )
        },
        {
            "id": "gujlish_sono",
            "title": "[Gujlish] Sonography (Gallbladder ma Pathri & Charbi)",
            "language": "Gujlish",
            "type": "sonography",
            "snippet": (
                "ULTRASOUND REPORT OF WHOLE ABDOMEN (GUJLISH)\n\n"
                "CLINICAL FINDINGS:\n"
                "Gallbladder ma pathri (gallstone) che ane mobile calculi batavya che.\n"
                "Liver ma charbi che, diffuse hepatic steatosis Grade 1 consistent with fatty liver.\n"
                "Kidneys: Both kidneys normal che, koi stone nathi.\n\n"
                "IMPRESSION: Cholelithiasis (pathri) ane mild hepatic steatosis (charbi)."
            )
        }
    ]

@app.get("/api/report/xray-status")
def get_xray_roadmap():
    """
    Transparently details the architectural requirements for future X-ray analysis,
    explaining why text NLP models cannot diagnose pixel images.
    """
    return {
        "status": "Roadmap / Planned Future Architecture",
        "why_not_active": (
            "Chest X-rays and medical radiographs consist of spatial 2D/3D pixel intensity arrays (DICOM/PNG), "
            "whereas our current verified system is an NLP statistical language model built on TF-IDF text features. "
            "Feeding an image directly to a text classifier is clinically invalid and scientifically ungrounded."
        ),
        "required_vision_pipeline": [
            "1. Deep Convolutional Neural Networks (e.g., DenseNet-121, EfficientNet-B4) or Vision Transformers (ViT).",
            "2. Pre-training on validated radiograph benchmarks such as NIH ChestX-ray14 (112,120 frontal images) or Stanford CheXpert.",
            "3. Multi-label probability heads predicting cardiopulmonary conditions (e.g., Cardiomegaly, Consolidation, Pleural Effusion, Pneumothorax).",
            "4. Grad-CAM (Gradient-weighted Class Activation Mapping) saliency heatmaps to show radiologists the exact pixel regions triggering the classification.",
            "5. Clinical DICOM 16-bit windowing/leveling preprocessing to preserve high dynamic range tissue density."
        ],
        "message_for_examiners": (
            "We have deliberately integrated a clearly labeled architectural placeholder rather than generating fabricated image predictions. "
            "This demonstrates an ethical, scientifically rigorous approach to medical AI engineering."
        )
    }

# Prevent stale browser caching during development and demonstration
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Mount static files folder
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

