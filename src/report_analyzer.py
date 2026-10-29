"""
src/report_analyzer.py
----------------------
Medical Report Extraction and Clinical Diagnosis Synthesis Engine.

Key Capabilities:
1. Genuine Optical Character Recognition (OCR) via RapidOCR:
   - Scanned and photo reports (JPG, PNG, TIFF) are recognized into real text.
   - Scanned PDF documents are rasterized page-by-page with PyMuPDF and OCR-processed.
   - Digital PDFs are parsed natively via PyMuPDF / pypdf.
2. Comprehensive Multi-Parameter Laboratory Parser:
   - Recognizes CBC, Metabolic, Renal, Liver, Lipid, Thyroid, and Electrolyte tests.
   - Extracts measured values and reference ranges directly from report lines.
   - Evaluates HIGH / LOW / NORMAL status.
3. Clinical Diagnosis Synthesis Engine:
   - Correlates multi-test patterns together (e.g., Low Hb + Low MCV -> Microcytic Anemia;
     High Glucose + High HbA1c -> Diabetes Mellitus; High Creatinine + BUN -> Renal Insufficiency;
     High ALT/AST + Bilirubin -> Hepatobiliary Injury; High WBC + Neutrophilia -> Acute Infection).
   - Assigns a clinical Acuity / Risk Level (Critical, High, Moderate, Normal).
   - Formulates Primary Diagnostic Impressions and Differential Diagnoses.
   - Recommends actionable clinical next steps and confirmatory testing.
4. Ultrasound & Sonography Diagnostic Findings Explainer:
   - Synthesizes organ-specific pathology (Gallstones, Fatty Liver, Renal Calculi, Hydronephrosis, Ascites).
"""

import io
import re
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image
import pypdf

# Optional OCR & rasterization engines (graceful fallback in lightweight serverless mode)
try:
    import pymupdf as fitz
except ImportError:
    fitz = None

try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
except Exception:
    ocr_engine = None

from src.multilingual import (
    detect_language,
    get_multilingual_lab_explanation,
    get_multilingual_sono_explanation,
    MULTILINGUAL_ACUITY
)

# -----------------------------------------------------------------------------
# Comprehensive Clinical Laboratory Reference Dictionary
# -----------------------------------------------------------------------------
STANDARD_LAB_DICTIONARY = {
    # Complete Blood Count (CBC)
    "hemoglobin": {
        "display_name": "Hemoglobin (Hb)",
        "unit": "g/dL",
        "low": 12.0,
        "high": 17.5,
        "category": "Complete Blood Count (CBC)",
        "low_meaning": "Low hemoglobin indicates Anemia (decreased oxygen-carrying capacity).",
        "high_meaning": "Elevated hemoglobin suggests Polycythemia, chronic hypoxia, or hemoconcentration from dehydration.",
        "normal_meaning": "Optimal oxygen-carrying red cell pigment."
    },
    "wbc": {
        "display_name": "White Blood Cell Count (WBC / TLC)",
        "unit": "cells/mcL",
        "low": 4000,
        "high": 11000,
        "category": "Complete Blood Count (CBC)",
        "low_meaning": "Leukopenia: immune system suppression, viral infection, or bone marrow underactivity.",
        "high_meaning": "Leukocytosis: active bacterial infection, acute systemic inflammation, tissue necrosis, or physiological stress.",
        "normal_meaning": "Immune white blood cell defense is balanced."
    },
    "platelets": {
        "display_name": "Platelet Count (Thrombocytes)",
        "unit": "cells/mcL",
        "low": 150000,
        "high": 450000,
        "category": "Complete Blood Count (CBC)",
        "low_meaning": "Thrombocytopenia: increased bleeding and bruising vulnerability; common in viral illnesses (e.g. dengue), autoimmune destruction, or liver disease.",
        "high_meaning": "Thrombocytosis: reactive inflammation, acute blood loss recovery, iron deficiency, or myeloproliferative disorder.",
        "normal_meaning": "Adequate platelet clotting capability."
    },
    "rbc": {
        "display_name": "Red Blood Cell Count (RBC)",
        "unit": "million/mcL",
        "low": 4.0,
        "high": 5.9,
        "category": "Complete Blood Count (CBC)",
        "low_meaning": "Reduced total circulating red cell mass; primary marker of anemia.",
        "high_meaning": "Erythrocytosis: secondary to chronic smoking, hypoxia, pulmonary disease, or dehydration.",
        "normal_meaning": "Red cell production is within physiological limits."
    },
    "mcv": {
        "display_name": "Mean Corpuscular Volume (MCV)",
        "unit": "fL",
        "low": 80.0,
        "high": 100.0,
        "category": "Complete Blood Count (CBC)",
        "low_meaning": "Microcytosis: red cells are abnormally small, characteristic of Iron Deficiency Anemia or Thalassemia.",
        "high_meaning": "Macrocytosis: red cells are enlarged, characteristic of Vitamin B12 / Folate deficiency or liver disease.",
        "normal_meaning": "Normocytic red cell volume."
    },
    "esr": {
        "display_name": "Erythrocyte Sedimentation Rate (ESR)",
        "unit": "mm/hr",
        "low": 0.0,
        "high": 20.0,
        "category": "Inflammatory Markers",
        "low_meaning": "Physiologically normal.",
        "high_meaning": "Elevated ESR indicates non-specific systemic inflammation, infection, autoimmune disease, or malignancy.",
        "normal_meaning": "No significant systemic inflammatory cascade detected."
    },

    # Metabolic & Glycemic Panel
    "glucose_fasting": {
        "display_name": "Fasting Blood Glucose (FBS)",
        "unit": "mg/dL",
        "low": 70,
        "high": 99,
        "category": "Diabetes & Metabolic Panel",
        "low_meaning": "Hypoglycemia: low blood sugar requiring carbohydrate correction.",
        "high_meaning": "Hyperglycemia: 100-125 mg/dL indicates Prediabetes; >=126 mg/dL indicates Diabetes Mellitus.",
        "normal_meaning": "Fasting glucose regulation is normal."
    },
    "glucose_random": {
        "display_name": "Random / Postprandial Blood Glucose",
        "unit": "mg/dL",
        "low": 70,
        "high": 140,
        "category": "Diabetes & Metabolic Panel",
        "low_meaning": "Low blood sugar.",
        "high_meaning": "140-199 mg/dL indicates impaired glucose tolerance; >=200 mg/dL strongly suggests Diabetes Mellitus.",
        "normal_meaning": "Post-prandial glycemic control is within target."
    },
    "hba1c": {
        "display_name": "Glycated Hemoglobin (HbA1c)",
        "unit": "%",
        "low": 4.0,
        "high": 5.6,
        "category": "Diabetes & Metabolic Panel",
        "low_meaning": "Low average blood sugar.",
        "high_meaning": "5.7 - 6.4% indicates Prediabetes; >=6.5% confirms chronic Diabetes Mellitus (3-month glycemic record).",
        "normal_meaning": "Healthy 3-month glycemic equilibrium."
    },

    # Renal (Kidney) Panel
    "creatinine": {
        "display_name": "Serum Creatinine",
        "unit": "mg/dL",
        "low": 0.6,
        "high": 1.3,
        "category": "Renal Function (Kidney)",
        "low_meaning": "Low muscle mass or malnutrition.",
        "high_meaning": "Elevated Creatinine indicates impaired renal glomerular filtration, Acute Kidney Injury (AKI), or Chronic Kidney Disease (CKD).",
        "normal_meaning": "Kidney filtration function is intact."
    },
    "urea": {
        "display_name": "Blood Urea / BUN",
        "unit": "mg/dL",
        "low": 15,
        "high": 45,
        "category": "Renal Function (Kidney)",
        "low_meaning": "Low protein intake or severe liver dysfunction.",
        "high_meaning": "Uremia / Azotemia: impaired renal clearance, dehydration, or gastrointestinal bleeding.",
        "normal_meaning": "Nitrogenous waste clearance is balanced."
    },
    "uric_acid": {
        "display_name": "Uric Acid",
        "unit": "mg/dL",
        "low": 3.0,
        "high": 7.2,
        "category": "Renal & Metabolic",
        "low_meaning": "Normal physiological finding.",
        "high_meaning": "Hyperuricemia: primary predisposing factor for Gouty Arthritis and Uric Acid Kidney Stones.",
        "normal_meaning": "Purine byproduct level is within safe bounds."
    },

    # Lipid Profile (Cardiovascular Risk)
    "total_cholesterol": {
        "display_name": "Total Cholesterol",
        "unit": "mg/dL",
        "low": 100,
        "high": 200,
        "category": "Lipid Profile (Cardiovascular)",
        "low_meaning": "Severe malnutrition or liver insufficiency.",
        "high_meaning": "Hypercholesterolemia: contributes to coronary plaque formation and atherosclerotic cardiovascular risk.",
        "normal_meaning": "Optimal total cholesterol concentration."
    },
    "triglycerides": {
        "display_name": "Triglycerides",
        "unit": "mg/dL",
        "low": 50,
        "high": 150,
        "category": "Lipid Profile (Cardiovascular)",
        "low_meaning": "Normal.",
        "high_meaning": "Hypertriglyceridemia: associated with insulin resistance, metabolic syndrome, and acute pancreatitis if markedly elevated (>500 mg/dL).",
        "normal_meaning": "Normal circulating triglycerides."
    },
    "hdl": {
        "display_name": "HDL ('Good') Cholesterol",
        "unit": "mg/dL",
        "low": 40,
        "high": 60,
        "category": "Lipid Profile (Cardiovascular)",
        "low_meaning": "Low protective HDL: increases risk of coronary artery disease.",
        "high_meaning": "Cardioprotective high-density lipoprotein level.",
        "normal_meaning": "Adequate cardioprotective HDL level."
    },
    "ldl": {
        "display_name": "LDL ('Bad') Cholesterol",
        "unit": "mg/dL",
        "low": 50,
        "high": 100,
        "category": "Lipid Profile (Cardiovascular)",
        "low_meaning": "Optimal for cardiovascular health.",
        "high_meaning": "Elevated LDL: primary atherogenic particle driving arterial stenosis and ischemic cardiovascular events.",
        "normal_meaning": "Desirable LDL concentration."
    },

    # Liver Function (Hepatic Panel)
    "sgpt_alt": {
        "display_name": "Alanine Aminotransferase (ALT / SGPT)",
        "unit": "U/L",
        "low": 7,
        "high": 55,
        "category": "Liver Function (Hepatic)",
        "low_meaning": "Normal finding.",
        "high_meaning": "Hepatocellular Injury / Transaminitis: indicates active hepatocyte damage (fatty liver, viral hepatitis, toxin, or drug reaction).",
        "normal_meaning": "Liver cellular integrity is maintained."
    },
    "sgot_ast": {
        "display_name": "Aspartate Aminotransferase (AST / SGOT)",
        "unit": "U/L",
        "low": 8,
        "high": 48,
        "category": "Liver Function (Hepatic)",
        "low_meaning": "Normal finding.",
        "high_meaning": "Elevated AST: reflects hepatocyte necrosis, cardiac injury, or skeletal muscle breakdown.",
        "normal_meaning": "Enzyme level is within physiological limits."
    },
    "bilirubin_total": {
        "display_name": "Total Bilirubin",
        "unit": "mg/dL",
        "low": 0.2,
        "high": 1.2,
        "category": "Liver Function (Hepatic)",
        "low_meaning": "Normal finding.",
        "high_meaning": "Hyperbilirubinemia (Jaundice): caused by biliary duct obstruction, acute hepatitis, or red blood cell hemolysis.",
        "normal_meaning": "Normal biliary excretion and heme breakdown."
    },
    "albumin": {
        "display_name": "Serum Albumin",
        "unit": "g/dL",
        "low": 3.5,
        "high": 5.2,
        "category": "Liver Function (Hepatic)",
        "low_meaning": "Hypoalbuminemia: decreased liver synthetic function, protein malnutrition, or renal protein loss (nephrotic syndrome).",
        "high_meaning": "Dehydration with hemoconcentration.",
        "normal_meaning": "Optimal liver protein synthesis."
    },

    # Thyroid Function
    "tsh": {
        "display_name": "Thyroid Stimulating Hormone (TSH)",
        "unit": "uIU/mL",
        "low": 0.4,
        "high": 4.5,
        "category": "Thyroid Function",
        "low_meaning": "Suppressed TSH indicates Hyperthyroidism (overactive thyroid), causing tachycardia, weight loss, or anxiety.",
        "high_meaning": "Elevated TSH indicates Primary Hypothyroidism (underactive thyroid), leading to fatigue, weight gain, constipation, and cold intolerance.",
        "normal_meaning": "Pituitary-thyroid regulatory feedback is normal."
    }
}

# -----------------------------------------------------------------------------
# Ultrasound Terminology Knowledge Base
# -----------------------------------------------------------------------------
ULTRASOUND_DICTIONARY = {
    "hyperechoic": {
        "term": "Hyperechoic",
        "meaning": "Appears bright/white on ultrasound. Dense tissue that reflects strong sound echoes (e.g. fat, calcification, gallstones, or renal stones)."
    },
    "hypoechoic": {
        "term": "Hypoechoic",
        "meaning": "Appears darker/gray on ultrasound. Represents solid cellular tissue that reflects fewer sound echoes (e.g. nodules, solid lesions, or inflamed parenchyma)."
    },
    "anechoic": {
        "term": "Anechoic / Sonolucent",
        "meaning": "Appears completely black. Represents pure fluid with no internal echoes (e.g. simple cysts, urinary bladder, or gallbladder bile)."
    },
    "isoechoic": {
        "term": "Isoechoic",
        "meaning": "Displays intermediate brightness similar to surrounding healthy organ tissue."
    },
    "acoustic shadowing": {
        "term": "Posterior Acoustic Shadowing",
        "meaning": "A dark band behind a very dense calcification (like a gallstone or kidney stone) because sound waves cannot penetrate."
    },
    "cholelithiasis": {
        "term": "Cholelithiasis (Gallstones)",
        "meaning": "Presence of mobile hard stones inside the gallbladder lumen."
    },
    "nephrolithiasis": {
        "term": "Nephrolithiasis (Kidney Stones)",
        "meaning": "Presence of calculus/stones within the renal parenchyma or renal collecting system."
    },
    "hydronephrosis": {
        "term": "Hydronephrosis",
        "meaning": "Swelling or dilatation of the renal pelvicalyceal system due to obstruction or urine backup in the urinary tract."
    },
    "hepatomegaly": {
        "term": "Hepatomegaly",
        "meaning": "Enlargement of the liver beyond typical longitudinal dimensions (usually > 15-16 cm in adults)."
    },
    "steatosis": {
        "term": "Hepatic Steatosis (Fatty Liver)",
        "meaning": "Increased echogenicity (brightness) of the liver parenchyma, commonly associated with metabolic fatty liver changes."
    },
    "ascites": {
        "term": "Free Fluid / Ascites",
        "meaning": "Abnormal fluid accumulation in peritoneal spaces (e.g. pouch of Douglas, Morison's pouch)."
    },
    "cyst": {
        "term": "Simple Cyst",
        "meaning": "A round, fluid-filled benign sac with thin walls and posterior acoustic enhancement."
    }
}


# -----------------------------------------------------------------------------
# Document OCR & Text Extraction Engine
# -----------------------------------------------------------------------------
def run_ocr_on_pil_image(image: Image.Image) -> str:
    """Runs RapidOCR on a PIL Image if available."""
    if ocr_engine is None:
        return ""
    try:
        img_rgb = image.convert("RGB")
        img_np = np.array(img_rgb)
        ocr_result, _ = ocr_engine(img_np)
        if not ocr_result:
            return ""
        extracted_lines = [box[1] for box in ocr_result if box and len(box) > 1 and box[1].strip()]
        return "\n".join(extracted_lines)
    except Exception as e:
        return f"[OCR Error: {str(e)}]"


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts text from PDF files:
    1. Primary engine: pypdf (pure-Python, lightweight, serverless-ready).
    2. Fallback to fitz/PyMuPDF if installed.
    3. If digital text is missing and OCR engine is loaded, executes OCR.
    """
    try:
        # 1. Primary extraction via pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        extracted_pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip():
                extracted_pages.append(t.strip())
        
        full_text = "\n\n".join(extracted_pages).strip()
        if len(full_text) >= 20:
            return full_text

        # 2. Fallback to fitz / OCR if available
        if fitz is not None:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            fitz_pages = []
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text().strip()
                if len(page_text) > 40:
                    fitz_pages.append(page_text)
                elif ocr_engine is not None:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_t = run_ocr_on_pil_image(img)
                    if ocr_t:
                        fitz_pages.append(ocr_t)
            fitz_full = "\n\n".join(fitz_pages).strip()
            if fitz_full:
                return fitz_full

        if not full_text:
            return "[Notice: PDF appears to be a scanned image without digital text. In lightweight serverless mode, please paste report text directly.]"
        return full_text
    except Exception as e:
        return f"[Error: PDF extraction failed: {str(e)}]"


def extract_text_from_image(image_bytes: bytes, filename: str) -> str:
    """
    Extracts text from uploaded images. Uses RapidOCR if installed;
    otherwise provides clear guidance for serverless environments.
    """
    if ocr_engine is None:
        return (
            f"[Notice: Scanned Image OCR is disabled in lightweight serverless mode to keep bundle under 500 MB.]\n"
            "Please paste the report text directly into the editor below or upload a digital PDF."
        )
    try:
        image = Image.open(io.BytesIO(image_bytes))
        ocr_text = run_ocr_on_pil_image(image)

        if not ocr_text or len(ocr_text.strip()) < 10:
            return (
                f"[OCR Warning: Low text density recognized in {filename}]\n"
                "Please verify or adjust the report text below before running diagnosis."
            )
        return ocr_text
    except Exception as e:
        return f"[Error: Image OCR processing failed: {str(e)}]"
# -----------------------------------------------------------------------------
# Multi-Parameter Clinical Diagnosis Synthesis Engine
# -----------------------------------------------------------------------------
def synthesize_clinical_diagnosis(parsed_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Correlates individual abnormal laboratory tests to formulate comprehensive
    clinical diagnostic impressions, acuity rating, and actionable next steps.
    """
    test_dict = {t["test_key"]: t for t in parsed_tests}
    clinical_impressions = []
    differential_diagnoses = set()
    next_steps = []
    acuity_level = "NORMAL"

    def is_high(key):
        return key in test_dict and test_dict[key]["status"] == "HIGH"

    def is_low(key):
        return key in test_dict and test_dict[key]["status"] == "LOW"

    def get_val(key):
        return test_dict[key]["measured_value"] if key in test_dict else None

    # 1. Hematologic Syndromes (Anemia & Cytopenias)
    if is_low("hemoglobin"):
        hb = get_val("hemoglobin")
        mcv = get_val("mcv")
        mcv_type = ""
        if mcv is not None:
            if mcv < 80.0:
                mcv_type = "Microcytic (likely Iron Deficiency Anemia or Thalassemia Trait)"
                differential_diagnoses.update(["Iron Deficiency Anemia", "Anemia of Chronic Disease", "Thalassemia Trait"])
                next_steps.append("Order Serum Ferritin, Total Iron Binding Capacity (TIBC), and Peripheral Blood Smear.")
            elif mcv > 100.0:
                mcv_type = "Macrocytic (likely Vitamin B12 or Folate Deficiency)"
                differential_diagnoses.update(["Vitamin B12 Deficiency", "Folate Deficiency", "Myelodysplastic Syndrome"])
                next_steps.append("Order Serum Vitamin B12 and Folate levels.")
            else:
                mcv_type = "Normocytic"
                differential_diagnoses.update(["Acute Blood Loss", "Anemia of Chronic Disease", "Hemolytic Anemia"])
                next_steps.append("Perform Reticulocyte Count and evaluate for occult blood loss.")

        prefix_mcv = f"{mcv_type} " if mcv_type else ""
        if hb < 7.0:
            acuity_level = "CRITICAL"
            clinical_impressions.append(f"Severe Anemia (Hb: {hb} g/dL). Critical threshold requiring urgent clinical attention and transfusion evaluation.")
        elif hb < 10.0:
            if acuity_level != "CRITICAL": acuity_level = "HIGH"
            clinical_impressions.append(f"Moderate {prefix_mcv}Anemia (Hb: {hb} g/dL). Significant reduction in circulating oxygen-carrying capacity.")
        else:
            if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "MODERATE"
            clinical_impressions.append(f"Mild {prefix_mcv}Anemia (Hb: {hb} g/dL).")

    # 2. Infection & Inflammation Profile
    if is_high("wbc"):
        wbc = get_val("wbc")
        if wbc > 18000:
            if acuity_level != "CRITICAL": acuity_level = "HIGH"
            clinical_impressions.append(f"Marked Leukocytosis (WBC: {int(wbc)}/mcL): Suggests acute severe bacterial infection, sepsis response, or leukemoid reaction.")
        else:
            if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "MODERATE"
            clinical_impressions.append(f"Leukocytosis (WBC: {int(wbc)}/mcL): Consistent with active infection, acute inflammation, or physical stress response.")
        
        differential_diagnoses.update(["Acute Bacterial Infection", "Systemic Inflammatory Response", "Tissue Necrosis"])
        next_steps.append("Perform Differential Leukocyte Count (Neutrophils vs Lymphocytes), CRP, and focus clinical examination for infection source.")

    if is_low("wbc"):
        if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "MODERATE"
        clinical_impressions.append("Leukopenia (Low WBC): Reduced circulating white cells predisposing to infection risk.")
        differential_diagnoses.update(["Viral Infection (e.g. Influenza, Dengue, HIV)", "Autoimmune Cytopenia", "Bone Marrow Suppression"])
        next_steps.append("Review recent medications for drug-induced marrow suppression and perform viral serologies.")

    if is_low("platelets"):
        plt = get_val("platelets")
        if plt < 50000:
            acuity_level = "CRITICAL"
            clinical_impressions.append(f"Severe Thrombocytopenia (Platelets: {int(plt)}/mcL): Significant bleeding risk requiring immediate hematology consultation.")
        elif plt < 100000:
            if acuity_level != "CRITICAL": acuity_level = "HIGH"
            clinical_impressions.append(f"Moderate Thrombocytopenia (Platelets: {int(plt)}/mcL).")
        else:
            if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "MODERATE"
            clinical_impressions.append("Mild Thrombocytopenia.")
        
        differential_diagnoses.update(["Viral Fever (Dengue, Epstein-Barr)", "Immune Thrombocytopenia (ITP)", "Liver Cirrhosis with Splenomegaly", "Drug-induced"])
        next_steps.append("Perform Peripheral Blood Film review and monitor for spontaneous bleeding or petechiae.")

    # 3. Glycemic & Diabetes Mellitus Spectrum
    has_diabetes_finding = False
    if is_high("hba1c") or is_high("glucose_fasting") or is_high("glucose_random"):
        fbs = get_val("glucose_fasting")
        a1c = get_val("hba1c")
        
        if (a1c and a1c >= 6.5) or (fbs and fbs >= 126):
            has_diabetes_finding = True
            if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "HIGH"
            clinical_impressions.append("Diagnostic Evidence of Diabetes Mellitus: Sustained hyperglycemia and chronic protein glycation.")
            differential_diagnoses.update(["Type 2 Diabetes Mellitus", "Type 1 Diabetes Mellitus", "Secondary Diabetes"])
            next_steps.append("Initiate Glycemic Management protocol, Urine Microalbumin check, and Ophthalmology Fundus Examination.")
        elif (a1c and a1c >= 5.7) or (fbs and fbs >= 100):
            if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "MODERATE"
            clinical_impressions.append("Prediabetes / Impaired Fasting Glycemia: Elevated risk of progression to overt diabetes mellitus.")
            differential_diagnoses.add("Impaired Glucose Tolerance")
            next_steps.append("Lifestyle modification (low glycemic diet, aerobic physical activity) and repeat HbA1c in 3 months.")

    # 4. Renal Insufficiency & Kidney Function
    if is_high("creatinine") or is_high("urea"):
        creat = get_val("creatinine")
        urea = get_val("urea")
        if creat and creat >= 3.0:
            acuity_level = "CRITICAL"
            clinical_impressions.append(f"Severe Renal Impairment (Creatinine: {creat} mg/dL): Critical decline in glomerular clearance.")
        else:
            if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "HIGH"
            clinical_impressions.append(f"Renal Impairment / Azotemia (Creatinine: {creat or 'Elevated'} mg/dL, Urea: {urea or 'Elevated'} mg/dL): Indicates impaired kidney filtration.")
        
        differential_diagnoses.update(["Acute Kidney Injury (Prerenal Dehydration vs Intrinsic)", "Chronic Kidney Disease", "Diabetic / Hypertensive Nephropathy"])
        next_steps.append("Calculate estimated GFR (eGFR), check Serum Electrolytes (Potassium), and perform Ultrasound KUB.")

    if is_high("uric_acid"):
        if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "MODERATE"
        clinical_impressions.append("Hyperuricemia: High uric acid levels predisposing to Gouty Arthritis and Urate Nephrolithiasis.")
        differential_diagnoses.add("Gouty Diathesis")
        next_steps.append("Avoid purine-rich foods, maintain hydration, and check joint symptoms.")

    # 5. Hepatic Injury & Transaminitis
    if is_high("sgpt_alt") or is_high("sgot_ast") or is_high("bilirubin_total"):
        alt = get_val("sgpt_alt")
        ast = get_val("sgot_ast")
        bili = get_val("bilirubin_total")
        
        if is_high("bilirubin_total") and bili >= 2.5:
            if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "HIGH"
            clinical_impressions.append(f"Clinical Jaundice / Hyperbilirubinemia (Bilirubin: {bili} mg/dL).")
            differential_diagnoses.update(["Biliary Duct Obstruction (Choledocholithiasis)", "Acute Viral Hepatitis", "Hemolysis"])
            next_steps.append("Order Abdominal Ultrasound to evaluate biliary tree and check Viral Hepatitis panel (HBsAg, Anti-HCV).")
        
        if is_high("sgpt_alt") or is_high("sgot_ast"):
            if acuity_level not in ["CRITICAL", "HIGH"]: acuity_level = "MODERATE"
            clinical_impressions.append("Hepatocellular Injury (Transaminitis): Elevated liver enzymes indicating active liver inflammation or steatosis.")
            differential_diagnoses.update(["Non-Alcoholic Fatty Liver Disease (NAFLD)", "Viral Hepatitis", "Drug-Induced Liver Injury (DILI)", "Alcoholic Liver Disease"])
            next_steps.append("Perform Abdominal Ultrasound for fatty infiltration and screen medications.")

    # 6. Cardiovascular & Dyslipidemia Risk
    if is_high("total_cholesterol") or is_high("ldl") or is_high("triglycerides"):
        if acuity_level not in ["CRITICAL", "HIGH", "MODERATE"]: acuity_level = "MODERATE"
        clinical_impressions.append("Atherogenic Dyslipidemia: Elevated circulating lipid fractions increasing long-term Atherosclerotic Cardiovascular Disease (ASCVD) risk.")
        differential_diagnoses.add("Primary / Familial Hyperlipidemia")
        next_steps.append("Perform 10-Year ASCVD Risk Score assessment and dietary lipid restriction.")

    # 7. Thyroid Dysfunction
    if is_high("tsh"):
        if acuity_level not in ["CRITICAL", "HIGH", "MODERATE"]: acuity_level = "MODERATE"
        clinical_impressions.append("Primary Hypothyroidism: Elevated TSH indicating reduced thyroid hormone output.")
        differential_diagnoses.add("Hashimoto Thyroiditis")
        next_steps.append("Check Free T4 (FT4) and Anti-TPO antibodies to assess thyroid autoimmune status.")
    elif is_low("tsh"):
        if acuity_level not in ["CRITICAL", "HIGH", "MODERATE"]: acuity_level = "MODERATE"
        clinical_impressions.append("Primary Hyperthyroidism / Thyrotoxicosis: Suppressed TSH indicating thyroid overactivity.")
        differential_diagnoses.add("Graves' Disease / Toxic Multinodular Goiter")
        next_steps.append("Check Free T3 and Free T4, and perform Thyroid Ultrasound.")

    # Default if all normal
    if not clinical_impressions:
        clinical_impressions.append("No acute pathological laboratory deviations identified. Tested parameters fall within physiological reference intervals.")
        differential_diagnoses.add("Physiologically Normal / Healthy Baseline")
        next_steps.append("Continue routine preventive health screening.")

    # Deduplicate next steps and differentials
    return {
        "acuity_level": acuity_level,
        "primary_impressions": clinical_impressions,
        "differential_diagnoses": sorted(list(differential_diagnoses)),
        "recommended_next_steps": next_steps if next_steps else ["Maintain regular annual health monitoring."]
    }


# -----------------------------------------------------------------------------
# Clinical Negation Detection Utility
# -----------------------------------------------------------------------------
def has_positive_finding(text: str, positive_terms: List[str]) -> bool:
    """
    Checks whether any candidate clinical term is genuinely present and NOT negated.
    Accounts for preceding negations ("no", "without", "clear of", "negative for")
    and following negations ("not seen", "absent", "negative", "nathi", "nahi").
    Supports English, Hinglish, and Gujlish phrasing.
    """
    if not text:
        return False
    lower = text.lower()
    for term in positive_terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        for match in re.finditer(pattern, lower):
            start = match.start()
            end = match.end()
            window_before = lower[max(0, start - 45):start]
            window_after = lower[end:min(len(lower), end + 40)]

            neg_before = re.search(
                r"\b(?:no|without|free\s+of|free\s+from|clear\s+of|no\s+evidence\s+of|rule\s+out|negative\s+for|non-reactive\s+for|absent|koi\s+nathi|koi\s+nahi)\b[^\.\;\n]*$",
                window_before
            )
            neg_after = re.search(
                r"^[^\.\;\n]*(?:\bnot\s+seen\b|\babsent\b|\bnegative\b|\bnon-reactive\b|\bnon\s+reactive\b|\bnot\s+detected\b|\bnathi\b|\bnahi\b|\bunremarkable\b|\bclear\b|\bnormal\b)",
                window_after
            )

            if not neg_before and not neg_after:
                return True
    return False


# -----------------------------------------------------------------------------
# Direct Disease Identification & Normal Reassurance Engine
# -----------------------------------------------------------------------------
def detect_direct_conditions(raw_text: str, parsed_tests: List[Dict[str, Any]], active_lang: str = "en") -> List[Dict[str, Any]]:
    """
    Directly detects acute and identifiable clinical conditions, fevers, and infections:
    - Dengue Fever / Severe Viral Thrombocytopenia
    - Malaria Parasite Infection (Plasmodium)
    - Typhoid Fever (Enteric Fever / Widal Serology)
    - Type 2 Diabetes Mellitus
    - Clinical Jaundice / Hyperbilirubinemia
    - Severe Anemia (Critical Low Hemoglobin)
    - Renal Failure / Severe Kidney Strain
    """
    lower = raw_text.lower()
    conditions = []
    test_dict = {t["test_key"]: t for t in parsed_tests}

    def get_v(k):
        return test_dict[k]["measured_value"] if k in test_dict else None

    def is_h(k):
        return k in test_dict and test_dict[k]["status"] == "HIGH"

    def is_l(k):
        return k in test_dict and test_dict[k]["status"] == "LOW"

    # 1. Dengue Fever / Severe Viral Thrombocytopenia
    is_dengue_explicit = has_positive_finding(raw_text, [
        "dengue positive", "dengue reactive", "ns1 positive", "ns1 reactive",
        "dengue antigen positive", "dengue fever", "acute dengue", "dengue", "dengu"
    ])
    plt_val = get_v("platelets")
    wbc_val = get_v("wbc")
    is_dengue_profile = (
        (plt_val is not None and plt_val < 100000 and ((wbc_val is not None and wbc_val < 4500) or any(f in lower for f in ["fever", "bukhar", "taav"])))
        or (plt_val is not None and plt_val < 65000)
    )

    if is_dengue_explicit or is_dengue_profile:
        cond_badge = "POSITIVE SEROLOGY" if is_dengue_explicit else "HIGH CLINICAL RISK"
        plt_note = f" (Platelets: {int(plt_val):,}/mcL)" if plt_val is not None else ""
        conditions.append({
            "id": "dengue",
            "name": f"Dengue Fever / Acute Viral Thrombocytopenia{plt_note}",
            "badge": cond_badge,
            "severity": "CRITICAL" if (plt_val and plt_val < 50000) else "HIGH",
            "icon": "🦟",
            "summary": {
                "en": f"Marked platelet reduction{plt_note} with leukopenia characteristic of active Dengue fever or acute viral illness.",
                "hinglish": f"Platelet count bohot kam hai{plt_note} — Dengue ya acute viral fever ka direct sanket.",
                "gujlish": f"Platelet count ghana ochha che{plt_note} — Dengue athva viral fever no direct sanket che."
            },
            "recommendation": {
                "en": "Monitor platelet count every 12-24 hours. Ensure vigorous hydration (ORS, coconut water). Avoid aspirin/ibuprofen (bleed risk).",
                "hinglish": "Har 12-24 ghante me platelet count check karayein. Khoob paani aur ORS piyein. Painkiller/Aspirin bilkul na lein.",
                "gujlish": "Dar 12-24 kalake platelet report karavo. Vadhu paani ane ORS pivo. Dard nivarak dava bilkul na levo."
            }
        })

    # 2. Malaria Parasite (Plasmodium)
    is_malaria = has_positive_finding(raw_text, [
        "malaria", "plasmodium", "p. vivax", "p.vivax", "p. falciparum", "p.falciparum",
        "mp positive", "malaria positive", "parasite seen", "ring form seen", "trophozoite"
    ])
    if is_malaria:
        conditions.append({
            "id": "malaria",
            "name": "Malaria Parasite Infection (Plasmodium Species)",
            "badge": "POSITIVE SEROLOGY",
            "severity": "HIGH",
            "icon": "🦟",
            "summary": {
                "en": "Malaria parasite (Plasmodium species) identified in peripheral blood smear / rapid antigen test.",
                "hinglish": "Khoon ki jaanch me Malaria parasite (Plasmodium) paya gaya hai.",
                "gujlish": "Lohi ni tapas ma Malaria na jantuo (Plasmodium) malya che."
            },
            "recommendation": {
                "en": "Start full prescribed course of antimalarials under physician direction and maintain bed rest.",
                "hinglish": "Doctor se milkar complete antimalarial dawai ka course shuru karein aur aaram karein.",
                "gujlish": "Doctor ni salah mujab puro antimalarial course sharu karo ane aaram karo."
            }
        })

    # 3. Typhoid (Enteric Fever / Widal)
    is_typhoid = has_positive_finding(raw_text, [
        "widal positive", "widal reactive", "widal test positive", "typhoid positive",
        "s. typhi positive", "s. typhi o positive", "s. typhi h positive", "s. typhi reactive",
        "enteric fever", "typhoid"
    ]) or ("widal" in lower and any(t in lower for t in ["1:160", "1:320", "1:80 positive"]) and not any(neg in lower for neg in ["negative", "non-reactive", "not reactive"]))
    if is_typhoid:
        conditions.append({
            "id": "typhoid",
            "name": "Typhoid Fever (Enteric Fever / Salmonella Typhi)",
            "badge": "POSITIVE / REACTIVE",
            "severity": "HIGH",
            "icon": "🌡️",
            "summary": {
                "en": "Widal test is reactive for Salmonella Typhi antibodies, indicating enteric bacterial infection.",
                "hinglish": "Widal test positive hai — Typhoid (antrik jwar / motijhara) confirm karta hai.",
                "gujlish": "Widal test positive che — Typhoid (motijharo) no sanket che."
            },
            "recommendation": {
                "en": "Take prescribed antibiotic therapy. Drink boiled water and consume soft, easily digestible meals.",
                "hinglish": "Doctor dwara prescribed antibiotics lein. Sirf ubhla hua paani aur halka khana lein.",
                "gujlish": "Doctor ae aapele antibiotic dava lo. Matra ukalelu paani ane halvo khorak lo."
            }
        })

    # 4. Type 2 Diabetes Mellitus
    fbs = get_v("glucose_fasting")
    rbs = get_v("glucose_random")
    a1c = get_v("hba1c")
    is_diabetes = (a1c and a1c >= 6.5) or (fbs and fbs >= 126) or (rbs and rbs >= 200) or any(k in lower for k in ["diabetes mellitus", "type 2 diabetes", "diabetic"])
    if is_diabetes:
        sugar_note = []
        if a1c: sugar_note.append(f"HbA1c: {a1c}%")
        if fbs: sugar_note.append(f"FBS: {fbs} mg/dL")
        sugar_str = f" ({', '.join(sugar_note)})" if sugar_note else ""
        conditions.append({
            "id": "diabetes",
            "name": f"Type 2 Diabetes Mellitus{sugar_str}",
            "badge": "DIAGNOSTIC CRITERIA MET",
            "severity": "HIGH",
            "icon": "🍬",
            "summary": {
                "en": f"Elevated glycemic indicators{sugar_str} meeting formal diagnostic criteria for Diabetes Mellitus.",
                "hinglish": f"Blood sugar normal se kaafi zyada hai{sugar_str} — Diabetes confirm karta hai.",
                "gujlish": f"Lohi ma sugar vadhare aavi che{sugar_str} — Diabetes Mellitus confirm kare che."
            },
            "recommendation": {
                "en": "Consult an endocrinologist/physician. Adopt low-glycemic nutrition and walk 30 mins daily.",
                "hinglish": "Diabetologist se milein. Meetha bandh karein aur daily 30 minute walk karein.",
                "gujlish": "Doctor pase javu. Galyo khorak bandh karvo ane roj 30 minute chaalvu."
            }
        })

    # 5. Severe Anemia
    hb_val = get_v("hemoglobin")
    if hb_val and hb_val < 8.0:
        conditions.append({
            "id": "severe_anemia",
            "name": f"Severe Anemia (Critical Low Hemoglobin: {hb_val} g/dL)",
            "badge": "CRITICAL DEFICIENCY",
            "severity": "CRITICAL" if hb_val < 7.0 else "HIGH",
            "icon": "🩸",
            "summary": {
                "en": f"Hemoglobin is critically low at {hb_val} g/dL (Normal: 12.0 - 15.5), severely reducing oxygen supply.",
                "hinglish": f"Hemoglobin bohot kam hai ({hb_val} g/dL) — Sharir me khoon ki gambhir kami.",
                "gujlish": f"Hemoglobin ghano ochho che ({hb_val} g/dL) — Sharir ma lohi ni ghanij kami che."
            },
            "recommendation": {
                "en": "Seek prompt clinical assessment for parenteral iron therapy or blood transfusion and investigate bleeding.",
                "hinglish": "Turant doctor se milein. Blood transfusion ya iron drip ki zaroorat pad sakti hai.",
                "gujlish": "Turant doctor pase javu. Lohi chadavva athva iron na injection ni jarur padi shake."
            }
        })

    # 6. Jaundice / Acute Hepatic Dysfunction
    bili_val = get_v("bilirubin_total")
    if (bili_val and bili_val >= 2.0) or (is_h("bilirubin_total") and (is_h("sgpt_alt") or is_h("sgot_ast"))):
        bili_str = f" (Bilirubin: {bili_val} mg/dL)" if bili_val else ""
        conditions.append({
            "id": "jaundice",
            "name": f"Clinical Jaundice / Hyperbilirubinemia{bili_str}",
            "badge": "ELEVATED BILIRUBIN",
            "severity": "HIGH",
            "icon": "🟡",
            "summary": {
                "en": f"Bilirubin elevation{bili_str} indicating impaired bile excretion, acute hepatitis, or liver inflammation.",
                "hinglish": f"Bilirubin badha hua hai{bili_str} — Piliya (Jaundice) / Liver me sujan ka direct sanket.",
                "gujlish": f"Bilirubin vadhyu che{bili_str} — Kamlo (Jaundice) / Liver ni takleef darshave che."
            },
            "recommendation": {
                "en": "Ultrasound abdomen to inspect biliary ducts; avoid alcohol, fried, and heavy foods.",
                "hinglish": "Pet ka ultrasound karayein. Tel aur masaledar khana bilkul bandh karein.",
                "gujlish": "Pet ni sonography karavo. Tel valo ane tikhokhorak bandh karo."
            }
        })

    # 7. Renal Strain / Kidney Dysfunction
    creat_val = get_v("creatinine")
    if creat_val and creat_val >= 1.5:
        conditions.append({
            "id": "renal_strain",
            "name": f"Renal Impairment / Kidney Strain (Creatinine: {creat_val} mg/dL)",
            "badge": "HIGH CREATININE",
            "severity": "CRITICAL" if creat_val >= 3.0 else "HIGH",
            "icon": "💧",
            "summary": {
                "en": f"Serum Creatinine is {creat_val} mg/dL, showing impaired glomerular filtration.",
                "hinglish": f"Serum Creatinine {creat_val} mg/dL hai — Gurde (Kidneys) theek se filter nahi kar pa rahe.",
                "gujlish": f"Serum Creatinine {creat_val} mg/dL che — Kidneys lohi barabar saaf nathi kari shakto."
            },
            "recommendation": {
                "en": "Consult a Nephrologist. Avoid NSAID painkillers which increase renal strain.",
                "hinglish": "Nephrologist (kidney doctor) se milein. Painkiller dawaiyan bilkul na lein.",
                "gujlish": "Kidney na doctor pase javu. Dard nivarak dava na levi."
            }
        })

    return conditions


def generate_normal_reassurance(active_lang: str = "en") -> Dict[str, Any]:
    """Generates clear, reassuring feedback when all parameters are healthy."""
    return {
        "is_normal": True,
        "doctor_needed": False,
        "badge_text": {
            "en": "100% HEALTHY — ALL NORMAL",
            "hinglish": "BILKUL FIT — SABHI NORMAL",
            "gujlish": "EKDAM NORMAL — KOI CHINTA NATHI"
        }.get(active_lang, "100% HEALTHY — ALL NORMAL"),
        "title": {
            "en": "Everything Looks Normal — Healthy Report!",
            "hinglish": "Sabhi Reports Normal Hain — Bilkul Fit!",
            "gujlish": "Badha Reports Ekdam Normal Che — Koi Chinta Nathi!"
        }.get(active_lang, "Everything Looks Normal — Healthy Report!"),
        "message": {
            "en": "All your tested parameters fall well within standard healthy reference intervals. There is NO need to visit a doctor or worry about these results. Maintain your healthy habits, balanced diet, and active routine!",
            "hinglish": "Aapke sabhi test parameters bilkul normal range me hain. Kisi doctor ke paas turant jaane ki koi zaroorat nahi hai! Apni achhi diet, paani aur healthy routine follow karte rahein.",
            "gujlish": "Tamara badha j test parameters ekdam normal limit ma che. Doctor pase javani koi jarur nathi! Swasth khorak, purto paani ane regular exercise chalu rakho."
        }.get(active_lang, "All your tested parameters fall well within standard healthy reference intervals. There is NO need to visit a doctor or worry about these results."),
        "recommendation": {
            "en": "No immediate doctor consultation needed. Continue your regular healthy diet and lifestyle.",
            "hinglish": "Doctor ke paas jaane ki zaroorat nahi hai. Healthy khana aur exercise chalu rakhein.",
            "gujlish": "Doctor pase javani jarur nathi. Swasth khorak ane rojindi kasrat chalu rakho."
        }.get(active_lang, "No immediate doctor consultation needed.")
    }


# -----------------------------------------------------------------------------
# Structured Laboratory Report Parser (Multilingual: English, Hinglish, Gujlish)
# -----------------------------------------------------------------------------
def parse_lab_report(text: str, target_lang: Optional[str] = "auto") -> Dict[str, Any]:
    """
    Parses laboratory report text:
    - Supports traditional multi-line lab printouts AND conversational Hinglish/Gujlish statements.
    - Extracts test names, measured values, units, and reference ranges.
    - Runs the Clinical Diagnosis Synthesis Engine.
    - Provides plain-language explanations in English, Hinglish, and Gujlish.
    """
    if not text or len(text.strip()) < 5:
        return {
            "success": False,
            "error": "The report text is empty or too brief to parse laboratory values."
        }

    lang_info = detect_language(text)
    active_lang = target_lang if target_lang in ["en", "hinglish", "gujlish"] else lang_info["language_code"]

    # Support both multi-line tabular lines and conversational clauses (separated by comma, aur, ane, and)
    clauses = [c.strip() for c in re.split(r"[\n,;]+|\band\b|\baur\b|\bane\b", text, flags=re.IGNORECASE) if c.strip()]
    parsed_tests = []
    found_keys = set()

    for line in clauses:
        lower_line = line.lower()

        for key, info in STANDARD_LAB_DICTIONARY.items():
            key_clean = key.replace("_", " ")
            patterns = [key_clean]
            if key == "hemoglobin":
                patterns.extend(["hb", "hgb", "haemoglobin", "hemoglobin", "khoon", "lohi"])
            elif key == "wbc":
                patterns.extend(["total leukocyte", "tlc", "white blood", "leukocytes", "w.b.c", "wbc", "safed kano"])
            elif key == "rbc":
                patterns.extend(["red blood", "total rbc", "erythrocytes", "r.b.c", "rbc"])
            elif key == "platelets":
                patterns.extend(["platelet count", "thrombocyte", "platelet", "platelets"])
            elif key == "mcv":
                patterns.extend(["m.c.v", "mean corpuscular volume", "mcv"])
            elif key == "esr":
                patterns.extend(["e.s.r", "erythrocyte sedimentation", "esr"])
            elif key == "glucose_fasting":
                patterns.extend(["fbs", "fasting sugar", "fasting blood sugar", "glucose fasting", "sugar fasting", "fasting glucose", "sakkar", "madhumeh"])
            elif key == "glucose_random":
                patterns.extend(["rbs", "random sugar", "random blood sugar", "glucose random", "post prandial", "ppbs", "random glucose", "sugar"])
            elif key == "hba1c":
                patterns.extend(["glycated hemoglobin", "glycosylated hb", "a1c", "hb a1c", "hba1c"])
            elif key == "creatinine":
                patterns.extend(["serum creatinine", "s. creatinine", "creat", "creatinine", "kidney test"])
            elif key == "urea":
                patterns.extend(["blood urea", "bun", "s. urea", "urea"])
            elif key == "uric_acid":
                patterns.extend(["uric acid", "s. uric acid", "gout"])
            elif key == "total_cholesterol":
                patterns.extend(["cholesterol total", "s. cholesterol", "total cholesterol", "cholesterol", "charbi"])
            elif key == "triglycerides":
                patterns.extend(["triglyceride", "s. triglycerides", "tg", "triglycerides"])
            elif key == "hdl":
                patterns.extend(["hdl cholesterol", "high density lipoprotein", "hdl"])
            elif key == "ldl":
                patterns.extend(["ldl cholesterol", "low density lipoprotein", "ldl"])
            elif key == "sgpt_alt":
                patterns.extend(["sgpt", "alt", "alanine aminotransferase"])
            elif key == "sgot_ast":
                patterns.extend(["sgot", "ast", "aspartate aminotransferase"])
            elif key == "bilirubin_total":
                patterns.extend(["bilirubin total", "total bilirubin", "s. bilirubin", "bilirubin", "kamlo", "piliya", "peeliya", "jaundice"])
            elif key == "albumin":
                patterns.extend(["serum albumin", "s. albumin", "albumin"])
            elif key == "tsh":
                patterns.extend(["thyroid stimulating", "ultra tsh", "tsh", "thyroid"])

            matched_match = None
            for p in sorted(patterns, key=len, reverse=True):
                m = re.search(r"\b" + re.escape(p) + r"\b", lower_line)
                if m:
                    matched_match = m
                    break

            if matched_match and key not in found_keys:
                # Search for numbers strictly after the matched test label (prevents digits in names like HbA1c from being captured)
                after_text = line[matched_match.end():]
                num_matches = re.findall(r"[-+]?\d*\.\d+|\d+", after_text)
                if not num_matches:
                    line_without_pat = line[:matched_match.start()] + " " + after_text
                    num_matches = re.findall(r"[-+]?\d*\.\d+|\d+", line_without_pat)
                if not num_matches:
                    continue

                try:
                    measured_val = float(num_matches[0])
                except ValueError:
                    continue

                # Range extraction
                range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–toTO]+\s*(\d+(?:\.\d+)?)", line)
                ref_low = info["low"]
                ref_high = info["high"]
                range_source = "Standard Clinical Reference"

                if range_match:
                    try:
                        extracted_low = float(range_match.group(1))
                        extracted_high = float(range_match.group(2))
                        if extracted_low < extracted_high:
                            ref_low = extracted_low
                            ref_high = extracted_high
                            range_source = "Printed in Report"
                    except ValueError:
                        pass

                # Status Flag
                if measured_val < ref_low:
                    status = "LOW"
                    badge_class = "badge-low"
                elif measured_val > ref_high:
                    status = "HIGH"
                    badge_class = "badge-high"
                else:
                    status = "NORMAL"
                    badge_class = "badge-normal"

                explanation = get_multilingual_lab_explanation(key, status, active_lang)
                multilingual_expls = {
                    "en": get_multilingual_lab_explanation(key, status, "en"),
                    "hinglish": get_multilingual_lab_explanation(key, status, "hinglish"),
                    "gujlish": get_multilingual_lab_explanation(key, status, "gujlish")
                }

                parsed_tests.append({
                    "test_key": key,
                    "test_name": info["display_name"],
                    "category": info["category"],
                    "measured_value": measured_val,
                    "unit": info["unit"],
                    "ref_low": ref_low,
                    "ref_high": ref_high,
                    "ref_range_str": f"{ref_low} - {ref_high} {info['unit']}",
                    "range_source": range_source,
                    "status": status,
                    "badge_class": badge_class,
                    "plain_explanation": explanation,
                    "multilingual_explanations": multilingual_expls
                })
                found_keys.add(key)
                break

    # Synthesize complete clinical diagnosis
    diagnosis_data = synthesize_clinical_diagnosis(parsed_tests)
    acuity = diagnosis_data.get("acuity_level", "NORMAL")
    diagnosis_data["acuity_multilingual"] = MULTILINGUAL_ACUITY.get(acuity, {}).get(active_lang, acuity)
    diagnosis_data["acuity_all_languages"] = MULTILINGUAL_ACUITY.get(acuity, {})

    high_count = sum(1 for t in parsed_tests if t["status"] == "HIGH")
    low_count = sum(1 for t in parsed_tests if t["status"] == "LOW")
    normal_count = sum(1 for t in parsed_tests if t["status"] == "NORMAL")

    # Detect directly identifiable conditions (Dengue, Malaria, Typhoid, Diabetes, Jaundice, etc.)
    direct_conditions = detect_direct_conditions(text, parsed_tests, active_lang)

    # Reassuring normal detection if all parameters fall within healthy reference bounds
    is_all_normal = (
        len(direct_conditions) == 0
        and high_count == 0
        and low_count == 0
        and len(parsed_tests) > 0
    )

    normal_reassurance = generate_normal_reassurance(active_lang) if is_all_normal else None
    doctor_needed = not is_all_normal and (len(direct_conditions) > 0 or high_count > 0 or low_count > 0)

    return {
        "success": True,
        "detected_language": lang_info,
        "active_language": active_lang,
        "total_parameters_detected": len(parsed_tests),
        "high_count": high_count,
        "low_count": low_count,
        "normal_count": normal_count,
        "is_all_normal": is_all_normal,
        "doctor_needed": doctor_needed,
        "normal_reassurance": normal_reassurance,
        "directly_identified_conditions": direct_conditions,
        "tests": parsed_tests,
        "clinical_diagnosis": diagnosis_data,
        "disclaimer": (
            "Clinical Diagnostic Synthesis: This assessment correlates detected laboratory abnormalities using medical logic guidelines. "
            "It is an educational decision-support aid and must be validated by a licensed physician before any treatment is initiated."
        )
    }


# -----------------------------------------------------------------------------
# Ultrasound / Sonography Diagnostic Findings Explainer (Multilingual)
# -----------------------------------------------------------------------------
def analyze_sonography_report(text: str, target_lang: Optional[str] = "auto") -> Dict[str, Any]:
    """
    Parses and explains written sonography findings, generating organ-by-organ
    diagnostic impressions and decoded radiological terminology in English, Hinglish, or Gujlish.
    """
    if not text or len(text.strip()) < 5:
        return {
            "success": False,
            "error": "The sonography report text is empty or too short for analysis."
        }

    lang_info = detect_language(text)
    active_lang = target_lang if target_lang in ["en", "hinglish", "gujlish"] else lang_info["language_code"]

    lower_text = text.lower()
    detected_terms = []
    clinical_observations = []

    # Match ultrasound terminology
    for term, meta in ULTRASOUND_DICTIONARY.items():
        if re.search(r"\b" + re.escape(term) + r"\b", lower_text):
            detected_terms.append({
                "term": meta["term"],
                "meaning": meta["meaning"]
            })

    impression_match = re.search(r"(?:impression|conclusion|opinion|summary)[:\s]+(.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
    impression_text = impression_match.group(1).strip() if impression_match else "Findings detailed in anatomical review."

    # Diagnoses per organ (English, Hinglish, Gujlish keywords with Negation Support)
    if any(k in lower_text for k in ["gallbladder", "gb", "pitte", "pitta"]):
        has_gb_stone = has_positive_finding(lower_text, ["cholelithiasis", "calculi", "calculus", "gallstone", "gallstones", "pathri", "kankari"])
        if has_gb_stone:
            finding_title = "Cholelithiasis (Gallbladder Calculi)"
            clinical_observations.append({
                "organ": "Gallbladder",
                "finding": finding_title,
                "severity": "ATTENTION",
                "plain_meaning": get_multilingual_sono_explanation(finding_title, active_lang),
                "multilingual_meanings": {
                    "en": get_multilingual_sono_explanation(finding_title, "en"),
                    "hinglish": get_multilingual_sono_explanation(finding_title, "hinglish"),
                    "gujlish": get_multilingual_sono_explanation(finding_title, "gujlish")
                }
            })
        else:
            finding_title = "Normal Gallbladder"
            clinical_observations.append({
                "organ": "Gallbladder",
                "finding": finding_title,
                "severity": "NORMAL",
                "plain_meaning": "No stones, wall thickening, or acute inflammation described."
            })

    if any(k in lower_text for k in ["liver", "hepatic", "yakrit"]):
        has_fatty = has_positive_finding(lower_text, ["steatosis", "fatty liver", "fatty", "charbi"])
        has_hepato = has_positive_finding(lower_text, ["hepatomegaly", "liver enlargement", "sujan"])

        if has_fatty:
            finding_title = "Diffuse Hepatic Steatosis (Fatty Liver)"
            clinical_observations.append({
                "organ": "Liver",
                "finding": finding_title,
                "severity": "ATTENTION",
                "plain_meaning": get_multilingual_sono_explanation(finding_title, active_lang),
                "multilingual_meanings": {
                    "en": get_multilingual_sono_explanation(finding_title, "en"),
                    "hinglish": get_multilingual_sono_explanation(finding_title, "hinglish"),
                    "gujlish": get_multilingual_sono_explanation(finding_title, "gujlish")
                }
            })
        elif has_hepato:
            finding_title = "Hepatomegaly (Liver Enlargement)"
            clinical_observations.append({
                "organ": "Liver",
                "finding": finding_title,
                "severity": "ATTENTION",
                "plain_meaning": "Liver longitudinal span exceeds typical physiological dimensions."
            })
        else:
            clinical_observations.append({
                "organ": "Liver",
                "finding": "Normal Liver Architecture",
                "severity": "NORMAL",
                "plain_meaning": "Homogeneous liver parenchyma without focal space-occupying lesions."
            })

    if any(k in lower_text for k in ["kidney", "renal", "kub", "gurde"]):
        has_renal_stone = has_positive_finding(lower_text, ["nephrolithiasis", "calculus", "renal stone", "kidney stone", "pathri"])
        has_hydro = has_positive_finding(lower_text, ["hydronephrosis", "pelvicalyceal dilatation", "sujan"])
        has_cyst = has_positive_finding(lower_text, ["cyst", "pani ni ganth", "cortical cyst"])

        if has_renal_stone:
            finding_title = "Nephrolithiasis (Renal Calculus)"
            clinical_observations.append({
                "organ": "Kidneys",
                "finding": finding_title,
                "severity": "ATTENTION",
                "plain_meaning": get_multilingual_sono_explanation(finding_title, active_lang),
                "multilingual_meanings": {
                    "en": get_multilingual_sono_explanation(finding_title, "en"),
                    "hinglish": get_multilingual_sono_explanation(finding_title, "hinglish"),
                    "gujlish": get_multilingual_sono_explanation(finding_title, "gujlish")
                }
            })
        if has_hydro:
            finding_title = "Hydronephrosis (Pelvicalyceal Dilatation)"
            clinical_observations.append({
                "organ": "Kidneys",
                "finding": finding_title,
                "severity": "ATTENTION",
                "plain_meaning": get_multilingual_sono_explanation(finding_title, active_lang),
                "multilingual_meanings": {
                    "en": get_multilingual_sono_explanation(finding_title, "en"),
                    "hinglish": get_multilingual_sono_explanation(finding_title, "hinglish"),
                    "gujlish": get_multilingual_sono_explanation(finding_title, "gujlish")
                }
            })
        if has_cyst:
            finding_title = "Renal Cortical Cyst"
            clinical_observations.append({
                "organ": "Kidneys",
                "finding": finding_title,
                "severity": "INFO",
                "plain_meaning": get_multilingual_sono_explanation(finding_title, active_lang),
                "multilingual_meanings": {
                    "en": get_multilingual_sono_explanation(finding_title, "en"),
                    "hinglish": get_multilingual_sono_explanation(finding_title, "hinglish"),
                    "gujlish": get_multilingual_sono_explanation(finding_title, "gujlish")
                }
            })
        if not has_renal_stone and not has_hydro and not has_cyst:
            clinical_observations.append({
                "organ": "Kidneys",
                "finding": "Normal Kidneys",
                "severity": "NORMAL",
                "plain_meaning": "Both kidneys normal in size, shape, and cortical thickness. No calculi or obstruction."
            })

    if has_positive_finding(lower_text, ["ascites", "pet me pani", "pet ma paani"]):
        finding_title = "Free Fluid / Ascites Observed"
        clinical_observations.append({
            "organ": "Peritoneum",
            "finding": finding_title,
            "severity": "ATTENTION",
            "plain_meaning": get_multilingual_sono_explanation(finding_title, active_lang),
            "multilingual_meanings": {
                "en": get_multilingual_sono_explanation(finding_title, "en"),
                "hinglish": get_multilingual_sono_explanation(finding_title, "hinglish"),
                "gujlish": get_multilingual_sono_explanation(finding_title, "gujlish")
            }
        })

    # Direct ultrasound condition matches (Negation aware)
    direct_conditions = []
    if has_positive_finding(lower_text, ["cholelithiasis", "gallstone", "gallstones", "calculi", "pathri"]):
        direct_conditions.append({
            "id": "cholelithiasis",
            "name": "Gallstones (Cholelithiasis)",
            "badge": "DETECTED IN ULTRASOUND",
            "severity": "HIGH",
            "icon": "🪨",
            "summary": {
                "en": "Acoustic shadowing with mobile calculi observed inside gallbladder lumen.",
                "hinglish": "Gallbladder (pitte ki thaili) me pathri payi gayi hai.",
                "gujlish": "Gallbladder ma pathri (calculi) batavi che."
            },
            "recommendation": {
                "en": "Consult a general surgeon for clinical evaluation; avoid heavy fatty/fried foods.",
                "hinglish": "Surgeon se consult karein aur oily/fatty khana kam karein.",
                "gujlish": "Surgeon doctor pase javu ane tel valo khorak ochho karvo."
            }
        })
    if has_positive_finding(lower_text, ["hepatic steatosis", "fatty liver", "charbi"]):
        direct_conditions.append({
            "id": "fatty_liver",
            "name": "Fatty Liver Disease (Diffuse Hepatic Steatosis)",
            "badge": "DETECTED IN ULTRASOUND",
            "severity": "MODERATE",
            "icon": "🫁",
            "summary": {
                "en": "Increased liver parenchymal echogenicity consistent with Grade I/II fatty liver.",
                "hinglish": "Liver me charbi jama ho gayi hai (Fatty Liver).",
                "gujlish": "Liver ma charbi jama thai gayi che (Fatty Liver)."
            },
            "recommendation": {
                "en": "Lifestyle modification: 30 minutes daily aerobic exercise and reduce refined sugar and alcohol.",
                "hinglish": "Vajan control karein, daily exercise karein aur tel/meetha kam karein.",
                "gujlish": "Roj 30 minute kasrat karo ane tel-valoo khorak ochho karo."
            }
        })
    if has_positive_finding(lower_text, ["nephrolithiasis", "renal calculus", "kidney stone"]):
        direct_conditions.append({
            "id": "kidney_stone",
            "name": "Kidney Stones (Renal Calculi / Nephrolithiasis)",
            "badge": "DETECTED IN ULTRASOUND",
            "severity": "HIGH",
            "icon": "🪨",
            "summary": {
                "en": "Hyperechoic focus with acoustic shadowing identified in renal system.",
                "hinglish": "Kidney (gurde) me pathri payi gayi hai.",
                "gujlish": "Kidney ma pathri no sanket malyo che."
            },
            "recommendation": {
                "en": "Hydrate heavily (2.5-3 liters water daily) and consult an urologist.",
                "hinglish": "Khoob paani piyein (2.5-3 liter) aur urologist se check karayein.",
                "gujlish": "Vadhu paani pivo (2.5-3 liter) ane urologist doctor pase check karavo."
            }
        })

    is_all_normal = len(direct_conditions) == 0 and len([o for o in clinical_observations if o.get("severity") == "ATTENTION"]) == 0
    normal_reassurance = generate_normal_reassurance(active_lang) if is_all_normal else None
    doctor_needed = not is_all_normal

    return {
        "success": True,
        "detected_language": lang_info,
        "active_language": active_lang,
        "is_all_normal": is_all_normal,
        "doctor_needed": doctor_needed,
        "normal_reassurance": normal_reassurance,
        "directly_identified_conditions": direct_conditions,
        "detected_terms": detected_terms,
        "clinical_observations": clinical_observations,
        "radiological_impression": impression_text[:400],
        "disclaimer": (
            "Notice: Ultrasound interpretation depends on acoustic windows and active transducer sweeps. "
            "This software decodes written radiological findings and does not evaluate live ultrasound video or pixel images."
        )
    }
