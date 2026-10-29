import pytest
from src.report_analyzer import (
    parse_lab_report,
    analyze_sonography_report,
    STANDARD_LAB_DICTIONARY
)


def test_standard_lab_dictionary_contains_essentials():
    assert "hemoglobin" in STANDARD_LAB_DICTIONARY
    assert "glucose_fasting" in STANDARD_LAB_DICTIONARY
    assert "creatinine" in STANDARD_LAB_DICTIONARY
    assert "platelets" in STANDARD_LAB_DICTIONARY
    assert "urine_protein" in STANDARD_LAB_DICTIONARY


def test_unit_conversion_glucose_mmol_to_mgdl():
    # 6.8 mmol/L is approx 122.5 mg/dL
    report_text = "Fasting Blood Glucose: 6.8 mmol/L (Normal: 70 - 100 mg/dL)"
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert len(res["tests"]) == 1
    t = res["tests"][0]
    assert t["test_key"] == "glucose_fasting"
    assert abs(t["measured_value"] - 122.5) < 1.0


def test_unit_conversion_hemoglobin_gl_to_gdl():
    # 135 g/L is 13.5 g/dL
    report_text = "Hemoglobin (Hb): 135 g/L"
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert len(res["tests"]) == 1
    t = res["tests"][0]
    assert t["test_key"] == "hemoglobin"
    assert t["measured_value"] == 13.5
    assert t["status"] == "NORMAL"


def test_bounded_operator_parsing():
    # e.g., TSH: < 0.05
    report_text = "Thyroid Stimulating Hormone (TSH): < 0.05 uIU/mL"
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert len(res["tests"]) == 1
    t = res["tests"][0]
    assert t["test_key"] == "tsh"
    assert t["measured_value"] == 0.05
    assert t["status"] == "LOW"


def test_normal_report_generates_healthy_reassurance():
    report_text = """
    COMPLETE BLOOD COUNT & METABOLIC PANEL
    Hemoglobin: 14.5 g/dL (13.0 - 17.0)
    Fasting Blood Sugar: 90 mg/dL (70 - 100)
    Serum Creatinine: 0.9 mg/dL (0.6 - 1.2)
    Platelet Count: 250000 /mcL (150000 - 450000)
    Total Leukocyte Count (WBC): 6500 /mcL (4000 - 11000)
    """
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert res["is_all_normal"] is True
    assert res["doctor_needed"] is False
    assert res["normal_reassurance"] is not None
    assert "Healthy" in res["normal_reassurance"]["title"] or "Fit" in res["normal_reassurance"]["title"]
    assert len(res["directly_identified_conditions"]) == 0


def test_direct_disease_identification_diabetes():
    report_text = """
    Fasting Blood Sugar: 180 mg/dL
    HbA1c: 8.5 %
    Serum Creatinine: 1.0 mg/dL
    """
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert res["is_all_normal"] is False
    assert res["doctor_needed"] is True
    assert res["normal_reassurance"] is None

    # Must identify Type 2 Diabetes Mellitus
    cond_ids = [c["id"] for c in res["directly_identified_conditions"]]
    assert "diabetes" in cond_ids


def test_direct_disease_identification_dengue():
    report_text = """
    Platelet Count: 42000 /mcL
    WBC: 3100 /mcL
    Hemoglobin: 13.8 g/dL
    Patient experiencing high grade fever.
    """
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert res["is_all_normal"] is False
    assert res["normal_reassurance"] is None

    cond_ids = [c["id"] for c in res["directly_identified_conditions"]]
    assert "dengue" in cond_ids


def test_direct_disease_identification_severe_anemia():
    report_text = "Hemoglobin: 6.2 g/dL (Severe weakness, pallor)"
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert res["is_all_normal"] is False
    assert res["normal_reassurance"] is None

    cond_ids = [c["id"] for c in res["directly_identified_conditions"]]
    assert "severe_anemia" in cond_ids


def test_strict_mutual_exclusivity():
    # Both normal reassurance and identified conditions must NEVER coexist
    report_text = """
    Fasting Blood Sugar: 174 mg/dL
    Serum Creatinine: 2.1 mg/dL
    """
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert len(res["directly_identified_conditions"]) > 0
    # Crucial guarantee: is_all_normal is False, normal_reassurance is None
    assert res["is_all_normal"] is False
    assert res["normal_reassurance"] is None


def test_sonography_report_parsing_cholelithiasis():
    sono_text = """
    ULTRASOUND ABDOMEN & PELVIS
    Liver: Normal size and echotexture.
    Gallbladder: Well distended. Multiple hyperechoic mobile calculi seen, largest measuring 8mm, with posterior acoustic shadowing. No gallbladder wall thickening.
    Kidneys: Bilateral kidneys normal in size, shape, and cortical thickness.
    Impression: Cholelithiasis (Gallstones). No acute cholecystitis.
    """
    res = analyze_sonography_report(sono_text)
    assert res["success"] is True
    assert res["is_all_normal"] is False
    cond_ids = [c["id"] for c in res["directly_identified_conditions"]]
    assert "cholelithiasis" in cond_ids or "gallstones" in cond_ids


def test_direct_disease_identification_tuberculosis():
    report_text = """
    CLINICAL MOLECULAR PATHOLOGY
    Specimen: Sputum
    GeneXpert MTB/RIF: MTB DETECTED (High bacterial load)
    Sputum AFB Smear: Positive for Acid Fast Bacilli (2+)
    ESR: 65 mm/hr
    """
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert res["is_all_normal"] is False
    assert res["doctor_needed"] is True
    assert res["normal_reassurance"] is None

    cond_ids = [c["id"] for c in res["directly_identified_conditions"]]
    assert "tuberculosis" in cond_ids


def test_direct_disease_identification_bronchitis():
    report_text = """
    CHEST CLINIC EVALUATION
    Patient presenting with persistent cough and wheezing for 1 week.
    Clinical Impression: Acute Bronchitis with airway hyper-responsiveness.
    WBC: 11200 /mcL
    """
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert res["is_all_normal"] is False
    assert res["doctor_needed"] is True
    assert res["normal_reassurance"] is None

    cond_ids = [c["id"] for c in res["directly_identified_conditions"]]
    assert "bronchitis" in cond_ids


def test_truthful_normal_negation_not_flagging_tb():
    # When report explicitly says "no tuberculosis / negative for AFB", it must NOT flag TB
    report_text = """
    CHEST X-RAY & SPUTUM TEST
    Hemoglobin: 14.2 g/dL (13.0 - 17.0)
    Fasting Blood Sugar: 88 mg/dL (70 - 100)
    Serum Creatinine: 0.9 mg/dL (0.6 - 1.2)
    Sputum AFB: Negative for Acid Fast Bacilli.
    Chest X-Ray: Lung fields clear. No evidence of tuberculosis or bronchitis.
    """
    res = parse_lab_report(report_text)
    assert res["success"] is True
    assert res["is_all_normal"] is True
    assert res["doctor_needed"] is False
    assert res["normal_reassurance"] is not None
    assert len(res["directly_identified_conditions"]) == 0
