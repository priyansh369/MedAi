import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "ok"]
    assert "uptime_seconds" in data
    assert "models_loaded" in data


def test_api_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "ok"]
    assert "models_loaded" in data


def test_security_and_timing_headers():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert "x-process-time-ms" in response.headers


def test_model_metrics_endpoint():
    response = client.get("/api/model-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "evaluation_dataset" in data or "models" in data


def test_dataset_info_endpoint():
    response = client.get("/api/dataset-info")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data or "total_abstracts" in data


def test_sample_texts_endpoint():
    response = client.get("/api/sample-texts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_classify_endpoint_valid_medical_abstract():
    payload = {
        "text": "A randomized controlled trial of intravenous thrombolytic therapy in acute myocardial infarction demonstrated significant reduction in cardiac mortality and restoration of coronary blood flow.",
        "model_type": "logistic_regression"
    }
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert "probabilities" in data
    assert "Cardiovascular" in data["predicted_category"]
    assert "warning" not in data or data["warning"] is None


def test_classify_endpoint_out_of_distribution_guardrail():
    # Non-medical text should trigger the OOD guardrail warning
    payload = {
        "text": "How to make a delicious vegan chocolate cake with flour, cocoa, vanilla extract, and almond milk frosting.",
        "model_type": "logistic_regression"
    }
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "warning" in data
    assert data["warning"] is not None
    assert "guardrail" in data["warning"].lower() or "density" in data["warning"].lower()


def test_report_analyze_lab_endpoint_normal():
    payload = {
        "report_text": "Hemoglobin: 14.0 g/dL\nFasting Blood Sugar: 88 mg/dL\nCreatinine: 0.9 mg/dL",
        "language": "en"
    }
    response = client.post("/api/report/analyze-lab", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["is_all_normal"] is True
    assert data["doctor_needed"] is False
    assert data["normal_reassurance"] is not None


def test_report_analyze_lab_endpoint_disease_detected():
    payload = {
        "report_text": "Fasting Blood Sugar: 195 mg/dL\nHbA1c: 9.1 %\nPlatelets: 45000 /mcL",
        "language": "en"
    }
    response = client.post("/api/report/analyze-lab", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["is_all_normal"] is False
    assert data["doctor_needed"] is True
    assert data["normal_reassurance"] is None
    assert len(data["directly_identified_conditions"]) > 0


def test_upload_medical_report_magic_byte_validation_rejection():
    # Fake PDF with executable/corrupted header
    files = {"file": ("malicious.pdf", b"MZ\x90\x00\x03\x00\x00\x00this is not a pdf", "application/pdf")}
    response = client.post("/api/report/upload", files=files)
    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()
