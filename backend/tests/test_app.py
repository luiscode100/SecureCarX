import os

import pytest

from config import MODEL_PATH

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="modelo local no disponible (se genera ejecutando train.py)",
)

VALID_SAMPLE = {
    "OBJECT_ID": 1, "INSR_BEGIN": "2024-01-01", "INSR_END": "2025-01-01",
    "INSR_TYPE": "1202", "PROD_YEAR": 2020, "PREMIUM": 1200.0,
    "INSURED_VALUE": 35000.0, "SEX": "1", "MAKE": "TOYOTA", "USAGE": "Private",
    "TYPE_VEHICLE": "Car", "SEATS_NUM": 5, "CARRYING_CAPACITY": 0.0,
    "CCM_TON": 1600.0, "EFFECTIVE_YR": 2024,
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    import feedback
    import prediction_log
    monkeypatch.setattr(feedback, "FEEDBACK_LOG_PATH", str(tmp_path / "feedback_logs.csv"))
    monkeypatch.setattr(prediction_log, "PREDICTION_LOG_PATH", str(tmp_path / "prediction_logs.csv"))

    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_prediction(client):
    response = client.post("/predict", json=VALID_SAMPLE)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_feedback_endpoint_records_label(client):
    response = client.post("/feedback", json={**VALID_SAMPLE, "CLAIM_PAID": 1500.0})
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"


def test_reload_endpoint(client):
    response = client.post("/reload")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
