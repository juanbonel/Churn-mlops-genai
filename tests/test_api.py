from fastapi.testclient import TestClient

from churn_project.api import app

client = TestClient(app)


def test_health_endpoint_is_available():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_features_endpoint_lists_contract():
    response = client.get("/features")
    assert response.status_code == 200
    assert "monthly_charges" in response.json()["required_features"]


def test_predict_validates_input():
    response = client.post("/predict", json={"tenure_months": 0})
    assert response.status_code == 422
