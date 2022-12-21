import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from ml_project.models.model_fit_predict import load_local_model

from online_inference.model_load import get_model
from online_inference.app import app


model_path = Path(__file__).parent / "assets" / "LogisticRegression_experiment"
model = load_local_model(model_path)


def get_model_mock():
    return model


def death_model_mock():
    return "model"


@pytest.fixture()
def client():
    app.dependency_overrides[get_model] = get_model_mock
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def dead_client():
    app.dependency_overrides[get_model] = death_model_mock
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200


def test_death(dead_client):
    response = dead_client.get("/health")

    assert response.status_code == 503


def test_predict_json(client):
    json_data = {
        "patient": {
            "age": 6,
            "sex": 0,
            "cp": 3,
            "trestbps": 178,
            "chol": 228,
            "fbs": 1,
            "restecg": 0,
            "thalach": 165,
            "exang": 1,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 2,
            "thal": 2,
        }
    }

    response = client.post("/predict/json", json=json_data)
    assert response.status_code == 200
    assert response.json()["pos"] == 0
    assert response.json()["target"] == 1


def test_predict_json_validation_err(client):
    json_data = {
        "patient": {
            "age": 6,
            "sex": 0,
            "cp": 3,
        }
    }

    response = client.post("/predict/json", json=json_data)
    assert response.status_code == 422
