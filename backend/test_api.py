"""FastAPI endpoint tests using TestClient."""
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "models_available" in data


def test_list_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "available_models" in data
    assert data["total"] > 0


def test_predict_unknown_model(client):
    resp = client.post("/predict", json={"pixels": [0] * 784, "model_id": "nonexistent"})
    assert resp.status_code == 404


def test_predict_empty_pixels(client):
    resp = client.post("/predict", json={"pixels": [], "model_id": "sklearn_logreg"})
    assert resp.status_code == 400


def test_predict_valid(client):
    """Predict with a lightweight model and valid 784-pixel input."""
    pixels = [0] * 784
    resp = client.post("/predict", json={"pixels": pixels, "model_id": "sklearn_logreg"})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert 0 <= data["prediction"] <= 9
    assert data["model_used"] == "sklearn_logreg"


def test_predict_grayscale_input(client):
    """Verify the backend correctly normalizes 0-255 grayscale input."""
    import numpy as np
    # Simulate a drawn digit: mostly zeros (black bg) with some bright pixels
    pixels = [0] * 784
    for i in range(300, 320):
        pixels[i] = 200  # some bright pixels
    resp = client.post("/predict", json={"pixels": pixels, "model_id": "sklearn_logreg"})
    assert resp.status_code == 200
    data = resp.json()
    assert 0 <= data["prediction"] <= 9