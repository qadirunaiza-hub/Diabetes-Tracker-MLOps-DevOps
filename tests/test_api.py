import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    with patch("src.api.app._load_model"), \
         patch("src.api.app._model", mock_model), \
         patch("src.api.app._scaler", None):
        from src.api.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert json.loads(resp.data)["status"] == "healthy"


def test_features_endpoint(client):
    resp = client.get("/features")
    data = json.loads(resp.data)
    assert "feature_columns" in data
    assert len(data["feature_columns"]) > 0


def test_predict_single_missing_field(client):
    resp = client.post(
        "/predict/single",
        json={"Pregnancies": 3, "Glucose": 100},
        content_type="application/json",
    )
    assert resp.status_code == 500
