import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.models.evaluate import compute_metrics


@pytest.fixture
def fitted_dummy():
    rng = np.random.default_rng(42)
    X = rng.random((200, 8))
    y = rng.integers(0, 2, 200)
    model = DummyClassifier(strategy="stratified", random_state=42)
    model.fit(X, y)
    return model, X, y


def test_compute_metrics_keys(fitted_dummy):
    model, X, y = fitted_dummy
    metrics = compute_metrics(model, X, y)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1_score", "roc_auc"}


def test_compute_metrics_bounds(fitted_dummy):
    model, X, y = fitted_dummy
    for key, val in compute_metrics(model, X, y).items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"
