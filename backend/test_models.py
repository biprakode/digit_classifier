"""Tests for model registry: loading, prediction shape, and output range."""
import pytest
import numpy as np
from models import models as model_classes


def test_registry_not_empty():
    assert len(model_classes) > 0, "Model registry is empty"


@pytest.mark.parametrize("model_id", list(model_classes.keys()))
def test_model_instantiates(model_id):
    model = model_classes[model_id]()
    assert hasattr(model, "predict"), f"{model_id} missing predict() method"


# Use a lightweight model for prediction tests to keep CI fast
LIGHT_MODELS = [
    "custom_logreg",
    "sklearn_logreg",
    "custom_nb",
    "sklearn_nb",
    "sklearn_DT",
    "custom_DT",
]

# Filter to only models that exist in registry
LIGHT_MODELS = [m for m in LIGHT_MODELS if m in model_classes]


@pytest.mark.parametrize("model_id", LIGHT_MODELS)
def test_flat_model_prediction_shape(model_id):
    """Non-CNN models receive (1, 784) and return shape (1,) or (n,)."""
    model = model_classes[model_id]()
    dummy = np.random.rand(1, 784).astype(np.float32)
    preds = model.predict(dummy)
    assert preds.shape == (1,), f"{model_id} returned shape {preds.shape}, expected (1,)"


@pytest.mark.parametrize("model_id", LIGHT_MODELS)
def test_prediction_is_digit(model_id):
    """Predictions should be integers in 0-9."""
    model = model_classes[model_id]()
    dummy = np.random.rand(1, 784).astype(np.float32)
    preds = model.predict(dummy)
    assert 0 <= int(preds[0]) <= 9, f"{model_id} predicted {preds[0]}, expected 0-9"