from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.predictor import RoadSignPredictor


@pytest.fixture
def predictor():
    return RoadSignPredictor(model_path=Path("models/best_model.keras"))


def test_validate_image_rejects_none(predictor):
    with pytest.raises(ValueError):
        predictor.validate_image(None)


def test_validate_image_rejects_invalid_type(predictor):
    with pytest.raises(TypeError):
        predictor.validate_image("not-an-image")


def test_validate_image_accepts_valid_image(predictor):
    img = Image.new("RGB", (64, 64), color="red")
    validated = predictor.validate_image(img)
    assert isinstance(validated, Image.Image)


def test_preprocess_image_returns_expected_shape(predictor):
    img = Image.new("RGB", (64, 64), color="blue")
    arr = predictor.preprocess_image(img)
    assert arr.shape == (1, 224, 224, 3)


def test_predict_returns_safe_message_on_model_error(predictor, monkeypatch):
    def fail_preprocess(_image):
        raise Exception("boom")

    monkeypatch.setattr(predictor, "preprocess_image", fail_preprocess)
    result = predictor.predict(Image.new("RGB", (64, 64), color="green"))

    assert result == (None, "The uploaded file could not be processed safely. Please upload a valid image.")


def test_predict_returns_probabilities_and_label(predictor, monkeypatch):
    original_preprocess = predictor.preprocess_image

    def fake_preprocess(_image):
        return original_preprocess(Image.new("RGB", (64, 64), color="green"))

    monkeypatch.setattr(predictor, "preprocess_image", fake_preprocess)
    predictor.model = MagicMock()
    predictor.model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.2, 0.2]])

    probs, text = predictor.predict(Image.new("RGB", (64, 64), color="green"))

    assert "pedestrian" in probs
    assert "Predicted class:" in text
