import json
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "road_sign_final_cand_model_3_best.keras"
CLASSES = ["giveway", "noentry", "pedestrian", "speed", "stop"]
IMG_SIZE = 224
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_PIXELS = 20_000_000


def prepare_compatible_model(model_path: Path) -> Path:
    patched_path = model_path.with_suffix(".patched.keras")
    if patched_path.exists():
        return patched_path

    with zipfile.ZipFile(model_path, "r") as src, zipfile.ZipFile(patched_path, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "config.json":
                config = json.loads(data.decode("utf-8"))

                def remove_value_range(obj):
                    if isinstance(obj, dict):
                        obj.pop("value_range", None)
                        for value in obj.values():
                            remove_value_range(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            remove_value_range(item)

                remove_value_range(config)
                data = json.dumps(config).encode("utf-8")

            dst.writestr(info.filename, data)

    return patched_path


class RoadSignPredictor:
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        compatible_model_path = prepare_compatible_model(self.model_path)
        self.model = tf.keras.models.load_model(compatible_model_path, compile=False)

    def validate_image(self, image: Image.Image) -> Image.Image:
        if image is None:
            raise ValueError("Please upload an image of a road sign.")
        if not isinstance(image, Image.Image):
            raise TypeError("Uploaded content must be an image.")

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("The uploaded image is invalid.")
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValueError("The uploaded image is too large.")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("The uploaded image is too large to process safely.")

        try:
            image.load()
        except Exception as exc:
            raise ValueError("The uploaded image could not be read.") from exc

        return image

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        img = self.validate_image(image)
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, image: Image.Image):
        try:
            image_array = self.preprocess_image(image)
            probabilities = self.model.predict(image_array, verbose=0)[0]
        except Exception:
            return None, "The uploaded file could not be processed safely. Please upload a valid image."

        probs = {label: round(float(prob), 4) for label, prob in zip(CLASSES, probabilities)}
        predicted_class = max(probs, key=probs.get)
        confidence = probs[predicted_class] * 100

        result_text = f"Predicted class: {predicted_class} ({confidence:.1f}%)"
        return probs, result_text
