from pathlib import Path
import sys

import gradio as gr

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from predictor import RoadSignPredictor


predictor = RoadSignPredictor()


def predict(image):
    try:
        return predictor.predict(image)
    except Exception:
        return None, "The uploaded file could not be processed safely. Please upload a valid image."


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", sources=["upload"], label="Upload a road sign image"),
    outputs=[
        gr.Label(label="Class probabilities"),
        gr.Textbox(label="Prediction result"),
    ],
    title="Road Sign Classifier",
    description="Upload an image of a road sign to classify it into one of the supported classes.",
)


if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
