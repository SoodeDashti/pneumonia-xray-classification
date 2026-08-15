"""
Chest X-Ray Pneumonia Classifier — Gradio dashboard.

Loads the trained Model 3 (ResNet50 transfer learning) and serves an
upload-an-image -> get-a-prediction interface, with Monte Carlo Dropout
uncertainty estimation to flag low-confidence predictions for manual
review.

Educational / portfolio project only — not a diagnostic tool.
"""

import numpy as np
import gradio as gr
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- Configuration ---
MODEL_PATH = "models/model_3.keras"   # adjust if your folder structure differs
IMG_SIZE = (150, 150)
MC_DROPOUT_ITERATIONS = 30
UNCERTAINTY_THRESHOLD = 0.15  # std above this -> flagged as uncertain
PORT = 8860  # distinct from other Docker projects on this machine

# --- Load model once at startup ---
model = load_model(MODEL_PATH)


def predict_with_uncertainty(img_array, n_iter=MC_DROPOUT_ITERATIONS):
    """
    Run n_iter stochastic forward passes with dropout active
    (training=True) to get a distribution of predictions instead of a
    single point estimate. Returns (mean_prediction, std_prediction).
    """
    preds = np.array([
        model(img_array, training=True).numpy().flatten()[0]
        for _ in range(n_iter)
    ])
    return preds.mean(), preds.std()


def predict_pneumonia(image):
    if image is None:
        return None, ""

    # Preprocess to match what the ResNet50-based model expects
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    mean_pred, std_pred = predict_with_uncertainty(img_array)

    result = {
        "NORMAL": float(1 - mean_pred),
        "PNEUMONIA": float(mean_pred),
    }

    if std_pred > UNCERTAINTY_THRESHOLD:
        note = f"\u26a0\ufe0f High uncertainty \u2014 recommend radiologist review\nUncertainty (std): {std_pred:.4f}"
    else:
        note = f"Model is confident\nUncertainty (std): {std_pred:.4f}"

    return result, note


demo = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Textbox(label="Uncertainty Note"),
    ],
    title="Chest X-Ray Pneumonia Classifier",
    description=(
        "Upload a chest X-ray image to get a PNEUMONIA/NORMAL prediction "
        "from a ResNet50-based model (91.5% test accuracy, 97.4% sensitivity). "
        "\u26a0\ufe0f For educational/portfolio purposes only \u2014 not a diagnostic tool. "
        "Trained on pediatric (ages 1-5) X-rays from a single medical center."
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)