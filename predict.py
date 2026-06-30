"""
predict.py
----------
Given one image, prints a single number from 0 to 1:
    0 = real photo
    1 = photo of a screen (recapture / fraud)

Usage:
    python predict.py some_image.jpg

How it works (short version - see NOTE.md for the full writeup):
We extract a small set of hand-engineered features that capture the visual
giveaways of "a photo of a screen" - frequency-domain (moire) energy, color
cast, local sharpness uniformity, glare, and border/bezel edges (see
features.py for details on each). A Logistic Regression model (trained by
train.py on real/ and screen/ photos, weights saved in model.json) turns
those features into a single fraud-probability score.

predict.py itself has NO machine-learning library dependency at runtime -
it just reads model.json (plain numbers) and does a dot product. This keeps
it tiny, fast, and trivial to port to a phone (Swift/Kotlin/JS - it's just
arithmetic).
"""

import json
import os
import sys
import numpy as np

from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")


def _sigmoid(z):
    import math
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _load_model():
    with open(MODEL_PATH, "r") as f:
        return json.load(f)


def predict(image_path_or_img) -> float:
    model = _load_model()
    vec, _ = extract_features(image_path_or_img)

    mean = np.array(model["scaler_mean"])
    scale = np.array(model["scaler_scale"])
    x_scaled = (vec - mean) / np.where(scale != 0, scale, 1.0)

    model_type = model.get("model_type", "logistic_regression")

    if model_type == "logistic_regression":
        coef = np.array(model["coef"])
        intercept = model["intercept"]
        z = np.dot(x_scaled, coef) + intercept
        score = _sigmoid(z)
    elif model_type == "svm_rbf":
        support_vectors = np.array(model["support_vectors"])
        dual_coef = np.array(model["dual_coef"])
        intercept = model["intercept"]
        gamma = model["gamma"]
        prob_a = model["prob_a"]
        prob_b = model["prob_b"]

        # RBF Kernel distance: K(x, xi) = exp(-gamma * ||x - xi||^2)
        diff = support_vectors - x_scaled
        sq_dist = np.sum(diff ** 2, axis=1)
        k = np.exp(-gamma * sq_dist)
        
        # Decision function value
        df = np.dot(dual_coef, k) + intercept
        
        # Platt scaling probability estimation
        score = 1.0 / (1.0 + np.exp(prob_a * df + prob_b))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return float(score)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py some_image.jpg", file=sys.stderr)
        sys.exit(1)
    
    # Try the first argument directly
    target_path = sys.argv[1]
    
    # If the file doesn't exist and there are multiple arguments, they might have
    # been split by the command shell due to spaces (e.g., WhatsApp Image...).
    # Try joining all command line arguments starting from index 1.
    if not os.path.exists(target_path) and len(sys.argv) > 2:
        joined_path = " ".join(sys.argv[1:])
        if os.path.exists(joined_path):
            target_path = joined_path

    if not os.path.exists(target_path):
        print(f"Error: File not found at '{target_path}'", file=sys.stderr)
        if len(sys.argv) > 2 and target_path == sys.argv[1]:
            print("Note: If your file path has spaces, wrap it in double quotes, e.g.:", file=sys.stderr)
            print(f'  python predict.py "C:\\path\\to\\WhatsApp Image.jpeg"', file=sys.stderr)
        sys.exit(1)

    try:
        score = predict(target_path)
        print(round(score, 4))
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)
