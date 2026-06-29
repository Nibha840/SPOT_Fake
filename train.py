"""
train.py
--------
Reads your photos from real/ and screen/, extracts features.py features for
each, fits a small Logistic Regression classifier, and saves it to
model.json (no pickle, no heavy deps - just plain numbers so predict.py can
load it instantly, even on a phone, with zero ML-runtime dependency).

Usage:
    python train.py
    (expects ./real/*.jpg|png and ./screen/*.jpg|png in the same folder)

Prints train/test accuracy and a small report.
"""

import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from features import extract_features, FEATURE_NAMES

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.heic", "*.HEIC")


def load_folder(folder):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    # Deduplicate paths (important on Windows where glob is case-insensitive)
    unique_paths = list(set(os.path.normcase(os.path.normpath(p)) for p in paths))
    return sorted(unique_paths)


def build_dataset():
    real_paths = load_folder("real")
    screen_paths = load_folder("screen")

    if len(real_paths) == 0 or len(screen_paths) == 0:
        print(f"ERROR: found {len(real_paths)} real/ images and {len(screen_paths)} screen/ images.")
        print("Put your photos in ./real/ and ./screen/ before running train.py")
        sys.exit(1)

    print(f"Found {len(real_paths)} real photos, {len(screen_paths)} screen photos.")

    X, y, paths_used = [], [], []
    for p in real_paths:
        try:
            vec, _ = extract_features(p)
            X.append(vec)
            y.append(0)  # 0 = real
            paths_used.append(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}")

    for p in screen_paths:
        try:
            vec, _ = extract_features(p)
            X.append(vec)
            y.append(1)  # 1 = screen
            paths_used.append(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}")

    return np.array(X), np.array(y), paths_used


def main():
    t0 = time.time()
    X, y, paths_used = build_dataset()
    print(f"Feature extraction for {len(y)} images took {time.time() - t0:.2f}s "
          f"({(time.time() - t0) / len(y) * 1000:.1f} ms/image avg)")

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths_used, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_s))
    test_acc = accuracy_score(y_test, clf.predict(X_test_s))

    print(f"\nTrain accuracy: {train_acc*100:.1f}%")
    print(f"Test accuracy:  {test_acc*100:.1f}%  (on {len(y_test)} held-out images)")
    print("\nConfusion matrix (rows=true, cols=pred) [0=real, 1=screen]:")
    print(confusion_matrix(y_test, clf.predict(X_test_s)))
    print("\nClassification report:")
    print(classification_report(y_test, clf.predict(X_test_s), target_names=["real", "screen"]))

    # Show misclassified files so you can inspect them
    preds_test = clf.predict(X_test_s)
    wrong = [(p, yt, yp) for p, yt, yp in zip(paths_test, y_test, preds_test) if yt != yp]
    if wrong:
        print("Misclassified:")
        for p, yt, yp in wrong:
            print(f"  {p}  true={'real' if yt==0 else 'screen'}  pred={'real' if yp==0 else 'screen'}")

    # Refit on ALL data for the final shipped model (use all available signal)
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf_full.fit(X_full_s, y)

    model = {
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler_full.mean_.tolist(),
        "scaler_scale": scaler_full.scale_.tolist(),
        "coef": clf_full.coef_[0].tolist(),
        "intercept": float(clf_full.intercept_[0]),
        "held_out_test_accuracy": test_acc,
        "n_train_images": int(len(y)),
    }
    with open("model.json", "w") as f:
        json.dump(model, f, indent=2)

    print(f"\nSaved model.json (held-out test accuracy: {test_acc*100:.1f}%)")
    print("Now run: python predict.py some_image.jpg")


if __name__ == "__main__":
    main()
