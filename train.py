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

    # 1. Logistic Regression
    clf_lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf_lr.fit(X_train_s, y_train)
    lr_train_acc = accuracy_score(y_train, clf_lr.predict(X_train_s))
    lr_test_acc = accuracy_score(y_test, clf_lr.predict(X_test_s))

    # 2. SVM with RBF Kernel
    from sklearn.svm import SVC
    clf_svm = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42)
    clf_svm.fit(X_train_s, y_train)
    svm_train_acc = accuracy_score(y_train, clf_svm.predict(X_train_s))
    svm_test_acc = accuracy_score(y_test, clf_svm.predict(X_test_s))

    # 3. Random Forest
    from sklearn.ensemble import RandomForestClassifier
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
    clf_rf.fit(X_train_s, y_train)
    rf_train_acc = accuracy_score(y_train, clf_rf.predict(X_train_s))
    rf_test_acc = accuracy_score(y_test, clf_rf.predict(X_test_s))

    print("\n--- Model Performance Comparison ---")
    print(f"Logistic Regression: Train={lr_train_acc*100:.1f}%, Test={lr_test_acc*100:.1f}%")
    print(f"SVM (RBF Kernel):     Train={svm_train_acc*100:.1f}%, Test={svm_test_acc*100:.1f}%")
    print(f"Random Forest:        Train={rf_train_acc*100:.1f}%, Test={rf_test_acc*100:.1f}%")
    print("------------------------------------\n")

    # Select the best model (prefer Logistic Regression if accuracy is tied, for simplicity)
    best_model_name = "logistic_regression"
    best_test_acc = lr_test_acc
    best_clf = clf_lr
    
    if svm_test_acc > best_test_acc:
        best_model_name = "svm_rbf"
        best_test_acc = svm_test_acc
        best_clf = clf_svm
    
    print(f"Selected Best Model: {best_model_name.upper()} (Test Accuracy: {best_test_acc*100:.1f}%)")

    print("\nConfusion matrix (rows=true, cols=pred) [0=real, 1=screen]:")
    print(confusion_matrix(y_test, best_clf.predict(X_test_s)))
    print("\nClassification report:")
    print(classification_report(y_test, best_clf.predict(X_test_s), target_names=["real", "screen"]))

    # Show misclassified files so you can inspect them
    preds_test = best_clf.predict(X_test_s)
    wrong = [(p, yt, yp) for p, yt, yp in zip(paths_test, y_test, preds_test) if yt != yp]
    if wrong:
        print("Misclassified:")
        for p, yt, yp in wrong:
            print(f"  {p}  true={'real' if yt==0 else 'screen'}  pred={'real' if yp==0 else 'screen'}")

    # Refit on ALL data for the final shipped model (use all available signal)
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)

    model = {
        "model_type": best_model_name,
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler_full.mean_.tolist(),
        "scaler_scale": scaler_full.scale_.tolist(),
        "held_out_test_accuracy": best_test_acc,
        "n_train_images": int(len(y)),
    }

    if best_model_name == "logistic_regression":
        clf_full = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf_full.fit(X_full_s, y)
        model["coef"] = clf_full.coef_[0].tolist()
        model["intercept"] = float(clf_full.intercept_[0])
    elif best_model_name == "svm_rbf":
        clf_full = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42)
        clf_full.fit(X_full_s, y)
        model["support_vectors"] = clf_full.support_vectors_.tolist()
        model["dual_coef"] = clf_full.dual_coef_[0].tolist()
        model["intercept"] = float(clf_full.intercept_[0])
        model["gamma"] = float(clf_full._gamma) if hasattr(clf_full, "_gamma") else (1.0 / X.shape[1])
        model["prob_a"] = float(clf_full.probA_[0])
        model["prob_b"] = float(clf_full.probB_[0])

    with open("model.json", "w") as f:
        json.dump(model, f, indent=2)

    print(f"\nSaved model.json (held-out test accuracy: {best_test_acc*100:.1f}%)")
    print("Now run: python predict.py some_image.jpg")


if __name__ == "__main__":
    main()
