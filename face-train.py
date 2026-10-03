"""
face-train.py
-------------
1. Loads FaceNet (Inception-ResNet-v1) pretrained on VGGFace2 via the
   `deepface` library (zero training cost — pure transfer learning).
2. Iterates over every *-preprocessed folder and extracts a 128-d
   embedding for each face image.
3. Trains a calibrated SVM classifier on those embeddings.
4. Saves the SVM + label encoder as 'face_model.pkl' next to this script.

Usage:
    python face-train.py
    python face-train.py --root_dir . --model_out face_model.pkl

Requirements:
    pip install deepface tensorflow scikit-learn numpy opencv-python Pillow joblib
    # deepface auto-downloads FaceNet weights on first run (~90 MB)
"""

import os
import argparse
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
import cv2

# We use DeepFace as a clean wrapper around FaceNet/Inception-ResNet-v1
from deepface import DeepFace


# ─── Configuration ────────────────────────────────────────────────────────────

FACENET_MODEL  = "Facenet"       # DeepFace model name  (Facenet = 128-d)
# Use "Facenet512" for 512-d embeddings — more accurate, slightly slower
FACENET_INPUT  = (160, 160)
MIN_IMAGES     = 5               # warn if a class has fewer than this
UNKNOWN_THRESH = 0.55            # SVM probability below this → "Unknown"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load image as RGB float32 numpy array normalised to [0,1]."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def extract_embedding(image_rgb_float: np.ndarray) -> np.ndarray:
    """
    Extract a FaceNet 128-d embedding from a preprocessed face image.
    DeepFace.represent() handles internal resizing and normalisation.

    Args:
        image_rgb_float: float32 RGB image in [0, 1] (any size ≥ 160×160)

    Returns:
        numpy array of shape (128,)
    """
    # DeepFace expects uint8 [0,255]
    img_uint8 = (image_rgb_float * 255).astype(np.uint8)

    # enforce_detection=False because we already have clean face crops
    result = DeepFace.represent(
        img_path=img_uint8,
        model_name=FACENET_MODEL,
        enforce_detection=False,
        detector_backend="skip",   # skip detection — image is already a face crop
    )
    return np.array(result[0]["embedding"], dtype=np.float32)


def discover_preprocessed_folders(root: str) -> list[tuple[str, str]]:
    """
    Find all *-preprocessed folders and return (person_name, folder_path) pairs.
    """
    pairs = []
    for entry in sorted(os.listdir(root)):
        if entry.endswith("-preprocessed") and os.path.isdir(os.path.join(root, entry)):
            person_name = entry.replace("-preprocessed", "")
            pairs.append((person_name, os.path.join(root, entry)))
    return pairs


# ─── Main ─────────────────────────────────────────────────────────────────────

def train(root_dir: str, model_out: str):
    print("[INFO] Discovering preprocessed face folders...")
    person_folders = discover_preprocessed_folders(root_dir)

    if len(person_folders) < 2:
        raise RuntimeError(
            "Need at least 2 people to train a classifier. "
            "Run face-dataset.py and face-preprocess.py for each person first."
        )

    print(f"[INFO] Found {len(person_folders)} people: {[n for n,_ in person_folders]}\n")

    embeddings, labels = [], []

    for person_name, folder_path in person_folders:
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(image_files) < MIN_IMAGES:
            print(f"  [WARN] '{person_name}' only has {len(image_files)} images "
                  f"(minimum recommended: {MIN_IMAGES})")

        print(f"  Extracting embeddings for '{person_name}' — {len(image_files)} images")
        person_embeddings = []

        for fname in image_files:
            img_path = os.path.join(folder_path, fname)
            try:
                img = load_image(img_path)
                emb = extract_embedding(img)
                person_embeddings.append(emb)
            except Exception as e:
                print(f"    [SKIP] {fname}: {e}")

        embeddings.extend(person_embeddings)
        labels.extend([person_name] * len(person_embeddings))
        print(f"    → {len(person_embeddings)} embeddings extracted")

    X = np.array(embeddings, dtype=np.float32)
    y = np.array(labels)

    print(f"\n[INFO] Total embeddings: {len(X)} | Classes: {np.unique(y).tolist()}")

    # ── Label encoding ───────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Model: L2-normalise → calibrated SVM (RBF kernel) ───────────────────
    #
    # Pipeline:
    #   1. Normalizer(l2)  → project embeddings onto unit hypersphere
    #                         (standard for cosine-distance-trained networks)
    #   2. SVC(RBF, C=10)  → non-linear classifier with probability estimates
    #                         (via Platt scaling: probability=True)
    #
    # Why SVM over a neural head?
    #   • Excellent generalisation with small datasets (< few hundred per class)
    #   • Probability calibration gives reliable confidence scores
    #   • Fast inference (microseconds per prediction)
    #
    clf = Pipeline([
        ("normalizer", Normalizer(norm="l2")),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,   # enables predict_proba for confidence scores
            class_weight="balanced",
        ))
    ])

    # ── Cross-validation ─────────────────────────────────────────────────────
    n_splits = min(5, len(np.unique(y_enc)))  # can't have more folds than classes
    if len(X) >= n_splits * 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")
        print(f"\n[CV]  {n_splits}-fold accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    else:
        print("[WARN] Not enough samples for cross-validation; skipping.")

    # ── Final fit on all data ─────────────────────────────────────────────────
    clf.fit(X, y_enc)
    print("[INFO] Final model trained on full dataset.")

    # ── Save model bundle ─────────────────────────────────────────────────────
    model_bundle = {
        "classifier": clf,
        "label_encoder": le,
        "unknown_threshold": UNKNOWN_THRESH,
        "facenet_model": FACENET_MODEL,
        "people": le.classes_.tolist(),
        "embedding_dim": X.shape[1],
    }
    joblib.dump(model_bundle, model_out)
    print(f"\n[SAVED] Model → '{model_out}'")
    print(f"        People recognised: {le.classes_.tolist()}")
    print(f"        Embedding dim:     {X.shape[1]}")
    print(f"        Unknown threshold: {UNKNOWN_THRESH}")
    print("\n[DONE] Next step → python face-recognition.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FaceNet + SVM face recognition model."
    )
    parser.add_argument(
        "--root_dir", type=str, default=".",
        help="Root directory containing *-preprocessed folders (default: '.')"
    )
    parser.add_argument(
        "--model_out", type=str, default="face_model.pkl",
        help="Output path for the saved model (default: face_model.pkl)"
    )
    args = parser.parse_args()

    train(root_dir=args.root_dir, model_out=args.model_out)
