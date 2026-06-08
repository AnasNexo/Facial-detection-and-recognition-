"""
face-preprocess.py
------------------
Reads raw face images from each person's folder, detects and aligns
faces using MTCNN, resizes to 160x160 (FaceNet input), normalises
pixel values, and saves the results as .jpg files in a sibling
*-preprocessed folder.

Usage:
    python face-preprocess.py --names daniel alice
    python face-preprocess.py --auto          # auto-discovers all person folders

Directory layout:
    Input:   daniel/0001.jpg ...
    Output:  daniel-preprocessed/0001.jpg ...

Requirements:
    pip install opencv-python mtcnn tensorflow numpy Pillow
"""

import cv2
import os
import argparse
import numpy as np
from mtcnn import MTCNN
from PIL import Image


# FaceNet expects 160×160 RGB images
FACENET_INPUT_SIZE = (160, 160)


def align_face(image_rgb: np.ndarray, keypoints: dict) -> np.ndarray:
    """
    Rotate the face crop so the eyes are horizontally aligned.

    Args:
        image_rgb: RGB face crop (H×W×3)
        keypoints: MTCNN keypoints dict with 'left_eye' and 'right_eye'

    Returns:
        Aligned face as RGB numpy array
    """
    left_eye = np.array(keypoints["left_eye"], dtype=np.float32)
    right_eye = np.array(keypoints["right_eye"], dtype=np.float32)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = ((left_eye[0] + right_eye[0]) / 2,
                  (left_eye[1] + right_eye[1]) / 2)

    h, w = image_rgb.shape[:2]
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    aligned = cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_LINEAR)
    return aligned


def normalize_pixels(face_array: np.ndarray) -> np.ndarray:
    """
    Standardise pixel values to zero mean / unit variance (per-image),
    which is the preprocessing expected by FaceNet/Inception-ResNet-v2.

    Args:
        face_array: float32 array of shape (160, 160, 3)

    Returns:
        Normalised float32 array
    """
    mean = face_array.mean()
    std = face_array.std()
    std = max(std, 1.0 / np.sqrt(face_array.size))  # avoid division by zero
    return (face_array - mean) / std


def preprocess_person(name: str, input_dir: str, output_dir: str, detector: MTCNN):
    """
    Process all images for one person.

    Args:
        name: Person name (e.g. 'daniel')
        input_dir: Path to raw images folder (e.g. './daniel')
        output_dir: Path to save preprocessed images (e.g. './daniel-preprocessed')
        detector: Shared MTCNN instance
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"  [WARN] No images found in '{input_dir}'. Skipping.")
        return

    print(f"\n[INFO] Processing '{name}' — {len(image_files)} images")
    print(f"       Input:  {input_dir}/")
    print(f"       Output: {output_dir}/")

    success, skipped = 0, 0

    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            print(f"  [SKIP] Cannot read: {filename}")
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)

        if not results:
            print(f"  [SKIP] No face detected: {filename}")
            skipped += 1
            continue

        # Pick the most confident detection
        best = max(results, key=lambda x: x["confidence"])
        if best["confidence"] < 0.85:
            print(f"  [SKIP] Low confidence ({best['confidence']:.2f}): {filename}")
            skipped += 1
            continue

        # ── 1. Align ──────────────────────────────────────────────────────
        aligned_rgb = align_face(img_rgb, best["keypoints"])

        # ── 2. Crop ──────────────────────────────────────────────────────
        # Re-detect on aligned image for a clean crop
        re_results = detector.detect_faces(aligned_rgb)
        if re_results:
            best2 = max(re_results, key=lambda x: x["confidence"])
            x, y, w, h = best2["box"]
        else:
            x, y, w, h = best["box"]  # fallback to original detection

        x, y = max(0, x), max(0, y)
        x2 = min(aligned_rgb.shape[1], x + w)
        y2 = min(aligned_rgb.shape[0], y + h)
        face_crop = aligned_rgb[y:y2, x:x2]

        if face_crop.size == 0:
            print(f"  [SKIP] Empty crop: {filename}")
            skipped += 1
            continue

        # ── 3. Resize to 160×160 ─────────────────────────────────────────
        face_resized = cv2.resize(face_crop, FACENET_INPUT_SIZE,
                                   interpolation=cv2.INTER_LANCZOS4)

        # ── 4. Normalise & save ──────────────────────────────────────────
        face_float = face_resized.astype(np.float32)
        face_norm = normalize_pixels(face_float)

        # Rescale back to [0, 255] uint8 for saving as JPEG
        # (embeddings are computed from the normalised float at train/inference time;
        #  we save the visual so the folder is human-inspectable)
        face_uint8 = ((face_norm - face_norm.min()) /
                      (face_norm.max() - face_norm.min() + 1e-6) * 255).astype(np.uint8)

        # Save as RGB → BGR for OpenCV
        cv2.imwrite(output_path, cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR))
        success += 1

    print(f"  ✓ Saved {success} | ✗ Skipped {skipped}")


def discover_person_folders(root: str) -> list[str]:
    """
    Auto-discover raw person folders (exclude *-preprocessed and hidden dirs).
    """
    return sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and not d.endswith("-preprocessed")
        and not d.startswith(".")
    ])


if __name__ == "__main__":
    names = ["donald", "mark", "daniel"]
    root_dir = "."

    if not names:
        print("[ERROR] No person folders found. Run face-dataset.py first.")
        exit(1)

    print(f"[INFO] Initialising MTCNN detector...")
    detector = MTCNN()

    for name in names:
        input_dir = os.path.join(root_dir, name)
        output_dir = os.path.join(root_dir, f"{name}-preprocessed")

        if not os.path.isdir(input_dir):
            print(f"[WARN] Folder not found: '{input_dir}'. Skipping '{name}'.")
            continue

        preprocess_person(name, input_dir, output_dir, detector)

    print("\n[DONE] Preprocessing complete.")
    print("       Next step → python face-train.py")