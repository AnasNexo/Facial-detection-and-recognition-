"""Prepare aligned grayscale faces for PCA, preserving the existing train/test split.

Run: python analyse/preprocess_pca.py
PNG files are inspection images; NPY files and train.npz/test.npz contain float32
pixels in [0, 1]. PCA and any learned standardization belong in the training pipeline.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


BASE = Path(__file__).resolve().parent
EYE_COLUMNS = ["left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y"]


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def pixel_digest(rgb):
    return hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()


def contained_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path outside dataset: {relative}")
    return path


def eye_transform(eyes, size):
    """Map source eyes to (30%, 35%) and (70%, 35%) in one similarity warp."""
    eyes = np.asarray(eyes, dtype=np.float64).reshape(2, 2)
    eyes = eyes[np.argsort(eyes[:, 0])]
    dx, dy = eyes[1] - eyes[0]
    distance = float(np.hypot(dx, dy))
    if not np.isfinite(eyes).all() or distance < 2 or dx <= 0:
        raise ValueError("Invalid eye landmarks")
    angle = float(np.degrees(np.arctan2(dy, dx)))
    span = size - 1
    target_eyes = np.array([[.30 * span, .35 * span], [.70 * span, .35 * span]])
    matrix = cv2.getRotationMatrix2D(tuple(eyes.mean(axis=0)), angle,
                                    .40 * span / distance)
    matrix[:, 2] += target_eyes.mean(axis=0) - eyes.mean(axis=0)
    mapped = cv2.transform(eyes[None, :, :], matrix)[0]
    if not np.allclose(mapped, target_eyes, atol=1e-5):
        raise ValueError("Eye alignment geometry failed")
    return matrix, angle


def prepare(split_dir, source_dir, landmarks_path, output_dir, size=80):
    split_dir, source_dir, output_dir = [p.resolve() for p in (split_dir, source_dir, output_dir)]
    if size < 16:
        raise ValueError("Image size must be at least 16")
    if (output_dir == split_dir or output_dir == source_dir
            or output_dir.is_relative_to(split_dir) or output_dir.is_relative_to(source_dir)):
        raise ValueError("Output must be separate from the input datasets")
    parts = []
    for split in ("train", "test"):
        table = pd.read_csv(split_dir / f"{split}.csv")
        if not table["split"].eq(split).all():
            raise ValueError(f"Incorrect split labels in {split}.csv")
        parts.append(table)
    manifest = pd.concat(parts, ignore_index=True)
    if not manifest.source.is_unique or not manifest.crop.is_unique:
        raise ValueError("Repeated sources/crops in input split")
    if set(parts[0]["class"]) != set(parts[1]["class"]):
        raise ValueError("Train and test must contain the same identities")
    detections = pd.read_csv(landmarks_path)
    required = ["path", "detection_id", "clipped_x", "clipped_y",
                "clipped_width", "clipped_height", *EYE_COLUMNS]
    # Match the selected detection, never substitute the most confident face.
    matched = manifest.merge(detections[required], left_on=["source", "detection_id"],
                             right_on=["path", "detection_id"], how="left",
                             validate="one_to_one", indicator=True)
    if not matched["_merge"].eq("both").all():
        raise ValueError("Some selected faces have no saved landmarks")
    expected_boxes = matched[["x1", "y1", "x2", "y2"]].to_numpy()
    cached_boxes = np.column_stack([
        np.floor(matched.clipped_x), np.floor(matched.clipped_y),
        np.ceil(matched.clipped_x + matched.clipped_width),
        np.ceil(matched.clipped_y + matched.clipped_height)])
    if not np.array_equal(expected_boxes, cached_boxes):
        raise ValueError("Saved landmarks do not match selected crop boxes")

    # Validate every source and transform before creating the output dataset.
    transforms = []
    for _, row in matched.iterrows():
        rgb = read_rgb(contained_path(source_dir, row.source))
        if pixel_digest(rgb) != row.source_pixel_sha256:
            raise ValueError(f"Source changed since split: {row.source}")
        eyes = row[EYE_COLUMNS].to_numpy(dtype=float).reshape(2, 2)
        if not (np.isfinite(eyes).all() and (eyes[:, 0] >= 0).all()
                and (eyes[:, 0] < rgb.shape[1]).all() and (eyes[:, 1] >= 0).all()
                and (eyes[:, 1] < rgb.shape[0]).all()):
            raise ValueError(f"Eyes outside source image: {row.source}")
        transforms.append(eye_transform(eyes, size))

    output_dir.mkdir(parents=True, exist_ok=True)
    records, arrays = [], {"train": [], "test": []}
    labels, sources = {"train": [], "test": []}, {"train": [], "test": []}
    for index, ((_, row), (matrix, angle)) in enumerate(zip(matched.iterrows(), transforms), 1):
        rgb = read_rgb(contained_path(source_dir, row.source))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        aligned = cv2.warpAffine(gray, matrix, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        normalized = aligned.astype(np.float32) / np.float32(255)
        png_rel = (Path(row["split"]) / row.crop).as_posix()
        npy_rel = str(Path(png_rel).with_suffix(".npy")).replace("\\", "/")
        png_path = contained_path(output_dir, png_rel)
        npy_path = contained_path(output_dir, npy_rel)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(aligned).save(png_path)
        np.save(npy_path, normalized, allow_pickle=False)
        split = row["split"]
        arrays[split].append(normalized.reshape(-1))
        labels[split].append(row["class"])
        sources[split].append(row.source)
        records.append({**row[manifest.columns].to_dict(), "image_path": png_rel,
                        "normalized_path": npy_rel, "height": size, "width": size,
                        "eye_angle_degrees": angle, **row[EYE_COLUMNS].to_dict(),
                        "affine_matrix": json.dumps(matrix.tolist())})
        if index % 100 == 0 or index == len(matched):
            print(f"Prepared {index}/{len(matched)} faces", flush=True)

    result = pd.DataFrame(records)
    result.to_csv(output_dir / "manifest.csv", index=False)
    for split in ("train", "test"):
        result.loc[result["split"].eq(split)].to_csv(output_dir / f"{split}.csv", index=False)
        np.savez_compressed(output_dir / f"{split}.npz",
                            X=np.stack(arrays[split]), y=np.asarray(labels[split], dtype=str),
                            sources=np.asarray(sources[split], dtype=str))
    counts = pd.crosstab(result["class"], result["split"])[["train", "test"]]
    counts.to_csv(output_dir / "class_counts.csv")
    settings = {"size": [size, size], "color": "grayscale", "dtype": "float32",
                "normalization": "uint8 / 255, range [0, 1]",
                "alignment": "similarity transform from selected MTCNN eye landmarks",
                "target_eyes_relative": [[.30, .35], [.70, .35]],
                "interpolation": "INTER_LINEAR", "border": "BORDER_REFLECT_101",
                "source_dir": str(source_dir), "split_dir": str(split_dir),
                "landmarks": str(landmarks_path.resolve()), "clahe": False, "gamma": False,
                "train": len(parts[0]), "test": len(parts[1])}
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# PCA input dataset\n\n"
        "Train/test membership is unchanged. PNG files are grayscale inspection images.\n"
        "NPY files contain normalized float32 images in [0,1].\n"
        "NPZ archives contain flattened X, string labels y, and source paths.\n\n"
        "```python\nimport numpy as np\n"
        "train = np.load('train.npz', allow_pickle=False)\n"
        "X_train, y_train = train['X'], train['y']\n"
        "test = np.load('test.npz', allow_pickle=False)\n"
        "X_test, y_test = test['X'], test['y']\n```\n\n"
        "Fit PCA/scalers only within training folds. No PCA has been fitted here.\n"
        "Alignment starts from the original selected face source, avoiding a second resize.\n"
        "Reflected pixels may appear at source boundaries; inspect alignment_preview.png.\n",
        encoding="utf-8")
    save_preview(result, split_dir, output_dir)
    print(counts.to_string())
    print(f"Saved dataset: {output_dir}")


def save_preview(manifest, split_dir, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = manifest.groupby("class", sort=True).head(1)
    fig, axes = plt.subplots(len(examples), 2, figsize=(7, 2.3 * len(examples)), squeeze=False)
    for pair, (_, row) in zip(axes, examples.iterrows()):
        with Image.open(split_dir / row.split_path) as image:
            pair[0].imshow(image)
        with Image.open(output_dir / row.image_path) as image:
            pair[1].imshow(image, cmap="gray", vmin=0, vmax=255)
        pair[0].set_title(row["class"] + " / original crop", fontsize=9)
        pair[1].set_title(f"Aligned grayscale / {row.width} x {row.height}", fontsize=9)
        for ax in pair:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "alignment_preview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    candidates = [BASE / "preprocessed_data/mtcnn_results/detections.csv",
                  BASE / "mtcnn_results/detections.csv"]
    default_landmarks = next((p for p in candidates if p.is_file()), candidates[0])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=BASE / "cropped_faces_split")
    parser.add_argument("--source-dir", type=Path, default=BASE / "preprocessed_data")
    parser.add_argument("--landmarks", type=Path, default=default_landmarks)
    parser.add_argument("--output-dir", type=Path, default=BASE / "preprocessed_crop_pca_split")
    parser.add_argument("--size", type=int, default=80)
    args = parser.parse_args()
    prepare(args.split_dir, args.source_dir, args.landmarks, args.output_dir, args.size)


if __name__ == "__main__":
    main()
