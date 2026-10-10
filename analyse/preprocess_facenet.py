"""Align selected faces for FaceNet and save RGB uint8 PNGs at 160 x 160.

Run: python analyse/preprocess_facenet.py
The existing train/test membership is preserved. Pixel normalization and
embedding extraction are left to the subsequent DeepFace.represent call.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Share the source-reading and alignment geometry with the PCA preparation.
if __package__:
    from .preprocess_pca import EYE_COLUMNS, contained_path, eye_transform, pixel_digest, read_rgb
else:
    from preprocess_pca import EYE_COLUMNS, contained_path, eye_transform, pixel_digest, read_rgb


BASE = Path(__file__).resolve().parent
SIZE = 160


def prepare(split_dir, source_dir, landmarks_path, output_dir):
    split_dir, source_dir, output_dir = [p.resolve() for p in (split_dir, source_dir, output_dir)]
    if any(output_dir == p or output_dir.is_relative_to(p) for p in (split_dir, source_dir)):
        raise ValueError("Output must be separate from the input datasets")

    tables = []
    for split in ("train", "test"):
        table = pd.read_csv(split_dir / f"{split}.csv")
        if table.empty or not table["split"].eq(split).all():
            raise ValueError(f"Invalid membership in {split}.csv")
        tables.append(table)
    manifest = pd.concat(tables, ignore_index=True)
    if not manifest.source.is_unique or not manifest.crop.is_unique:
        raise ValueError("Repeated sources or crops in train/test manifests")
    if set(tables[0]["class"]) != set(tables[1]["class"]):
        raise ValueError("Train and test must contain the same identities")

    columns = ["path", "detection_id", "clipped_x", "clipped_y",
               "clipped_width", "clipped_height", *EYE_COLUMNS]
    detections = pd.read_csv(landmarks_path)
    matched = manifest.merge(detections[columns], left_on=["source", "detection_id"],
                             right_on=["path", "detection_id"], how="left",
                             validate="one_to_one", indicator=True)
    if not matched["_merge"].eq("both").all():
        raise ValueError("Missing landmarks for a selected face")
    cached_boxes = np.column_stack([
        np.floor(matched.clipped_x), np.floor(matched.clipped_y),
        np.ceil(matched.clipped_x + matched.clipped_width),
        np.ceil(matched.clipped_y + matched.clipped_height)])
    if not np.array_equal(cached_boxes, matched[["x1", "y1", "x2", "y2"]].to_numpy()):
        raise ValueError("Landmarks do not match the selected face boxes")

    # Preflight all sources and landmarks before writing any images.
    transforms = []
    for _, row in matched.iterrows():
        rgb = read_rgb(contained_path(source_dir, row.source))
        if pixel_digest(rgb) != row.source_pixel_sha256:
            raise ValueError(f"Source changed since split: {row.source}")
        eyes = row[EYE_COLUMNS].to_numpy(dtype=float).reshape(2, 2)
        if not (np.isfinite(eyes).all() and (eyes[:, 0] >= 0).all()
                and (eyes[:, 0] < rgb.shape[1]).all() and (eyes[:, 1] >= 0).all()
                and (eyes[:, 1] < rgb.shape[0]).all()):
            raise ValueError(f"Invalid eyes in source: {row.source}")
        transforms.append(eye_transform(eyes, SIZE))

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_path = output_dir / "manifest.csv"
    previous = pd.read_csv(previous_path) if previous_path.exists() else pd.DataFrame()
    records = []
    for index, ((_, row), (matrix, angle)) in enumerate(zip(matched.iterrows(), transforms), 1):
        rgb = read_rgb(contained_path(source_dir, row.source))
        aligned = cv2.warpAffine(rgb, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        if aligned.shape != (SIZE, SIZE, 3) or aligned.dtype != np.uint8:
            raise ValueError(f"Unexpected output format: {row.source}")
        image_rel = (Path(row["split"]) / row.crop).as_posix()
        dst = contained_path(output_dir, image_rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # PIL preserves RGB channel order and the uint8 pixel range [0, 255].
        Image.fromarray(aligned).save(dst, format="PNG")
        records.append({**row[manifest.columns].to_dict(),
                        "input_split_path": row.split_path, "split_path": image_rel,
                        "image_path": image_rel, "width": SIZE, "height": SIZE,
                        "channels": 3, "color": "RGB", "dtype": "uint8",
                        "eye_angle_degrees": angle, **row[EYE_COLUMNS].to_dict(),
                        "affine_matrix": json.dumps(matrix.tolist()),
                        "aligned_pixel_sha256": pixel_digest(aligned)})
        if index % 100 == 0 or index == len(matched):
            print(f"Prepared {index}/{len(matched)} color faces", flush=True)

    result = pd.DataFrame(records)
    # Remove only outdated PNG copies previously generated by this script.
    if not previous.empty:
        for rel in set(previous.image_path) - set(result.image_path):
            dst = contained_path(output_dir, rel)
            if Path(rel).parts[0] not in {"train", "test"} or dst.suffix.lower() != ".png":
                raise ValueError(f"Unexpected previous output path: {rel}")
            if dst.is_file():
                dst.unlink()
    result.to_csv(previous_path, index=False)
    for split in ("train", "test"):
        result.loc[result["split"].eq(split)].to_csv(output_dir / f"{split}.csv", index=False)
    counts = pd.crosstab(result["class"], result["split"])[["train", "test"]]
    counts.to_csv(output_dir / "class_counts.csv")
    settings = {"shape": [SIZE, SIZE, 3], "color": "RGB", "dtype": "uint8",
                "pixel_range": [0, 255], "pixel_normalization_applied": False,
                "alignment": "similarity transform from selected MTCNN eye landmarks",
                "target_eyes_relative": [[.30, .35], [.70, .35]],
                "interpolation": "INTER_LINEAR", "border": "BORDER_REFLECT_101",
                "source_dir": str(source_dir), "split_dir": str(split_dir),
                "landmarks": str(landmarks_path.resolve()),
                "train": len(tables[0]), "test": len(tables[1])}
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# FaceNet input dataset\n\n"
        "Selected face eye landmarks are applied directly to the EXIF-corrected sources.\n"
        "PNG files are RGB uint8, 160 x 160 x 3, with pixels in [0,255].\n"
        "Train/test membership is unchanged. See train.csv and test.csv.\n"
        "No grayscale, pixel normalization, CLAHE or gamma correction is applied.\n"
        "Use the image file paths with DeepFace.represent and an explicit normalization setting.\n"
        "With prealigned images, set detector_backend='skip' and align=False.\n"
        "NumPy inputs to DeepFace must follow its BGR API convention.\n"
        "Reflected pixels can occur at source boundaries; inspect alignment_preview.png.\n",
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
        with Image.open(split_dir / row.input_split_path) as image:
            pair[0].imshow(image)
        with Image.open(output_dir / row.image_path) as image:
            pair[1].imshow(image)
        pair[0].set_title(row["class"] + " / original crop", fontsize=9)
        pair[1].set_title("Eye alignment / RGB 160 x 160", fontsize=9)
        for ax in pair:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "alignment_preview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    candidates = [BASE / "preprocessed_data/mtcnn_results/detections.csv",
                  BASE / "mtcnn_results/detections.csv"]
    landmarks = next((p for p in candidates if p.is_file()), candidates[0])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=BASE / "cropped_faces_split")
    parser.add_argument("--source-dir", type=Path, default=BASE / "preprocessed_data")
    parser.add_argument("--landmarks", type=Path, default=landmarks)
    parser.add_argument("--output-dir", type=Path, default=BASE / "cropped_faces_facenet_split")
    args = parser.parse_args()
    prepare(args.split_dir, args.source_dir, args.landmarks, args.output_dir)


if __name__ == "__main__":
    main()
