"""Frozen FaceNet512 embeddings + train-only CV for linear SVM, RBF SVM and kNN.

Run: python -u analyse/train_facenet_transfert.py
Progress is printed immediately and saved to facenet_transfer_results/training.log.
The first run downloads FaceNet512 weights into the project's .deepface directory.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


BASE = Path(__file__).resolve().parent
MODEL = "Facenet512"
NORMALIZATION = "Facenet"
SEED = 42


def setup_logger(output):
    logger = logging.getLogger("facenet_transfer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(output / "training.log", mode="w", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def load_manifest(dataset, split):
    table = pd.read_csv(dataset / f"{split}.csv")
    if table.empty or not table["split"].eq(split).all():
        raise ValueError(f"Invalid {split} manifest")
    for _, row in table.iterrows():
        path = (dataset / row.image_path).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError(f"Image outside dataset: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (160, 160):
                raise ValueError(f"Expected RGB 160x160: {path}")
            if np.asarray(image).std() < 1e-6:
                raise ValueError(f"Constant image cannot be standardized: {path}")
    return table


def fingerprint(dataset, table):
    digest = hashlib.sha256()
    digest.update(f"{MODEL}|{NORMALIZATION}|{version('deepface')}|L2".encode())
    for _, row in table.iterrows():
        digest.update(json.dumps([row.source, row["class"], row.image_path]).encode())
        digest.update((dataset / row.image_path).read_bytes())
    return digest.hexdigest()


def cv_splits(dataset, train, test, folds):
    """Keep exact duplicates and previously flagged near-duplicates in one fold."""
    parent = list(range(len(train)))
    indices = {source: i for i, source in enumerate(train.source)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    for _, rows in train.groupby("source_pixel_sha256"):
        for i in rows.index[1:]:
            union(int(rows.index[0]), int(i))
    report = dataset.parent / "cropped_faces_split" / "near_duplicate_check.csv"
    if report.exists():
        for pair in pd.read_csv(report).itertuples():
            a, b = pair.source_a, pair.source_b
            if (a in indices and b in set(test.source)) or (b in indices and a in set(test.source)):
                raise ValueError("Flagged near-duplicate pair overlaps train/test")
            if a in indices and b in indices:
                union(indices[a], indices[b])
    groups = np.asarray([find(i) for i in range(len(train))])
    if len(np.unique(groups)) == len(train):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        return list(cv.split(np.zeros(len(train)), train["class"])), "StratifiedKFold"
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    splits = list(cv.split(np.zeros(len(train)), train["class"], groups=groups))
    for fit, validate in splits:
        if set(groups[fit]) & set(groups[validate]):
            raise ValueError("Duplicate group overlaps CV folds")
    return splits, "StratifiedGroupKFold"


def extract_embeddings(dataset, table, split, output, batch_size, logger):
    signature = fingerprint(dataset, table)
    cache = output / f"{split}_embeddings.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["signature"].item()) == signature:
                X = data["X"]
                if X.shape == (len(table), 512) and np.isfinite(X).all():
                    logger.info("%s: loaded %d cached embeddings", split, len(X))
                    return X

    # Import only after configuring the project-local weight cache in main().
    from deepface import DeepFace
    logger.info("%s: initializing frozen %s (first use may download weights)", split, MODEL)
    model = DeepFace.build_model(MODEL)
    model.model.trainable = False
    started = time.perf_counter()
    batches = []
    for start in range(0, len(table), batch_size):
        paths = [str(dataset / rel) for rel in table.image_path.iloc[start:start + batch_size]]
        results = DeepFace.represent(
            img_path=paths, model_name=MODEL, detector_backend="skip",
            enforce_detection=False, align=False, normalization=NORMALIZATION)
        if len(results) != len(paths):
            raise ValueError("DeepFace returned an incorrect batch size")
        for faces in results:
            if len(faces) != 1:
                raise ValueError("Expected exactly one embedding per prealigned crop")
            batches.append(np.asarray(faces[0]["embedding"], dtype=np.float32))
        logger.info("%s embeddings: %d/%d | elapsed %.1fs", split,
                    min(start + batch_size, len(table)), len(table), time.perf_counter() - started)
    X = np.stack(batches)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if X.shape != (len(table), 512) or not np.isfinite(X).all() or (norms <= 0).any():
        raise ValueError("Invalid FaceNet512 embeddings")
    X = (X / norms).astype(np.float32)
    np.savez_compressed(cache, X=X, y=table["class"].to_numpy(dtype=str),
                        sources=table.source.to_numpy(dtype=str), signature=np.asarray(signature))
    return X


def run(dataset, output, batch_size):
    dataset, output = dataset.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output)
    train, test = load_manifest(dataset, "train"), load_manifest(dataset, "test")
    if not train.source.is_unique or not test.source.is_unique or set(train.source) & set(test.source):
        raise ValueError("Repeated or overlapping train/test sources")
    if set(train["class"]) != set(test["class"]):
        raise ValueError("Train and test classes differ")
    if set(train.source_pixel_sha256) & set(test.source_pixel_sha256):
        raise ValueError("Identical source pixels overlap train/test")
    folds = min(5, int(train.groupby("class").size().min()))
    if folds < 2:
        raise ValueError("Too few training samples for cross-validation")
    logger.info("Dataset: %d train / %d test / %d identities", len(train), len(test), train["class"].nunique())
    logger.info("Frozen FaceNet512; pixel normalization=%s; embedding normalization=L2", NORMALIZATION)
    X_train = extract_embeddings(dataset, train, "train", output, batch_size, logger)
    y_train = train["class"].to_numpy(dtype=str)
    cv, cv_method = cv_splits(dataset, train, test, folds)
    logger.info("CV: %s, keeping flagged duplicate groups together", cv_method)
    experiments = {
        "svm_linear": (SVC(kernel="linear"), {"svc__C": [.1, 1, 10, 100]}),
        "svm_rbf": (SVC(kernel="rbf"), {"svc__C": [1, 10, 100], "svc__gamma": ["scale", .01, .1]}),
        "knn": (KNeighborsClassifier(metric="cosine", algorithm="brute"),
                {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7],
                 "kneighborsclassifier__weights": ["uniform", "distance"]}),
    }
    trained = {}
    # Every model and its hyperparameters are selected using train only.
    for name, (classifier, grid) in experiments.items():
        logger.info("TRAINING %s: %d-fold CV, selection metric=macro F1", name, folds)
        search = GridSearchCV(make_pipeline(Normalizer(norm="l2"), classifier), grid,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            refit="f1_macro", cv=cv, n_jobs=1, verbose=3, error_score="raise")
        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        logger.info("TRAINED %s | CV macro F1=%.4f | params=%s | %.1fs", name,
                    search.best_score_, search.best_params_, elapsed)
        pd.DataFrame(search.cv_results_).to_csv(output / f"{name}_cv_results.csv", index=False)
        joblib.dump({"classifier": search.best_estimator_, "embedding_model": MODEL,
                     "normalization": NORMALIZATION, "embedding_l2": True,
                     "classes": np.unique(y_train), "best_params": search.best_params_},
                    output / f"{name}.joblib")
        trained[name] = (search, elapsed)

    logger.info("All training complete. Starting final evaluation on held-out test images.")
    X_test = extract_embeddings(dataset, test, "test", output, batch_size, logger)
    y_test = test["class"].to_numpy(dtype=str)
    metrics, matrices = [], {}
    predictions = test[["source", "image_path", "class"]].copy()
    classes = np.unique(y_train)
    for name, (search, elapsed) in trained.items():
        predicted = search.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predicted, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(y_test, predicted, average="weighted", zero_division=0)
        row = {"model": name, "accuracy": accuracy_score(y_test, predicted),
               "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
               "precision_weighted": weighted[0], "recall_weighted": weighted[1],
               "f1_weighted": weighted[2], "cv_f1_macro": search.best_score_,
               "correct": int((y_test == predicted).sum()), "test_images": len(test),
               "training_seconds": elapsed, "best_params": json.dumps(search.best_params_)}
        metrics.append(row)
        logger.info("TEST %s | accuracy=%.2f%% | precision=%.2f%% | recall=%.2f%% | F1=%.2f%% | %d/%d correct",
                    name, 100*row["accuracy"], 100*precision, 100*recall, 100*f1, row["correct"], len(test))
        report = classification_report(y_test, predicted, labels=classes, zero_division=0, output_dict=True)
        pd.DataFrame(report).T.to_csv(output / f"{name}_classification_report.csv")
        matrices[name] = confusion_matrix(y_test, predicted, labels=classes)
        pd.DataFrame(matrices[name], index=classes, columns=classes).to_csv(output / f"{name}_confusion_matrix.csv")
        predictions[f"prediction_{name}"] = predicted
        predictions[f"correct_{name}"] = predicted == y_test
        pd.DataFrame(metrics).to_csv(output / "test_metrics.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    scores = pd.DataFrame(metrics)
    plot_results(scores, matrices, classes, output)
    best_cv = max(trained, key=lambda name: trained[name][0].best_score_)
    summary = ["# FaceNet512 frozen transfer learning", "",
        f"Train: {len(train)}; test: {len(test)}; identities: {len(classes)}.",
        f"Parameters selected by {folds}-fold train-only CV (macro F1).",
        f"Model selected by CV: **{best_cv}**.", "",
        "| Model | Accuracy | Precision macro | Recall macro | F1 macro |",
        "|---|---:|---:|---:|---:|"]
    for row in metrics:
        summary.append(f"| {row['model']} | {row['accuracy']:.2%} | {row['precision_macro']:.2%} | {row['recall_macro']:.2%} | {row['f1_macro']:.2%} |")
    summary.extend(["", "Scores describe classification among these 10 known identities on this split.",
        "100 test images give limited precision: one error changes accuracy by one percentage point.",
        "Differences of a few predictions are not sufficient to establish a reliable winner.",
        "Unseen identities and webcam performance have not been evaluated.",
        "These celebrity identities/images may overlap the pretrained model's training data; independence is not established.",
        "No FaceNet weights were fine-tuned. Pixel normalization is applied by DeepFace in memory."])
    (output / "comparison.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    (output / "settings.json").write_text(json.dumps({"model": MODEL, "frozen": True,
        "normalization": NORMALIZATION, "embedding_l2": True, "random_state": SEED,
        "cv_folds": folds, "cv_method": cv_method, "cv_selection": "f1_macro", "cv_selected_model": best_cv,
        "train": len(train), "test": len(test), "deepface_version": version("deepface"),
        "sklearn_version": version("scikit-learn"), "dataset": str(dataset)}, indent=2), encoding="utf-8")
    logger.info("DONE. Results: %s", output)
    logger.info("\n%s", scores[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "cv_f1_macro"]].to_string(index=False))


def plot_results(scores, matrices, classes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    (scores.set_index("model")[columns] * 100).plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Test score (%)"); ax.set_ylim(0, 110)
    ax.set_title("Frozen FaceNet512: held-out test classification")
    ax.legend(["Accuracy", "Precision macro", "Recall macro", "F1 macro"], loc="lower right")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    fig.tight_layout(); fig.savefig(output / "test_comparison.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(23, 8))
    for ax, (name, matrix) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax, colorbar=False,
            xticks_rotation=90, cmap="Blues", values_format="d")
        ax.set_title(name)
    fig.tight_layout(); fig.savefig(output / "confusion_matrices.png", dpi=120); plt.close(fig)


def main():
    # DeepFace's download messages contain Unicode characters on Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BASE / "cropped_faces_facenet_split")
    parser.add_argument("--output", type=Path, default=BASE / "cropped_faces_facenet_split/facenet_transfer_results")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    os.environ.setdefault("DEEPFACE_HOME", str(BASE.parent))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    run(args.dataset, args.output, args.batch_size)


if __name__ == "__main__":
    main()
