"""Train PCA + linear/RBF SVM using train-only CV, then evaluate held-out test.

Run: python -u analyse/train_pca.py
Results, fitted pipelines and figures are saved in preprocessed_crop_pca_split/pca_results.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


BASE = Path(__file__).resolve().parent
SEED = 42


def load_data(dataset, split):
    manifest = pd.read_csv(dataset / f"{split}.csv")
    with np.load(dataset / f"{split}.npz", allow_pickle=False) as archive:
        X, y, sources = archive["X"], archive["y"], archive["sources"]
    if X.ndim != 2 or X.shape != (len(manifest), 6400):
        raise ValueError(f"{split}: expected flattened 80x80 grayscale images")
    if not np.isfinite(X).all() or X.min() < 0 or X.max() > 1:
        raise ValueError(f"{split}: invalid normalized pixels")
    if not (np.array_equal(y, manifest["class"].to_numpy())
            and np.array_equal(sources, manifest.source.to_numpy())
            and manifest["split"].eq(split).all() and manifest.source.is_unique):
        raise ValueError(f"{split}: archive and manifest disagree")
    return X, y, manifest


def make_folds(train, test, report):
    """Use the same five grouped folds as the FaceNet comparison."""
    parent = list(range(len(train)))
    lookup = {source: i for i, source in enumerate(train.source)}

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
    if report.exists():
        test_sources = set(test.source)
        for pair in pd.read_csv(report).itertuples():
            a, b = pair.source_a, pair.source_b
            if (a in lookup and b in test_sources) or (b in lookup and a in test_sources):
                raise ValueError("Flagged near-duplicates overlap train/test")
            if a in lookup and b in lookup:
                union(lookup[a], lookup[b])
    groups = np.array([find(i) for i in range(len(train))])
    n_folds = min(5, int(train.groupby("class").size().min()))
    if n_folds < 2:
        raise ValueError("Insufficient training samples")
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.zeros(len(train)), train["class"], groups))
    for fit, validation in folds:
        if set(groups[fit]) & set(groups[validation]):
            raise ValueError("Duplicate group overlaps CV folds")
        if set(train.iloc[fit]["class"]) != set(train.iloc[validation]["class"]):
            raise ValueError("A CV fold is missing a class")
    return folds


def run(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("pca_training")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(output / "training.log", mode="w", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        log.addHandler(handler)
    X_train, y_train, train = load_data(dataset, "train")
    X_test, y_test, test = load_data(dataset, "test")
    if set(train.source) & set(test.source) or set(train.source_pixel_sha256) & set(test.source_pixel_sha256):
        raise ValueError("Train/test overlap")
    if set(y_train) != set(y_test):
        raise ValueError("Train/test classes differ")
    folds = make_folds(train, test, BASE / "cropped_faces_split/near_duplicate_check.csv")
    components = [n for n in (50, 100, 150) if n < min(len(fit) for fit, _ in folds)]
    log.info("%d train / %d test / %d classes / %d pixels per image", len(train), len(test), len(set(y_train)), X_train.shape[1])
    log.info("%d-fold grouped CV; selection=macro F1; PCA components=%s; whiten=False/True", len(folds), components)
    log.info("PCA is fitted separately inside each training fold. Test is reserved for final evaluation.")
    trained = {}
    # Cache identical PCA fits across SVM parameter candidates, then discard cache.
    with tempfile.TemporaryDirectory(prefix="pca-cache-", dir=output) as cache_dir:
        memory = joblib.Memory(cache_dir, verbose=0)
        for name, kernel, svm_grid in [
            ("pca_svm_linear", "linear", {"svm__C": [.1, 1, 10, 100]}),
            ("pca_svm_rbf", "rbf", {"svm__C": [1, 10, 100], "svm__gamma": ["scale", .001, .01, .1]}),
        ]:
            pipeline = Pipeline([
                ("pca", PCA(svd_solver="randomized", random_state=SEED)),
                ("svm", SVC(kernel=kernel))], memory=memory)
            grid = {"pca__n_components": components, "pca__whiten": [False, True], **svm_grid}
            search = GridSearchCV(pipeline, grid,
                scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
                refit="f1_macro", cv=folds, n_jobs=1, verbose=3, error_score="raise")
            log.info("TRAINING %s", name)
            started = time.perf_counter()
            with threadpool_limits(limits=2):
                search.fit(X_train, y_train)
            elapsed = time.perf_counter() - started
            search.best_estimator_.memory = None
            trained[name] = (search, elapsed)
            pd.DataFrame(search.cv_results_).to_csv(output / f"{name}_cv_results.csv", index=False)
            joblib.dump({"pipeline": search.best_estimator_, "best_params": search.best_params_,
                         "input_shape": [80, 80], "input_range": [0, 1], "classes": np.unique(y_train)},
                        output / f"{name}.joblib")
            log.info("TRAINED %s | CV F1=%.2f%% | %s | %.1fs", name,
                     100*search.best_score_, search.best_params_, elapsed)

    selected = max(trained, key=lambda name: trained[name][0].best_score_)
    log.info("CV-selected model: %s. Starting held-out test evaluation.", selected)
    rows, matrices = [], {}
    predictions = test[["source", "image_path", "class"]].copy()
    classes = np.unique(y_train)
    for name, (search, elapsed) in trained.items():
        with threadpool_limits(limits=2):
            predicted = search.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(y_test, predicted, average="weighted", zero_division=0)
        rows.append({"model": name, "accuracy": accuracy_score(y_test, predicted),
            "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
            "precision_weighted": weighted[0], "recall_weighted": weighted[1], "f1_weighted": weighted[2],
            "correct": int((y_test == predicted).sum()), "test_images": len(test),
            "cv_f1_macro": search.best_score_, "training_seconds": elapsed,
            "best_params": json.dumps(search.best_params_)})
        predictions[f"prediction_{name}"] = predicted
        predictions[f"correct_{name}"] = predicted == y_test
        report = classification_report(y_test, predicted, labels=classes, zero_division=0, output_dict=True)
        pd.DataFrame(report).T.to_csv(output / f"{name}_classification_report.csv")
        matrices[name] = confusion_matrix(y_test, predicted, labels=classes)
        pd.DataFrame(matrices[name], index=classes, columns=classes).to_csv(output / f"{name}_confusion_matrix.csv")
        log.info("TEST %s | accuracy=%.2f%% | precision=%.2f%% | recall=%.2f%% | F1=%.2f%% | %d/%d correct",
                 name, 100*rows[-1]["accuracy"], 100*precision, 100*recall, 100*f1, rows[-1]["correct"], len(test))
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "test_metrics.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    per_class = predictions.groupby("class").agg(
        test_images=("source", "size"),
        linear_correct=("correct_pca_svm_linear", "sum"), rbf_correct=("correct_pca_svm_rbf", "sum"))
    per_class.to_csv(output / "per_class_correct.csv")
    save_figures(scores, matrices, classes, trained, output)
    summary = ["# PCA + SVM: held-out test comparison", "",
        f"{len(train)} train / {len(test)} test / {len(classes)} classes.",
        f"PCA and hyperparameters selected inside {len(folds)} grouped training folds.",
        f"CV-selected model: **{selected}**.", "",
        "| Model | Accuracy | Precision macro | Recall macro | F1 macro |",
        "|---|---:|---:|---:|---:|"]
    for row in rows:
        summary.append(f"| {row['model']} | {row['accuracy']:.2%} | {row['precision_macro']:.2%} | {row['recall_macro']:.2%} | {row['f1_macro']:.2%} |")
    for row in rows:
        summary.extend(["", f"{row['model']} parameters: `{row['best_params']}`; CV F1: {row['cv_f1_macro']:.2%}."])
    difference = rows[1]["correct"] - rows[0]["correct"]
    summary.extend(["", f"RBF minus linear: {difference:+d} correct predictions on the same 100 test images.",
        "Scores measure classification of the 10 known identities on this split.",
        "With 100 test images, one prediction changes accuracy by one percentage point.",
        "Use train-only validation for further tuning; do not tune against this test set."])
    (output / "comparison.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    (output / "settings.json").write_text(json.dumps({"dataset": str(dataset), "train": len(train),
        "test": len(test), "random_state": SEED, "cv": "StratifiedGroupKFold",
        "folds": len(folds), "selection": "f1_macro", "cv_selected_model": selected,
        "pca_components": components, "pca_whiten": [False, True], "pixel_standard_scaler": False}, indent=2), encoding="utf-8")
    log.info("DONE: %s", output)
    log.info("\n%s", scores[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].to_string(index=False))


def save_figures(scores, matrices, classes, trained, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    (scores.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]]*100).plot.bar(ax=ax, rot=0)
    ax.set_ylim(0, 110); ax.set_ylabel("Test score (%)"); ax.set_title("PCA + SVM: held-out test comparison")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(output / "test_comparison.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (name, matrix) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax,
            colorbar=False, xticks_rotation=90, cmap="Blues", values_format="d")
        ax.set_title(name)
    fig.tight_layout(); fig.savefig(output / "confusion_matrices.png", dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, (search, _)) in zip(axes, trained.items()):
        pca = search.best_estimator_.named_steps["pca"]
        ax.plot(np.arange(1, pca.n_components_+1), np.cumsum(pca.explained_variance_ratio_)*100)
        ax.set_title(name); ax.set_xlabel("PCA components"); ax.set_ylabel("Cumulative explained variance (%)")
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(output / "pca_explained_variance.png", dpi=120); plt.close(fig)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BASE / "preprocessed_crop_pca_split")
    parser.add_argument("--output", type=Path, default=BASE / "preprocessed_crop_pca_split/pca_results")
    args = parser.parse_args()
    run(args.dataset.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
