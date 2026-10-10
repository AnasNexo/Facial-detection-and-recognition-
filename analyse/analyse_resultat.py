"""Inspect the committed experiment snapshot and regenerate English report figures.

Run: python analyse/analyse_resultat.py
The results notebook uses these same helpers. This does not train models or
refresh the snapshot from a new experiment.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
NAMES = {
    "svm_linear": "FaceNet + linear SVM",
    "svm_rbf": "FaceNet + RBF SVM",
    "knn": "FaceNet + kNN",
    "pca_svm_linear": "PCA + linear SVM",
    "pca_svm_rbf": "PCA + RBF SVM",
}


def load_snapshot():
    """Recalculate test scores from predictions before accepting summary tables."""
    scores = pd.read_csv(RESULTS / "all_model_metrics.csv")
    if set(scores.model_id) != set(NAMES) or len(scores) != 5:
        raise ValueError("Expected exactly five model summaries")
    scores = scores.set_index("model_id").loc[list(NAMES)].reset_index()
    tables = {family: pd.read_csv(RESULTS / f"{family}_test_predictions.csv")
              for family in ("facenet", "pca")}
    reference = pd.read_csv(BASE / "reference_split/split_manifest.csv")
    train, test = reference[reference.split.eq("train")], reference[reference.split.eq("test")]
    if len(train) != 398 or len(test) != 100 or set(train.source) & set(test.source):
        raise ValueError("Unexpected reference membership")
    test = test.set_index("source").sort_index()
    joined = None
    for family, table in tables.items():
        if not table.source.is_unique:
            raise ValueError("Repeated test sources")
        table = table.set_index("source").sort_index()
        if not table.index.equals(test.index) or not table["class"].equals(test["class"]):
            raise ValueError(f"{family}: predictions do not match the reference test split")
        columns = [c for c in table if c.startswith("prediction_")]
        part = table[["class", *columns]]
        joined = part if joined is None else joined.join(part.drop(columns="class"), validate="one_to_one")
    for row in scores.itertuples():
        predicted = joined[f"prediction_{row.model_id}"]
        p, r, f, _ = precision_recall_fscore_support(joined["class"], predicted,
                                                   average="macro", zero_division=0)
        actual = [accuracy_score(joined["class"], predicted), p, r, f]
        saved = [row.accuracy_test, row.precision_test, row.recall_test, row.f1_test]
        if not np.allclose(actual, saved, atol=1e-10):
            raise ValueError(f"Saved metrics disagree with predictions: {row.model_id}")
        if int((joined["class"] == predicted).sum()) != row.correct:
            raise ValueError("Incorrect saved prediction count")
    return scores, joined


def save(fig, filename):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=140, bbox_inches="tight")
    return fig


def test_plot(scores):
    columns = ["accuracy_test", "precision_test", "recall_test", "f1_test"]
    fig, ax = plt.subplots(figsize=(12, 5))
    (scores.set_index("model_name")[columns] * 100).plot.bar(ax=ax, rot=15)
    ax.set_title("Held-out test results: 100 images across 10 identities")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 110)
    ax.legend(["Accuracy", "Macro precision", "Macro recall", "Macro F1"], loc="lower right")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    return save(fig, "test_metrics.png")


def generalization_plot(scores):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    (scores.set_index("model_name")[["f1_train", "f1_cv", "f1_test"]] * 100).plot.bar(ax=axes[0], rot=25)
    axes[0].set_title("Training, selected CV and test macro F1")
    axes[0].set_ylabel("Macro F1 (%)")
    axes[0].set_xlabel("")
    axes[0].legend(["Training", "Selected CV", "Test"])
    axes[0].set_ylim(0, 108)
    y = np.arange(len(scores))
    lo = (scores.accuracy_test - scores.ci95_low).to_numpy() * 100
    hi = (scores.ci95_high - scores.accuracy_test).to_numpy() * 100
    axes[1].errorbar(scores.accuracy_test * 100, y, xerr=np.array([lo, hi]), fmt="o", capsize=5)
    axes[1].set_yticks(y, scores.model_name)
    axes[1].set_xlim(80, 100)
    axes[1].set_xlabel("Test accuracy (%)")
    axes[1].set_title("Wilson 95% accuracy intervals")
    axes[1].grid(axis="x", alpha=.3)
    return save(fig, "generalization.png")


def learning_plot():
    curves = pd.read_csv(RESULTS / "learning_curves.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (model_id, name) in zip(axes.flat, NAMES.items()):
        rows = curves[curves.model_id.eq(model_id)].sort_values("train_size_target")
        x = rows.train_size_target.to_numpy()
        for prefix, label, color in [("train", "Training", "#3976a8"), ("validation", "Validation", "#dc8a32")]:
            mean, std = rows[f"{prefix}_mean"].to_numpy()*100, rows[f"{prefix}_std"].to_numpy()*100
            ax.plot(x, mean, marker="o", label=label, color=color)
            ax.fill_between(x, mean-std, mean+std, alpha=.15, color=color)
        ax.set_title(name)
        ax.set_xlabel("Target training subset size per fold")
        ax.set_ylabel("Macro F1 (%)")
        ax.set_ylim(60, 103)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Learning curves: fixed selected parameters, five grouped folds")
    return save(fig, "learning_curves.png")


def class_plot():
    counts = pd.read_csv(RESULTS / "per_class_correct.csv").set_index("class")
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(counts, vmin=0, vmax=10, cmap="Blues")
    ax.set_xticks(range(len(counts.columns)), counts.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(counts)), counts.index)
    for i in range(len(counts)):
        for j in range(len(counts.columns)):
            ax.text(j, i, f"{counts.iloc[i,j]}/10", ha="center", va="center",
                    color="white" if counts.iloc[i,j] >= 8 else "black")
    ax.set_title("Correct predictions among the ten test images per identity")
    fig.colorbar(im, ax=ax, label="Correct test predictions")
    return save(fig, "per_identity.png")


def confusion_plots(predictions):
    classes = sorted(predictions["class"].unique())
    figures = []
    for family, ids in [("facenet", list(NAMES)[:3]), ("pca", list(NAMES)[3:])]:
        fig, axes = plt.subplots(1, len(ids), figsize=(8*len(ids), 8), squeeze=False)
        for ax, model_id in zip(axes.flat, ids):
            matrix = confusion_matrix(predictions["class"], predictions[f"prediction_{model_id}"], labels=classes)
            ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax, colorbar=False,
                xticks_rotation=90, cmap="Blues", values_format="d")
            ax.set_title(NAMES[model_id])
        figures.append(save(fig, f"confusion_{family}.png"))
    return figures


def paired_comparisons(predictions):
    """Recompute exact paired McNemar p-values and Holm correction."""
    from itertools import combinations
    from scipy.stats import binomtest
    rows = []
    for a, b in combinations(NAMES, 2):
        ac = predictions[f"prediction_{a}"].eq(predictions["class"])
        bc = predictions[f"prediction_{b}"].eq(predictions["class"])
        aw, bw = int((ac & ~bc).sum()), int((bc & ~ac).sum())
        n = aw + bw
        p = binomtest(aw, n, .5).pvalue if n else 1.
        rows.append({"A": NAMES[a], "B": NAMES[b], "A_correct_B_wrong": aw,
                     "B_correct_A_wrong": bw, "discordant": n, "p_exact": p})
    result = pd.DataFrame(rows)
    order = np.argsort(result.p_exact.to_numpy())
    adjusted = np.maximum.accumulate((len(result)-np.arange(len(result))) * result.p_exact.to_numpy()[order])
    result["p_holm"] = 1.
    result.loc[order, "p_holm"] = np.minimum(adjusted, 1.)
    result["significant_5pct"] = result.p_holm < .05
    return result


def main():
    plt.switch_backend("Agg")
    scores, predictions = load_snapshot()
    print(scores[["model_name", "accuracy_test", "precision_test", "recall_test", "f1_test", "f1_cv"]].to_string(index=False))
    for fig in [test_plot(scores), generalization_plot(scores), learning_plot(), class_plot(), *confusion_plots(predictions)]:
        plt.close(fig)
    paired = paired_comparisons(predictions)
    print(f"Paired comparisons significant after Holm correction: {paired.significant_5pct.sum()}/10")
    print(f"Snapshot validated; figures saved in {FIGURES}")


if __name__ == "__main__":
    main()
