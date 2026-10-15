"""Create dataset and experiment visualizations for CliniFact.

The script is intentionally tolerant of partial experiment output. It creates
the plots supported by the files that are present and reports skipped plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


LABEL_NAMES = ["inconclusive", "evidence", "NEI"]
LABEL_COLORS = ["#4C78A8", "#59A14F", "#F28E2B"]


def save_figure(figure: Figure, output_dir: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_table(
    table: pd.DataFrame, output_dir: Path, name: str, title: str
) -> None:
    """Save a metric table as CSV and as a readable PNG image."""
    table.to_csv(output_dir / f"{name}.csv", index=False)
    if table.empty:
        return
    display = table.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else ""
        )
    figure, axis = plt.subplots(
        figsize=(max(9, len(display.columns) * 1.35), max(2.5, len(display) * 0.42))
    )
    axis.axis("off")
    axis.set_title(title, fontsize=14, fontweight="bold", pad=18)
    table_artist = axis.table(
        cellText=display.astype(str).values.tolist(),
        colLabels=display.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(1, 1.35)
    for (row, _), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor("#2F5597")
            cell.set_text_props(color="white", weight="bold")
    save_figure(figure, output_dir, name)


def load_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    splits = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}_set.csv"
        if path.exists():
            splits[split] = pd.read_csv(path)
    return splits


def plot_label_distribution(
    splits: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    counts = pd.DataFrame(
        {
            split: frame["label"].value_counts().reindex(range(3), fill_value=0)
            for split, frame in splits.items()
            if "label" in frame
        }
    )
    if counts.empty:
        print("Skipped label distribution: no label columns found.")
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    counts.index = LABEL_NAMES[: len(counts.index)]
    counts.plot.bar(ax=axis, color=LABEL_COLORS) # type: ignore
    axis.set_title("CliniFact label distribution")
    axis.set_xlabel("Label")
    axis.set_ylabel("Examples")
    axis.legend(title="Split")
    save_figure(figure, output_dir, "label_distribution")

    proportions = counts.div(counts.sum(axis=0), axis=1).T.reset_index()
    proportions = proportions.rename(columns={"index": "split"})
    proportions.columns = ["split", *LABEL_NAMES]
    save_table(proportions, output_dir, "label_proportions_table", "Label proportions")

    figure, axis = plt.subplots(figsize=(8, 5))
    proportions.set_index("split")[LABEL_NAMES].plot.bar(
        stacked=True, ax=axis, color=LABEL_COLORS # type: ignore
    )
    axis.set_title("Label proportions by split")
    axis.set_xlabel("Split")
    axis.set_ylabel("Proportion")
    axis.set_ylim(0, 1)
    axis.legend(title="Label")
    save_figure(figure, output_dir, "label_proportions")


def write_dataset_tables(
    splits: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    rows = []
    label_rows = []
    for split, frame in splits.items():
        row = {"split": split, "examples": len(frame), "columns": len(frame.columns)}
        if "claim" in frame:
            row["mean_claim_words"] = frame["claim"].fillna("").str.split().str.len().mean() # type: ignore
        if "article_abstract" in frame:
            row["mean_abstract_words"] = (
                frame["article_abstract"].fillna("").str.split().str.len().mean() # type: ignore
            )
        rows.append(row)
        if "label" in frame:
            for label, count in frame["label"].value_counts().sort_index().items():
                label_id = int(str(label))
                label_rows.append(
                    {
                        "split": split,
                        "label_id": label_id,
                        "label_name": LABEL_NAMES[label_id]
                        if label_id < len(LABEL_NAMES)
                        else str(label),
                        "count": int(count),
                        "proportion": float(count / len(frame)),
                    }
                )
    save_table(
        pd.DataFrame(rows),
        output_dir,
        "dataset_summary_table",
        "Dataset summary",
    )
    save_table(
        pd.DataFrame(label_rows),
        output_dir,
        "dataset_label_counts_table",
        "Dataset label counts",
    )


def plot_dataset_categories(
    splits: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    combined = []
    for split, frame in splits.items():
        current = frame.copy()
        current["split"] = split
        combined.append(current)
    data = pd.concat(combined, ignore_index=True)
    for column, title, filename in (
        ("outcome_type", "Outcome type", "outcome_type_distribution"),
        ("intervention_group_arm_group_type", "Intervention arm type", "intervention_type_distribution"),
        ("Publication Date", "Publication year", "publication_year_distribution"),
    ):
        if column not in data:
            continue
        values = data[column].fillna("Unknown").astype(str)
        if column == "Publication Date":
            values = pd.to_numeric(values.str.extract(r"(\d{4})")[0], errors="coerce")
            values = values.dropna().astype(int).astype(str)
        counts = pd.crosstab(data.loc[values.index, "split"], values)
        if counts.empty:
            continue
        figure, axis = plt.subplots(figsize=(max(8, len(counts.columns) * 0.8), 5))
        counts.plot.bar(ax=axis)
        axis.set_title(f"{title} by split")
        axis.set_xlabel("Split")
        axis.set_ylabel("Examples")
        axis.legend(title=title, bbox_to_anchor=(1.02, 1), loc="upper left")
        save_figure(figure, output_dir, filename)


def plot_text_lengths(
    splits: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    frames = []
    for split, frame in splits.items():
        if {"claim", "article_abstract"}.issubset(frame.columns):
            current = pd.DataFrame(
                {
                    "split": split,
                    "claim_words": frame["claim"].fillna("").str.split().str.len(), # type: ignore
                    "abstract_words": frame["article_abstract"]
                    .fillna("")
                    .str.split()
                    .str.len(), # type: ignore
                }
            )
            frames.append(current)
    if not frames:
        print("Skipped text lengths: claim/abstract columns not found.")
        return
    lengths = pd.concat(frames, ignore_index=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, column, title in zip(
        axes,
        ("claim_words", "abstract_words"),
        ("Claim length", "Abstract length"),
    ):
        for split, group in lengths.groupby("split"):
            axis.hist(group[column], bins=30, alpha=0.55, label=split)
        axis.set_title(f"{title} in words")
        axis.set_xlabel("Words")
        axis.set_ylabel("Examples")
        axis.legend()
    save_figure(figure, output_dir, "text_length_distributions")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, column, title in zip(
        axes,
        ("claim_words", "abstract_words"),
        ("Claim length", "Abstract length"),
    ):
        lengths.boxplot(column=column, by="split", ax=axis, grid=False)
        axis.set_title(f"{title} by split")
        axis.set_xlabel("Split")
        axis.set_ylabel("Words")
    figure.suptitle("")
    save_figure(figure, output_dir, "text_length_boxplots")


def load_training_metrics(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("seed_*/training_metrics.csv")):
        frame = pd.read_csv(path)
        frame = frame.rename(
            columns={
                "Epoch": "epoch",
                "Average Training Loss": "average_train_loss",
                "Validation Accuracy": "validation_accuracy",
                "Validation F1 Macro": "validation_f1_macro",
            }
        )
        frame["seed"] = path.parent.name.removeprefix("seed_")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_training_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        print("Skipped training plots: training_metrics.csv not found.")
        return
    metric_columns = [
        column
        for column in (
            "average_train_loss",
            "validation_accuracy",
            "validation_f1_macro",
        )
        if column in metrics.columns
    ]
    if not metric_columns or "epoch" not in metrics.columns:
        print("Skipped training plots: no supported metrics or epoch column found.")
        return
    sort_columns = [column for column in ("seed", "epoch") if column in metrics.columns]
    save_table(
        metrics.sort_values(sort_columns),
        output_dir,
        "training_metrics_table",
        "Training metrics",
    )
    figure, axes = plt.subplots(1, len(metric_columns), figsize=(6 * len(metric_columns), 5))
    axes = np.atleast_1d(axes)
    for seed, group in metrics.groupby("seed"):
        for axis, column in zip(axes, metric_columns):
            axis.plot(group["epoch"], group[column], label=f"seed {seed}")
    titles = {
        "average_train_loss": ("Training loss by epoch", "Average training loss"),
        "validation_accuracy": ("Validation accuracy by epoch", "Accuracy"),
        "validation_f1_macro": ("Validation macro-F1 by epoch", "Macro-F1"),
    }
    for axis, column in zip(axes, metric_columns):
        axis.set_title(titles[column][0])
        axis.set_xlabel("Epoch")
        axis.set_ylabel(titles[column][1])
        axis.legend()
        axis.grid(alpha=0.25)
    save_figure(figure, output_dir, "training_validation_metrics")


def load_seed_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("seed_*/metrics.json")):
        with path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        report = metrics.get("classification_report", {})
        rows.append(
            {
                "seed": metrics.get("seed", path.parent.name.removeprefix("seed_")),
                "accuracy": metrics.get("accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "best_validation_f1_macro": metrics.get(
                    "best_validation_f1_macro"
                ),
                "precision_macro": report.get("macro avg", {}).get("precision"),
                "recall_macro": report.get("macro avg", {}).get("recall"),
            }
        )
    return pd.DataFrame(rows)


def plot_seed_metrics(seed_metrics: pd.DataFrame, output_dir: Path) -> None:
    if seed_metrics.empty:
        print("Skipped seed comparison: seed metrics.json files not found.")
        return
    metric_columns = [
        column
        for column in (
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "best_validation_f1_macro",
        )
        if column in seed_metrics
    ]
    save_table(seed_metrics, output_dir, "seed_metrics_table", "Metrics by seed")
    values = seed_metrics.set_index("seed")[metric_columns]
    figure, axis = plt.subplots(figsize=(9, 5))
    values.plot.bar(ax=axis)
    axis.set_title("Performance by random seed")
    axis.set_xlabel("Seed")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, output_dir, "seed_metric_comparison")


def find_predictions(results_dir: Path) -> pd.DataFrame:
    paths = sorted(results_dir.glob("seed_*/test_predictions.csv"))
    if not paths:
        return pd.DataFrame()
    frame = pd.read_csv(paths[0])
    return frame


def plot_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    required = {"label", "Predicted_Label"}
    if not required.issubset(predictions.columns):
        print("Skipped confusion matrix: prediction label columns not found.")
        return
    matrix = confusion_matrix(
        predictions["label"], predictions["Predicted_Label"], labels=range(3)
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=LABEL_NAMES).plot(
        ax=axis, cmap="Blues", values_format="d", colorbar=False
    )
    axis.set_title("Test confusion matrix")
    save_figure(figure, output_dir, "test_confusion_matrix")

    report = pd.DataFrame(
        classification_report(
            predictions["label"],
            predictions["Predicted_Label"],
            labels=range(3),
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index().rename(columns={"index": "class"})
    save_table(report, output_dir, "classification_report_table", "Classification report")

    for normalize, suffix, title in (
        ("true", "normalized_by_true", "Confusion matrix normalized by true label"),
        ("pred", "normalized_by_prediction", "Confusion matrix normalized by prediction"),
    ):
        figure, axis = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(
            confusion_matrix(
                predictions["label"],
                predictions["Predicted_Label"],
                labels=range(3),
                normalize=normalize, # type: ignore
            ),
            display_labels=LABEL_NAMES,
        ).plot(ax=axis, cmap="Blues", values_format=".2f", colorbar=False)
        axis.set_title(title)
        save_figure(figure, output_dir, f"test_confusion_matrix_{suffix}")

    per_class = report[report["class"].isin(LABEL_NAMES)].set_index("class")[
        ["precision", "recall", "f1-score"]
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    per_class.plot.bar(ax=axis, color=["#4C78A8", "#59A14F", "#E15759"])
    axis.set_title("Per-class precision, recall, and F1")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, output_dir, "per_class_metrics")

    error_rows = predictions[predictions["label"] != predictions["Predicted_Label"]].copy()
    error_rows.to_csv(output_dir / "misclassified_examples.csv", index=False)
    if {"claim", "article_abstract"}.issubset(error_rows.columns):
        error_rows["claim_words"] = error_rows["claim"].fillna("").str.split().str.len() # type: ignore
        error_rows["abstract_words"] = error_rows["article_abstract"].fillna("").str.split().str.len() # type: ignore
        save_table(
            error_rows.groupby(["label", "Predicted_Label"], as_index=False)
            .size()
            .rename(columns={"size": "errors"}),
            output_dir,
            "error_summary_table",
            "Misclassification summary",
        )

    if "confidence" in predictions.columns:
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.hist(predictions["confidence"].dropna(), bins=20, color="#4C78A8")
        axis.set_title("Prediction confidence")
        axis.set_xlabel("Confidence")
        axis.set_ylabel("Examples")
        save_figure(figure, output_dir, "prediction_confidence")


def calculate_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions["label"].astype(int)
    y_pred = predictions["Predicted_Label"].astype(int)
    return pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "mcc": matthews_corrcoef(y_true, y_pred),
                "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            }
        ]
    )


def plot_score_metrics(predictions: pd.DataFrame, output_dir: Path) -> None:
    score_columns = [f"score_{index}" for index in range(3)]
    if not set(score_columns).issubset(predictions.columns):
        return
    y_true = predictions["label"].astype(int)
    scores = predictions[score_columns].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    plotted_classes = 0
    for index, label in enumerate(LABEL_NAMES):
        binary_true = (y_true == index).astype(int)
        if binary_true.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(binary_true, scores[:, index])
        auc = roc_auc_score(binary_true, scores[:, index])
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
        precision, recall, _ = precision_recall_curve(binary_true, scores[:, index])
        axes[1].plot(recall, precision, label=label)
        plotted_classes += 1
    if not plotted_classes:
        plt.close(figure)
        print("Skipped score curves: each class requires positive and negative examples.")
        return
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title("One-vs-rest ROC curves")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[1].set_title("One-vs-rest precision-recall curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    save_figure(figure, output_dir, "roc_and_precision_recall_curves")


def write_metrics_summary(
    seed_metrics: pd.DataFrame, output_dir: Path
) -> None:
    summary = {}
    if not seed_metrics.empty:
        for column in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
            values = pd.to_numeric(seed_metrics[column], errors="coerce").dropna()
            if len(values):
                summary[column] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create CliniFact dataset and experiment visualizations."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/primary_outcome_publication_dataset"),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("experiments/results/biobert_baseline")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.results_dir / "visualizations"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.data_dir)
    if not splits:
        raise FileNotFoundError(f"No split CSV files found in {args.data_dir}")
    plot_label_distribution(splits, args.output_dir)
    write_dataset_tables(splits, args.output_dir)
    plot_dataset_categories(splits, args.output_dir)
    plot_text_lengths(splits, args.output_dir)

    training_metrics = load_training_metrics(args.results_dir)
    plot_training_metrics(training_metrics, args.output_dir)

    seed_metrics = load_seed_metrics(args.results_dir)
    plot_seed_metrics(seed_metrics, args.output_dir)
    predictions = find_predictions(args.results_dir)
    plot_predictions(predictions, args.output_dir)
    if {"label", "Predicted_Label"}.issubset(predictions.columns):
        save_table(
            calculate_prediction_metrics(predictions),
            args.output_dir,
            "prediction_metrics_table",
            "Test prediction metrics",
        )
        plot_score_metrics(predictions, args.output_dir)
    write_metrics_summary(seed_metrics, args.output_dir)
    print(f"Visualization files written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()