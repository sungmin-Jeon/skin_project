from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


METRICS = (
    "grade_macro_mae",
    "grade_macro_exact_accuracy",
    "grade_macro_balanced_accuracy",
    "grade_macro_qwk",
    "continuous_macro_normalized_mae",
)


def short_target(value: str) -> str:
    return value.removeprefix("ann__").removeprefix("eq__")


def discover_runs(outputs_dir: Path, requested: list[str]) -> list[Path]:
    if requested:
        candidates = [outputs_dir / name for name in requested]
    else:
        candidates = sorted(
            path.parent for path in outputs_dir.glob("*/summary.json")
        )
    return [path for path in candidates if (path / "summary.json").is_file()]


def load_run_table(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        rows.append({"run": run_dir.name, **summary})
    return pd.DataFrame(rows)


def plot_run_comparison(table: pd.DataFrame, output: Path) -> None:
    available = [
        metric
        for metric in METRICS
        if metric in table and table[metric].notna().any()
    ]
    figure, axes = plt.subplots(
        1,
        len(available),
        figsize=(4.2 * len(available), 4.5),
        squeeze=False,
    )
    for axis, metric in zip(axes.ravel(), available):
        sns.barplot(data=table, x="run", y=metric, ax=axis, color="#4C78A8")
        title = metric.replace("grade_macro_", "").replace(
            "continuous_macro_", ""
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Model comparison on the same subject-level test split", y=1.03
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_target_heatmap(run_dirs: list[Path], output: Path) -> None:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "test_metrics.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["kind"] == "grade"].copy()
        frame["run"] = run_dir.name
        frame["target"] = frame["target"].map(short_target)
        frames.append(frame)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    values = combined.pivot(index="target", columns="run", values="qwk")
    figure, axis = plt.subplots(figsize=(max(7, len(values.columns) * 1.5), 7))
    sns.heatmap(
        values,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=axis,
    )
    axis.set_title("Per-target quadratic weighted kappa")
    axis.set_xlabel("Experiment")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_histories(run_dirs: list[Path], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    found = False
    for run_dir in run_dirs:
        path = run_dir / "history.csv"
        if not path.is_file():
            continue
        history = pd.read_csv(path)
        found = True
        axes[0].plot(
            history["epoch"],
            history["val_loss"],
            marker="o",
            label=run_dir.name,
        )
        qwk_column = "val_grade_macro_qwk"
        if qwk_column in history:
            axes[1].plot(
                history["epoch"],
                history[qwk_column],
                marker="o",
                label=run_dir.name,
            )
    if not found:
        plt.close(figure)
        return
    axes[0].set_title("Validation loss")
    axes[1].set_title("Validation macro QWK")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_pipeline(output: Path) -> None:
    stages = [
        "Smartphone\nF / L / R",
        "Face landmarks\n& quality gate",
        "Region crops\n& view routing",
        "Shared CNN\nfeature encoder",
        "Mean / Concat\n/ Attention",
        "Softmax / Ordinal\n+ moisture regression",
        "Skin report\n& recommendation",
    ]
    figure, axis = plt.subplots(figsize=(18, 3.2))
    axis.set_xlim(0, len(stages))
    axis.set_ylim(0, 1)
    axis.axis("off")
    colors = [
        "#DCEAF7",
        "#DCEAF7",
        "#FDE9D9",
        "#E2F0D9",
        "#E2F0D9",
        "#FFF2CC",
        "#E4DFEC",
    ]
    for index, (stage, color) in enumerate(zip(stages, colors)):
        axis.text(
            index + 0.5,
            0.5,
            stage,
            ha="center",
            va="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.6",
                "facecolor": color,
                "edgecolor": "#444",
            },
        )
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(index + 1.08, 0.5),
                xytext=(index + 0.90, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 1.5},
            )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_data_eda(
    subject_manifest: Path | None,
    region_manifest: Path | None,
    figure_dir: Path,
) -> None:
    if subject_manifest and subject_manifest.is_file():
        subjects = pd.read_csv(subject_manifest)
        grade_columns = [
            column
            for column in subjects
            if column.startswith("ann__") and subjects[column].notna().any()
        ]
        long = subjects.melt(
            value_vars=grade_columns,
            var_name="target",
            value_name="grade",
        )
        long["target"] = long["target"].map(short_target)
        grid = sns.catplot(
            data=long,
            x="grade",
            col="target",
            col_wrap=3,
            kind="count",
            height=2.7,
            color="#4C78A8",
            sharex=False,
            sharey=False,
        )
        grid.set_titles("{col_name}")
        grid.figure.suptitle("Expert-grade label distributions", y=1.02)
        grid.figure.savefig(
            figure_dir / "05_grade_distributions.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(grid.figure)

    if region_manifest and region_manifest.is_file():
        regions = pd.read_csv(region_manifest)
        if "region" not in regions:
            mapping = {
                0: "full_face",
                1: "forehead",
                2: "glabella",
                3: "right_eye",
                4: "left_eye",
                5: "right_cheek",
                6: "left_cheek",
                7: "lips",
                8: "chin",
            }
            regions["region"] = regions["facepart"].map(mapping)
        missing_values = regions["bbox"].astype(str).str.lower().isin(
            ["nan", "none", ""]
        )
        regions["available"] = regions["bbox"].notna() & ~missing_values
        table = regions.pivot_table(
            index="region",
            columns="view",
            values="available",
            aggfunc="mean",
        )
        figure, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            table,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=axis,
        )
        axis.set_title("BBox availability by region and view")
        figure.tight_layout()
        figure.savefig(
            figure_dir / "06_bbox_availability.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)


def best_run_directory(
    table: pd.DataFrame,
    run_dirs: list[Path],
) -> Path:
    candidates = table.dropna(subset=["grade_macro_qwk"])
    if candidates.empty:
        return run_dirs[0]
    best_name = str(
        candidates.sort_values("grade_macro_qwk", ascending=False)
        .iloc[0]["run"]
    )
    return next(path for path in run_dirs if path.name == best_name)


def plot_grade_confusions(run_dir: Path, output: Path) -> None:
    metrics_path = run_dir / "test_metrics.csv"
    predictions_path = run_dir / "test_predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        return
    metrics = pd.read_csv(metrics_path)
    selected = (
        metrics[metrics["kind"] == "grade"]
        .sort_values("qwk", ascending=True)
        .head(4)
    )
    predictions = pd.read_csv(predictions_path)
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (_, row) in zip(axes.ravel(), selected.iterrows()):
        target = str(row["target"])
        truth = np.rint(predictions[f"true__{target}"]).astype(int)
        predicted = np.rint(predictions[f"pred__{target}"]).astype(int)
        labels = np.arange(
            min(truth.min(), predicted.min()),
            max(truth.max(), predicted.max()) + 1,
        )
        matrix = confusion_matrix(truth, predicted, labels=labels)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=axis,
        )
        axis.set_title(f"{short_target(target)} (QWK={row['qwk']:.2f})")
        axis.set_xlabel("Predicted grade")
        axis.set_ylabel("True grade")
    for axis in axes.ravel()[len(selected):]:
        axis.axis("off")
    figure.suptitle(f"Lowest-QWK targets in {run_dir.name}", y=1.01)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_moisture_predictions(run_dir: Path, output: Path) -> None:
    metrics_path = run_dir / "test_metrics.csv"
    predictions_path = run_dir / "test_predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        return
    metrics = pd.read_csv(metrics_path)
    selected = metrics[metrics["kind"] == "continuous"]
    if selected.empty:
        return
    predictions = pd.read_csv(predictions_path)
    columns = 2
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.5 * columns, 4.5 * rows),
        squeeze=False,
    )
    for axis, (_, row) in zip(axes.ravel(), selected.iterrows()):
        target = str(row["target"])
        truth = predictions[f"true__{target}"].to_numpy()
        predicted = predictions[f"pred__{target}"].to_numpy()
        lower = float(min(truth.min(), predicted.min()))
        upper = float(max(truth.max(), predicted.max()))
        axis.scatter(truth, predicted, s=25, alpha=0.65, color="#4C78A8")
        axis.plot([lower, upper], [lower, upper], "--", color="#E45756")
        axis.set_title(
            f"{short_target(target)} (MAE={row['mae']:.2f})"
        )
        axis.set_xlabel("Measured")
        axis.set_ylabel("Predicted")
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(selected):]:
        axis.axis("off")
    figure.suptitle(f"Moisture predictions in {run_dir.name}", y=1.01)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create lab-meeting figures from experiment outputs."
    )
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--subject-manifest", type=Path)
    parser.add_argument("--region-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = discover_runs(args.outputs_dir, args.runs)
    if not run_dirs:
        raise SystemExit("No experiment folders with summary.json were found")
    table = load_run_table(run_dirs)
    table.to_csv(args.figure_dir / "experiment_summary.csv", index=False)
    plot_run_comparison(table, args.figure_dir / "01_model_comparison.png")
    plot_target_heatmap(run_dirs, args.figure_dir / "02_target_qwk_heatmap.png")
    plot_histories(run_dirs, args.figure_dir / "03_training_curves.png")
    plot_pipeline(args.figure_dir / "04_service_pipeline.png")
    plot_data_eda(args.subject_manifest, args.region_manifest, args.figure_dir)
    best_run = best_run_directory(table, run_dirs)
    plot_grade_confusions(
        best_run,
        args.figure_dir / "07_grade_confusion_matrices.png",
    )
    plot_moisture_predictions(
        best_run,
        args.figure_dir / "08_moisture_predictions.png",
    )
    print(f"Runs: {[path.name for path in run_dirs]}")
    print(f"Detailed prediction figures use: {best_run.name}")
    print(f"Saved figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()

