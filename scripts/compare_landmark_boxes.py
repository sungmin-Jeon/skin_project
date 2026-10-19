from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from skin_project.bbox import has_bbox, parse_bbox
from skin_project.tasks import FACEPART_TO_REGION
from skin_project.views import VIEW_POLICIES, views_for_region


KEYS = ["source_split", "subject_id", "view", "facepart"]


def intersection_over_union(first: object, second: object) -> float:
    if not has_bbox(first) or not has_bbox(second):
        return float("nan")
    ax1, ay1, ax2, ay2 = parse_bbox(first)
    bx1, by1, bx2, by2 = parse_bbox(second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else float("nan")


def ground_truth_coverage(ground_truth: object, landmark: object) -> float:
    """Fraction of the dataset box covered by the landmark-generated crop."""
    if not has_bbox(ground_truth) or not has_bbox(landmark):
        return float("nan")
    gx1, gy1, gx2, gy2 = parse_bbox(ground_truth)
    lx1, ly1, lx2, ly2 = parse_bbox(landmark)
    intersection_width = max(0, min(gx2, lx2) - max(gx1, lx1))
    intersection_height = max(0, min(gy2, ly2) - max(gy1, ly1))
    ground_truth_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    if ground_truth_area == 0:
        return float("nan")
    return intersection_width * intersection_height / ground_truth_area


def ground_truth_center_hit(ground_truth: object, landmark: object) -> float:
    if not has_bbox(ground_truth) or not has_bbox(landmark):
        return float("nan")
    gx1, gy1, gx2, gy2 = parse_bbox(ground_truth)
    lx1, ly1, lx2, ly2 = parse_bbox(landmark)
    center_x = (gx1 + gx2) / 2
    center_y = (gy1 + gy2) / 2
    return float(lx1 <= center_x <= lx2 and ly1 <= center_y <= ly2)


def load_comparison(ground_truth_path: Path, landmark_path: Path) -> pd.DataFrame:
    dtype = {"subject_id": "string", "device_id": "string"}
    ground_truth = pd.read_csv(ground_truth_path, dtype=dtype)
    landmarks = pd.read_csv(landmark_path, dtype=dtype)
    for frame in (ground_truth, landmarks):
        frame["subject_id"] = frame["subject_id"].str.zfill(4)
        frame["facepart"] = frame["facepart"].astype(int)
        frame["view"] = frame["view"].str.upper()
    merged = ground_truth[KEYS + ["bbox"]].merge(
        landmarks[KEYS + ["bbox"]],
        on=KEYS,
        how="left",
        suffixes=("_ground_truth", "_landmark"),
    )
    merged["region"] = merged["facepart"].map(FACEPART_TO_REGION)
    merged["iou"] = [
        intersection_over_union(first, second)
        for first, second in zip(
            merged["bbox_ground_truth"], merged["bbox_landmark"]
        )
    ]
    merged["ground_truth_coverage"] = [
        ground_truth_coverage(first, second)
        for first, second in zip(
            merged["bbox_ground_truth"], merged["bbox_landmark"]
        )
    ]
    merged["ground_truth_center_hit"] = [
        ground_truth_center_hit(first, second)
        for first, second in zip(
            merged["bbox_ground_truth"], merged["bbox_landmark"]
        )
    ]
    return merged


def filter_view_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    keep = [
        row.view in views_for_region(policy, row.region)
        for row in frame.itertuples(index=False)
    ]
    return frame.loc[keep].reset_index(drop=True)


def add_image_paths(frame: pd.DataFrame, subject_manifest: Path) -> pd.DataFrame:
    subjects = pd.read_csv(
        subject_manifest,
        dtype={"subject_id": "string", "device_id": "string"},
    )
    subjects["subject_id"] = subjects["subject_id"].str.zfill(4)
    paths = subjects.melt(
        id_vars=["source_split", "subject_id"],
        value_vars=["image_F", "image_L", "image_R"],
        var_name="view",
        value_name="image_path",
    )
    paths["view"] = paths["view"].str.replace(
        r"^image_", "", regex=True
    )
    return frame.merge(paths, on=["source_split", "subject_id", "view"], how="left")


def draw_worst_cases(
    frame: pd.DataFrame,
    data_root: Path,
    output: Path,
    count: int,
) -> None:
    examples = frame.dropna(
        subset=[
            "iou",
            "ground_truth_coverage",
            "ground_truth_center_hit",
            "image_path",
        ]
    ).nsmallest(
        count,
        ["ground_truth_center_hit", "ground_truth_coverage", "iou"],
    )
    columns = 4
    rows = max(1, (len(examples) + columns - 1) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 5 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for axis, (_, row) in zip(axes, examples.iterrows()):
        with Image.open(data_root / row["image_path"]) as source:
            image = source.convert("RGB")
        axis.imshow(image)
        for bbox, color, label in (
            (row["bbox_ground_truth"], "#00BFFF", "ground truth"),
            (row["bbox_landmark"], "#FF3B30", "landmark"),
        ):
            x1, y1, x2, y2 = parse_bbox(bbox)
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                    label=label,
                )
            )
        axis.set_title(
            f"{row['subject_id']} {row['view']} {row['region']}\n"
            f"IoU={row['iou']:.3f}, GT coverage="
            f"{row['ground_truth_coverage']:.3f}"
        )
        axis.axis("off")
    for axis in axes[len(examples):]:
        axis.axis("off")
    if len(examples):
        axes[0].legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare landmark-generated boxes with dataset boxes."
    )
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--landmark-manifest", type=Path, required=True)
    parser.add_argument("--subject-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worst-count", type=int, default=12)
    parser.add_argument(
        "--view-policy",
        choices=tuple(VIEW_POLICIES),
        default="service",
        help="Only evaluate region/view pairs used by this policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = load_comparison(
        args.ground_truth_manifest, args.landmark_manifest
    )
    comparison = filter_view_policy(comparison, args.view_policy)
    comparison = add_image_paths(comparison, args.subject_manifest)
    summary = (
        comparison.groupby("region")
        .agg(
            count=("iou", "count"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            mean_ground_truth_coverage=("ground_truth_coverage", "mean"),
            center_hit_rate=("ground_truth_center_hit", "mean"),
        )
        .sort_values("mean_ground_truth_coverage")
    )
    comparison.to_csv(args.output_dir / "landmark_iou_rows.csv", index=False)
    summary.to_csv(args.output_dir / "landmark_iou_summary.csv")

    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    comparison.boxplot(column="iou", by="region", ax=axes[0], grid=False)
    comparison.boxplot(
        column="ground_truth_coverage",
        by="region",
        ax=axes[1],
        grid=False,
    )
    axes[0].set_title("IoU by region")
    axes[0].set_ylabel("IoU")
    axes[1].set_title("Dataset-box coverage by landmark crop")
    axes[1].set_ylabel("Coverage")
    for axis in axes:
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=30)
    figure.suptitle("")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "landmark_box_metrics.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    draw_worst_cases(
        comparison,
        args.data_root,
        args.output_dir / "landmark_worst_cases.png",
        args.worst_count,
    )
    print(summary.round(3).to_string())
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

