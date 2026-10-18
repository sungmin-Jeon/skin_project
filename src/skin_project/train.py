from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader
from torchvision import transforms

from .data import MultiRegionDataset, load_manifests, prepare_crop_cache
from .losses import GradeFrequencyWeights
from .model import MultiRegionMultiViewModel
from .tasks import TargetSpec, get_task_specs, required_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train F/L/R multi-view regional skin assessment models."
    )
    parser.add_argument(
        "--task",
        choices=("forehead_1", "grades_11", "core_15"),
        required=True,
    )
    parser.add_argument("--subject-manifest", type=Path, required=True)
    parser.add_argument("--region-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--run-name",
        help="Output subdirectory name; defaults to the task name.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--margin-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grade-balance-power",
        type=float,
        default=0.5,
        help="Inverse-frequency strength for grade loss; use 0 to disable.",
    )
    parser.add_argument(
        "--max-grade-weight",
        type=float,
        default=3.0,
        help="Maximum per-example grade loss weight.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TargetNormalizer:
    def __init__(self, specs: tuple[TargetSpec, ...], train_df: pd.DataFrame):
        means: list[float] = []
        scales: list[float] = []
        for spec in specs:
            if spec.kind == "continuous":
                values = pd.to_numeric(train_df[spec.column], errors="raise")
                mean = float(values.mean())
                scale = float(values.std(ddof=0))
                if not np.isfinite(scale) or scale < 1e-8:
                    scale = 1.0
            else:
                mean, scale = 0.0, 1.0
            means.append(mean)
            scales.append(scale)
        self.means = np.asarray(means, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        means = torch.as_tensor(self.means, device=values.device)
        scales = torch.as_tensor(self.scales, device=values.device)
        return (values - means) / scales

    def denormalize_array(self, values: np.ndarray) -> np.ndarray:
        return values * self.scales + self.means

    def state_dict(self) -> dict[str, list[float]]:
        return {"means": self.means.tolist(), "scales": self.scales.tolist()}


def build_transforms():
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, scale=(0.85, 1.0), ratio=(0.90, 1.10)
            ),
            transforms.ColorJitter(
                brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02
            ),
            transforms.RandomRotation(4),
            transforms.ToTensor(),
            normalization,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalization]
    )
    return train_transform, eval_transform


def create_loaders(
    subjects: pd.DataFrame,
    cache_dir: Path,
    regions: tuple[str, ...],
    specs: tuple[TargetSpec, ...],
    available_crops: set[tuple[str, str, str, str]],
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    train_transform, eval_transform = build_transforms()
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        frame = subjects[subjects["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"No rows found for split={split!r}")
        dataset = MultiRegionDataset(
            frame,
            cache_root=cache_dir,
            regions=regions,
            target_specs=specs,
            available_crops=available_crops,
            transform=train_transform if split == "train" else eval_transform,
        )
        loader_options = {
            "batch_size": batch_size,
            "shuffle": split == "train",
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_options["prefetch_factor"] = 2
        loaders[split] = DataLoader(dataset, **loader_options)
    return loaders


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: TargetNormalizer,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    grade_weights: GradeFrequencyWeights | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    raw_targets: list[np.ndarray] = []
    normalized_predictions: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        normalized_targets = normalizer.normalize_tensor(targets)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                predictions = model(images, view_mask)
                element_loss = F.huber_loss(
                    predictions, normalized_targets, delta=1.0, reduction="none"
                )
                if grade_weights is None:
                    per_target_loss = element_loss.mean(dim=0)
                else:
                    sample_weights = grade_weights.for_targets(targets)
                    per_target_loss = (
                        (element_loss * sample_weights).sum(dim=0)
                        / sample_weights.sum(dim=0).clamp_min(1e-8)
                    )
                loss = per_target_loss.mean()
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * len(targets)
        raw_targets.append(targets.detach().cpu().numpy())
        normalized_predictions.append(predictions.detach().cpu().numpy())

    targets_array = np.concatenate(raw_targets, axis=0)
    predictions_array = normalizer.denormalize_array(
        np.concatenate(normalized_predictions, axis=0)
    )
    return total_loss / len(loader.dataset), targets_array, predictions_array


def target_metrics(
    specs: tuple[TargetSpec, ...],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    grade_maes: list[float] = []
    grade_exact_accuracies: list[float] = []
    grade_balanced_accuracies: list[float] = []
    grade_within_one: list[float] = []
    grade_qwks: list[float] = []
    continuous_nmaes: list[float] = []
    for index, spec in enumerate(specs):
        truth = targets[:, index]
        predicted = predictions[:, index]
        if spec.minimum is not None and spec.maximum is not None:
            predicted = np.clip(predicted, spec.minimum, spec.maximum)
        mae = float(np.mean(np.abs(truth - predicted)))
        row: dict[str, object] = {
            "target": spec.column,
            "region": spec.region,
            "kind": spec.kind,
            "mae": mae,
        }
        if spec.kind == "grade":
            truth_grade = np.rint(truth).astype(int)
            predicted_grade = np.rint(predicted).astype(int)
            row["exact_accuracy"] = float(np.mean(truth_grade == predicted_grade))
            row["balanced_accuracy"] = float(
                balanced_accuracy_score(truth_grade, predicted_grade)
            )
            row["within_one_grade"] = float(
                np.mean(np.abs(truth_grade - predicted_grade) <= 1)
            )
            row["qwk"] = float(
                cohen_kappa_score(truth_grade, predicted_grade, weights="quadratic")
            )
            grade_maes.append(mae)
            grade_exact_accuracies.append(float(row["exact_accuracy"]))
            grade_balanced_accuracies.append(float(row["balanced_accuracy"]))
            grade_within_one.append(float(row["within_one_grade"]))
            grade_qwks.append(float(row["qwk"]))
        else:
            scale = float(np.std(truth))
            row["normalized_mae"] = mae / scale if scale > 1e-8 else np.nan
            continuous_nmaes.append(float(row["normalized_mae"]))
        rows.append(row)

    summary: dict[str, float] = {}
    if grade_maes:
        summary["grade_macro_mae"] = float(np.mean(grade_maes))
        summary["grade_macro_exact_accuracy"] = float(
            np.mean(grade_exact_accuracies)
        )
        summary["grade_macro_balanced_accuracy"] = float(
            np.mean(grade_balanced_accuracies)
        )
        summary["grade_macro_within_one"] = float(np.mean(grade_within_one))
        summary["grade_macro_qwk"] = float(np.mean(grade_qwks))
    if continuous_nmaes:
        summary["continuous_macro_normalized_mae"] = float(
            np.nanmean(continuous_nmaes)
        )
    return rows, summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    specs = get_task_specs(args.task)
    regions = required_regions(specs)
    subjects, region_manifest = load_manifests(
        args.subject_manifest, args.region_manifest
    )
    missing_targets = [spec.column for spec in specs if spec.column not in subjects]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    if subjects[[spec.column for spec in specs]].isna().any().any():
        raise ValueError("Selected task contains missing target values")

    print(f"Task: {args.task}")
    print(f"Targets: {len(specs)} | Regions: {len(regions)} {regions}")
    print("Split counts:")
    print(subjects["split"].value_counts().to_string())
    available_crops = prepare_crop_cache(
        subjects,
        region_manifest,
        data_root=args.data_root,
        cache_root=args.cache_dir,
        required_regions=regions,
        margin_ratio=args.margin_ratio,
        overwrite=args.overwrite_cache,
    )
    if args.cache_only:
        print("Crop cache is ready; exiting because --cache-only was set.")
        return

    batch_size = args.batch_size or (16 if args.task == "forehead_1" else 4)
    loaders = create_loaders(
        subjects,
        args.cache_dir,
        regions,
        specs,
        available_crops,
        batch_size,
        args.num_workers,
    )
    train_df = subjects[subjects["split"] == "train"]
    normalizer = TargetNormalizer(specs, train_df)
    grade_weights = GradeFrequencyWeights(
        specs,
        train_df,
        power=args.grade_balance_power,
        max_weight=args.max_grade_weight,
    )
    print(
        "Grade balancing: "
        f"power={args.grade_balance_power:g}, "
        f"max_weight={args.max_grade_weight:g}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA was not detected. Multi-region training will be slow.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = MultiRegionMultiViewModel(
        regions, specs, pretrained=not args.no_pretrained
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_dir = args.output_dir / (args.run_name or args.task)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_targets, train_predictions = run_epoch(
            model,
            loaders["train"],
            normalizer,
            device,
            optimizer,
            scaler,
            grade_weights,
        )
        val_loss, val_targets, val_predictions = run_epoch(
            model,
            loaders["val"],
            normalizer,
            device,
            None,
            scaler,
            grade_weights,
        )
        _, train_summary = target_metrics(specs, train_targets, train_predictions)
        _, val_summary = target_metrics(specs, val_targets, val_predictions)
        scheduler.step()

        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        record.update({f"train_{k}": v for k, v in train_summary.items()})
        record.update({f"val_{k}": v for k, v in val_summary.items()})
        history.append(record)
        metrics_text = " ".join(
            f"{key}={value:.3f}" for key, value in val_summary.items()
        )
        print(
            f"[{epoch:02d}/{args.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} {metrics_text}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "task": args.task,
                    "regions": regions,
                    "target_specs": [asdict(spec) for spec in specs],
                    "normalizer": normalizer.state_dict(),
                    "grade_weights": grade_weights.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            print(f"  -> saved {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping")
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_targets, test_predictions = run_epoch(
        model, loaders["test"], normalizer, device, None, scaler
    )
    metric_rows, test_summary = target_metrics(specs, test_targets, test_predictions)
    pd.DataFrame(metric_rows).to_csv(run_dir / "test_metrics.csv", index=False)

    prediction_frame = subjects[subjects["split"] == "test"][["subject_id"]].copy()
    for index, spec in enumerate(specs):
        prediction_frame[f"true__{spec.column}"] = test_targets[:, index]
        prediction_frame[f"pred__{spec.column}"] = test_predictions[:, index]
    prediction_frame.to_csv(run_dir / "test_predictions.csv", index=False)

    result = {
        "task": args.task,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "test_loss": test_loss,
        **test_summary,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Test summary:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
