from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .tasks import TargetSpec


class GradeFrequencyWeights:
    """Per-target inverse-frequency weights for ordinal grade regression."""

    def __init__(
        self,
        specs: tuple[TargetSpec, ...],
        train_df: pd.DataFrame,
        power: float = 0.5,
        max_weight: float = 3.0,
    ) -> None:
        if power < 0:
            raise ValueError("grade balance power must be non-negative")
        if max_weight < 1:
            raise ValueError("maximum grade weight must be at least 1")

        self.specs = specs
        self.power = power
        self.max_weight = max_weight
        self.tables: dict[int, torch.Tensor] = {}
        self.minimums: dict[int, int] = {}

        for target_index, spec in enumerate(specs):
            if spec.kind != "grade":
                continue
            if spec.minimum is None or spec.maximum is None:
                raise ValueError(f"Grade target has no bounds: {spec.column}")

            minimum = int(spec.minimum)
            maximum = int(spec.maximum)
            grades = np.rint(
                pd.to_numeric(train_df[spec.column], errors="raise").to_numpy()
            ).astype(int)
            if np.any((grades < minimum) | (grades > maximum)):
                raise ValueError(f"Grade outside configured bounds: {spec.column}")

            counts = np.bincount(
                grades - minimum,
                minlength=maximum - minimum + 1,
            ).astype(np.float64)
            present = counts > 0
            weights = np.full_like(counts, max_weight, dtype=np.float64)

            if power == 0:
                weights[present] = 1.0
            else:
                raw = np.power(len(grades) / counts[present], power)
                # Keep the average weight of observed training samples near one.
                average = float(np.sum(counts[present] * raw) / len(grades))
                weights[present] = raw / average
                weights = np.clip(weights, 1.0 / max_weight, max_weight)

            self.tables[target_index] = torch.tensor(
                weights, dtype=torch.float32
            )
            self.minimums[target_index] = minimum

    def for_targets(self, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(targets)
        for target_index, table in self.tables.items():
            minimum = self.minimums[target_index]
            grade_indices = (
                torch.round(targets[:, target_index]).long() - minimum
            ).clamp(0, len(table) - 1)
            weights[:, target_index] = table.to(targets.device)[grade_indices]
        return weights

    def state_dict(self) -> dict[str, object]:
        return {
            "power": self.power,
            "max_weight": self.max_weight,
            "weights": {
                self.specs[index].column: table.tolist()
                for index, table in self.tables.items()
            },
        }


class ClassificationWeights:
    """Class-balanced weights and ordinal threshold weights from train labels."""

    def __init__(
        self,
        specs: tuple[TargetSpec, ...],
        train_df: pd.DataFrame,
        beta: float = 0.999,
        max_weight: float = 5.0,
    ) -> None:
        if not 0 <= beta < 1:
            raise ValueError("class balance beta must be in [0, 1)")
        if max_weight < 1:
            raise ValueError("maximum class weight must be at least 1")
        self.specs = specs
        self.beta = beta
        self.max_weight = max_weight
        self.class_tables: dict[str, torch.Tensor] = {}
        self.ordinal_positive_tables: dict[str, torch.Tensor] = {}

        for spec in specs:
            if spec.kind != "grade":
                continue
            if spec.minimum is None or spec.maximum is None:
                raise ValueError(f"Grade target has no bounds: {spec.column}")
            minimum = int(spec.minimum)
            maximum = int(spec.maximum)
            labels = np.rint(
                pd.to_numeric(train_df[spec.column], errors="raise").to_numpy()
            ).astype(int) - minimum
            class_count = maximum - minimum + 1
            counts = np.bincount(labels, minlength=class_count).astype(np.float64)

            effective = np.zeros_like(counts)
            present = counts > 0
            if beta == 0:
                effective[present] = 1.0
            else:
                effective[present] = (1.0 - beta) / (
                    1.0 - np.power(beta, counts[present])
                )
            if present.any():
                effective[present] /= effective[present].mean()
            effective[~present] = max_weight
            effective = np.clip(effective, 1.0 / max_weight, max_weight)
            self.class_tables[spec.column] = torch.tensor(
                effective, dtype=torch.float32
            )

            threshold_weights: list[float] = []
            for threshold in range(class_count - 1):
                positive = float(np.sum(labels > threshold))
                negative = float(len(labels) - positive)
                ratio = negative / max(positive, 1.0)
                threshold_weights.append(
                    min(max(ratio, 1.0 / max_weight), max_weight)
                )
            self.ordinal_positive_tables[spec.column] = torch.tensor(
                threshold_weights, dtype=torch.float32
            )

    def class_weights(self, column: str, device: torch.device) -> torch.Tensor:
        return self.class_tables[column].to(device)

    def ordinal_positive_weights(
        self, column: str, device: torch.device
    ) -> torch.Tensor:
        return self.ordinal_positive_tables[column].to(device)

    def state_dict(self) -> dict[str, object]:
        return {
            "beta": self.beta,
            "max_weight": self.max_weight,
            "class_weights": {
                key: value.tolist() for key, value in self.class_tables.items()
            },
            "ordinal_positive_weights": {
                key: value.tolist()
                for key, value in self.ordinal_positive_tables.items()
            },
        }

