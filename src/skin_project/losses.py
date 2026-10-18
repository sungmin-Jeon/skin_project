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

