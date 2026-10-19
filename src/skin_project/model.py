from __future__ import annotations

from collections import defaultdict
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .tasks import TargetSpec


FusionKind = Literal["mean", "concat", "attention"]
GradeObjective = Literal["regression", "softmax", "ordinal"]


class OrdinalHead(nn.Module):
    """Cumulative ordinal head with monotonically ordered cut points."""

    def __init__(self, input_dim: int, class_count: int) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("Ordinal targets need at least two classes")
        self.score = nn.Linear(input_dim, 1)
        self.base_cutpoint = nn.Parameter(torch.tensor(-1.0))
        self.cutpoint_steps = nn.Parameter(
            torch.zeros(class_count - 2, dtype=torch.float32)
        )

    def cutpoints(self) -> torch.Tensor:
        if self.cutpoint_steps.numel() == 0:
            return self.base_cutpoint.unsqueeze(0)
        steps = F.softplus(self.cutpoint_steps)
        return torch.cat(
            [
                self.base_cutpoint.unsqueeze(0),
                self.base_cutpoint + torch.cumsum(steps, dim=0),
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features) - self.cutpoints().unsqueeze(0)


class ConfigurableSkinModel(nn.Module):
    """Shared encoder with configurable view fusion and target objectives."""

    def __init__(
        self,
        regions: tuple[str, ...],
        target_specs: tuple[TargetSpec, ...],
        grade_objective: GradeObjective = "regression",
        fusion: FusionKind = "mean",
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.regions = regions
        self.target_specs = target_specs
        self.grade_objective = grade_objective
        self.fusion = fusion

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.encoder = efficientnet_b0(weights=weights)
        feature_dim = self.encoder.classifier[1].in_features
        self.feature_dim = feature_dim
        self.encoder.classifier = nn.Identity()

        self.attention = nn.ModuleDict()
        if fusion == "attention":
            self.view_embeddings = nn.Parameter(
                torch.zeros(len(regions), 3, feature_dim)
            )
            nn.init.normal_(self.view_embeddings, std=0.02)
            for region in regions:
                self.attention[region] = nn.Linear(feature_dim, 1)
            fused_dim = feature_dim
        elif fusion == "concat":
            fused_dim = feature_dim * 3 + 3
        elif fusion == "mean":
            fused_dim = feature_dim
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        self.target_heads = nn.ModuleDict()
        for spec in target_specs:
            hidden = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if spec.kind == "continuous" or grade_objective == "regression":
                output: nn.Module = nn.Linear(256, 1)
            else:
                if spec.minimum is None or spec.maximum is None:
                    raise ValueError(f"Grade target has no bounds: {spec.column}")
                class_count = int(spec.maximum - spec.minimum + 1)
                output = (
                    nn.Linear(256, class_count)
                    if grade_objective == "softmax"
                    else OrdinalHead(256, class_count)
                )
            self.target_heads[spec.column] = nn.Sequential(hidden, output)

    def _fuse(
        self,
        features: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = view_mask.to(dtype=features.dtype).unsqueeze(-1)
        if self.fusion == "mean":
            return (features * weights).sum(dim=2) / weights.sum(
                dim=2
            ).clamp_min(1.0)
        if self.fusion == "concat":
            masked = (features * weights).flatten(start_dim=2)
            return torch.cat([masked, view_mask.to(features.dtype)], dim=-1)

        attended_regions: list[torch.Tensor] = []
        for region_index, region in enumerate(self.regions):
            region_features = features[:, region_index] + self.view_embeddings[region_index]
            scores = self.attention[region](region_features).squeeze(-1)
            scores = scores.masked_fill(view_mask[:, region_index] == 0, -1e4)
            attention_weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            attended_regions.append(
                (features[:, region_index] * attention_weights).sum(dim=1)
            )
        return torch.stack(attended_regions, dim=1)

    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, region_count, view_count, channels, height, width = images.shape
        flattened = images.reshape(
            batch * region_count * view_count, channels, height, width
        )
        flat_mask = view_mask.reshape(-1) > 0
        if not flat_mask.any():
            raise RuntimeError("The batch contains no usable region views")
        encoded = self.encoder(flattened[flat_mask])
        all_features = encoded.new_zeros(
            batch * region_count * view_count, self.feature_dim
        )
        all_features[flat_mask] = encoded
        features = all_features.reshape(
            batch, region_count, view_count, self.feature_dim
        )
        fused = self._fuse(features, view_mask)
        region_indices = {region: index for index, region in enumerate(self.regions)}
        return {
            spec.column: self.target_heads[spec.column](
                fused[:, region_indices[spec.region]]
            )
            for spec in self.target_specs
        }


class MultiRegionMultiViewModel(nn.Module):
    """Shared image encoder with mean view fusion and region-specific heads."""

    def __init__(
        self,
        regions: tuple[str, ...],
        target_specs: tuple[TargetSpec, ...],
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.regions = regions
        self.target_specs = target_specs

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.encoder = efficientnet_b0(weights=weights)
        feature_dim = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()

        region_targets: dict[str, list[int]] = defaultdict(list)
        for target_index, spec in enumerate(target_specs):
            region_targets[spec.region].append(target_index)
        self.region_target_indices = dict(region_targets)

        self.heads = nn.ModuleDict()
        for region in regions:
            output_count = len(self.region_target_indices[region])
            self.heads[region] = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, output_count),
            )

    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [batch, region, view, channel, height, width]
        batch, region_count, view_count, channels, height, width = images.shape
        flattened = images.reshape(
            batch * region_count * view_count, channels, height, width
        )
        features = self.encoder(flattened)
        features = features.reshape(batch, region_count, view_count, -1)
        if view_mask is None:
            fused = features.mean(dim=2)
        else:
            weights = view_mask.to(dtype=features.dtype).unsqueeze(-1)
            denominator = weights.sum(dim=2).clamp_min(1.0)
            fused = (features * weights).sum(dim=2) / denominator

        columns: list[torch.Tensor | None] = [None] * len(self.target_specs)
        for region_index, region in enumerate(self.regions):
            region_output = self.heads[region](fused[:, region_index])
            for local_index, target_index in enumerate(
                self.region_target_indices[region]
            ):
                columns[target_index] = region_output[:, local_index]

        if any(column is None for column in columns):
            raise RuntimeError("One or more target outputs were not produced")
        return torch.stack(columns, dim=1)  # type: ignore[arg-type]
