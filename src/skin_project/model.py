from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .tasks import TargetSpec


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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # [batch, region, view, channel, height, width]
        batch, region_count, view_count, channels, height, width = images.shape
        flattened = images.reshape(
            batch * region_count * view_count, channels, height, width
        )
        features = self.encoder(flattened)
        features = features.reshape(batch, region_count, view_count, -1)
        fused = features.mean(dim=2)

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
