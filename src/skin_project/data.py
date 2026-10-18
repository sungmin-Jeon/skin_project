from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .tasks import FACEPART_TO_REGION, TargetSpec


VIEWS = ("F", "L", "R")


def load_manifests(
    subject_manifest_path: str | Path,
    region_manifest_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dtype = {"subject_id": "string", "device_id": "string"}
    subjects = pd.read_csv(subject_manifest_path, dtype=dtype)
    regions = pd.read_csv(region_manifest_path, dtype=dtype)

    required_subject_columns = {
        "source_split",
        "subject_id",
        "split",
        "image_F",
        "image_L",
        "image_R",
    }
    required_region_columns = {
        "source_split",
        "subject_id",
        "view",
        "facepart",
        "bbox",
    }
    missing_subject = required_subject_columns - set(subjects.columns)
    missing_region = required_region_columns - set(regions.columns)
    if missing_subject:
        raise ValueError(f"Subject manifest is missing columns: {sorted(missing_subject)}")
    if missing_region:
        raise ValueError(f"Region manifest is missing columns: {sorted(missing_region)}")

    subjects["subject_id"] = subjects["subject_id"].str.zfill(4)
    regions["subject_id"] = regions["subject_id"].str.zfill(4)
    regions["view"] = regions["view"].str.upper()
    regions["facepart"] = regions["facepart"].astype(int)
    regions["region"] = regions["facepart"].map(FACEPART_TO_REGION)
    return subjects, regions


def parse_bbox(value: object) -> tuple[int, int, int, int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Invalid bbox: {value!r}")
    x1, y1, x2, y2 = (int(round(float(number))) for number in value)
    return x1, y1, x2, y2


def crop_with_margin(
    image: Image.Image,
    bbox: object,
    margin_ratio: float,
) -> Image.Image:
    x1, y1, x2, y2 = parse_bbox(bbox)
    width, height = x2 - x1, y2 - y1
    margin_x = round(width * margin_ratio)
    margin_y = round(height * margin_ratio)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.width, x2 + margin_x)
    y2 = min(image.height, y2 + margin_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty crop after clamping: {(x1, y1, x2, y2)}")
    return image.crop((x1, y1, x2, y2))


def cache_path_for(
    cache_root: Path,
    source_split: str,
    subject_id: str,
    region: str,
    view: str,
) -> Path:
    return cache_root / source_split / subject_id / f"{region}_{view}.jpg"


def prepare_crop_cache(
    subjects: pd.DataFrame,
    regions: pd.DataFrame,
    data_root: str | Path,
    cache_root: str | Path,
    required_regions: Iterable[str],
    margin_ratio: float = 0.10,
    overwrite: bool = False,
) -> None:
    data_root = Path(data_root)
    cache_root = Path(cache_root)
    required_regions = tuple(required_regions)

    selected = regions[regions["region"].isin(required_regions)].copy()
    key_columns = ["source_split", "subject_id", "view", "region"]
    duplicates = selected.duplicated(key_columns, keep=False)
    if duplicates.any():
        example = selected.loc[duplicates, key_columns].head().to_dict("records")
        raise ValueError(f"Duplicate region rows found: {example}")
    bbox_lookup = selected.set_index(key_columns)["bbox"].to_dict()

    expected = len(subjects) * len(required_regions) * len(VIEWS)
    progress = tqdm(total=expected, desc="Caching ROI crops")
    for row in subjects.itertuples(index=False):
        source_split = str(row.source_split)
        subject_id = str(row.subject_id).zfill(4)
        for view in VIEWS:
            image_path = data_root / getattr(row, f"image_{view}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing source image: {image_path}")

            pending: list[tuple[str, Path, object]] = []
            for region in required_regions:
                output_path = cache_path_for(
                    cache_root, source_split, subject_id, region, view
                )
                key = (source_split, subject_id, view, region)
                if key not in bbox_lookup:
                    raise KeyError(f"Missing bbox for {key}")
                if overwrite or not output_path.is_file():
                    pending.append((region, output_path, bbox_lookup[key]))
                else:
                    progress.update(1)

            if not pending:
                continue

            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                for _, output_path, bbox in pending:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    crop = crop_with_margin(image, bbox, margin_ratio)
                    crop.save(output_path, "JPEG", quality=95, subsampling=0)
                    progress.update(1)
    progress.close()


class MultiRegionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cache_root: str | Path,
        regions: tuple[str, ...],
        target_specs: tuple[TargetSpec, ...],
        transform,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.cache_root = Path(cache_root)
        self.regions = regions
        self.target_specs = target_specs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.dataframe.iloc[index]
        region_tensors: list[torch.Tensor] = []
        for region in self.regions:
            view_tensors: list[torch.Tensor] = []
            for view in VIEWS:
                path = cache_path_for(
                    self.cache_root,
                    str(row["source_split"]),
                    str(row["subject_id"]).zfill(4),
                    region,
                    view,
                )
                with Image.open(path) as source:
                    image = source.convert("RGB")
                view_tensors.append(self.transform(image))
            region_tensors.append(torch.stack(view_tensors, dim=0))

        targets = torch.tensor(
            [float(row[spec.column]) for spec in self.target_specs],
            dtype=torch.float32,
        )
        return {
            "images": torch.stack(region_tensors, dim=0),
            "targets": targets,
            "subject_id": str(row["subject_id"]),
        }
