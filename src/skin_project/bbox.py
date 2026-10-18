from __future__ import annotations

import ast

import pandas as pd


def parse_bbox(value: object) -> tuple[int, int, int, int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Invalid bbox: {value!r}")
    x1, y1, x2, y2 = (int(round(float(number))) for number in value)
    return x1, y1, x2, y2


def has_bbox(value: object) -> bool:
    if value is None:
        return False
    # CSV parsing commonly produces numpy.float64 NaN rather than a built-in
    # Python float. is_scalar + isna covers both without applying an ambiguous
    # truth-value test to list-like bbox values.
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
        return False
    return True
