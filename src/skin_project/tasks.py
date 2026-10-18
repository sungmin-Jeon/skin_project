from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TargetKind = Literal["grade", "continuous"]


@dataclass(frozen=True)
class TargetSpec:
    column: str
    region: str
    kind: TargetKind
    minimum: float | None = None
    maximum: float | None = None


FACEPART_TO_REGION = {
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


# The l_/r_ names below are the dataset's original measurement-side names.
# Their image-side facepart mapping was verified during EDA and is intentionally
# preserved here instead of silently renaming the source label columns.
GRADE_TARGETS = (
    TargetSpec("ann__forehead_pigmentation", "forehead", "grade", 0, 5),
    TargetSpec("ann__forehead_wrinkle", "forehead", "grade", 0, 6),
    TargetSpec("ann__glabellus_wrinkle", "glabella", "grade", 0, 6),
    TargetSpec("ann__l_perocular_wrinkle", "right_eye", "grade", 0, 6),
    TargetSpec("ann__r_perocular_wrinkle", "left_eye", "grade", 0, 6),
    TargetSpec("ann__l_cheek_pigmentation", "right_cheek", "grade", 0, 5),
    TargetSpec("ann__l_cheek_pore", "right_cheek", "grade", 0, 5),
    TargetSpec("ann__r_cheek_pigmentation", "left_cheek", "grade", 0, 5),
    TargetSpec("ann__r_cheek_pore", "left_cheek", "grade", 0, 5),
    TargetSpec("ann__lip_dryness", "lips", "grade", 0, 4),
    TargetSpec("ann__chin_sagging", "chin", "grade", 0, 6),
)


MOISTURE_TARGETS = (
    TargetSpec("eq__forehead_moisture", "forehead", "continuous"),
    TargetSpec("eq__l_cheek_moisture", "right_cheek", "continuous"),
    TargetSpec("eq__r_cheek_moisture", "left_cheek", "continuous"),
    TargetSpec("eq__chin_moisture", "chin", "continuous"),
)


TASKS = {
    "forehead_1": (
        TargetSpec("ann__forehead_wrinkle", "forehead", "grade", 0, 6),
    ),
    "grades_11": GRADE_TARGETS,
    "core_15": GRADE_TARGETS + MOISTURE_TARGETS,
}


def get_task_specs(task: str) -> tuple[TargetSpec, ...]:
    try:
        return TASKS[task]
    except KeyError as exc:
        choices = ", ".join(TASKS)
        raise ValueError(f"Unknown task {task!r}. Choose one of: {choices}") from exc


def required_regions(specs: tuple[TargetSpec, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    for spec in specs:
        if spec.region not in ordered:
            ordered.append(spec.region)
    return tuple(ordered)
