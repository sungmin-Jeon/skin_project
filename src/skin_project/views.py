from __future__ import annotations


VIEWS = ("F", "L", "R")


VIEW_POLICIES: dict[str, dict[str, tuple[str, ...]]] = {
    "all": {},
    "front": {
        "forehead": ("F",),
        "glabella": ("F",),
        "right_eye": ("F",),
        "left_eye": ("F",),
        "right_cheek": ("F",),
        "left_cheek": ("F",),
        "lips": ("F",),
        "chin": ("F",),
    },
    # Anatomical right-side regions are visible in F/R; left-side regions in F/L.
    "service": {
        "forehead": ("F",),
        "glabella": ("F",),
        "right_eye": ("F", "R"),
        "left_eye": ("F", "L"),
        "right_cheek": ("F", "R"),
        "left_cheek": ("F", "L"),
        "lips": ("F",),
        "chin": ("F", "L", "R"),
    },
}


def views_for_region(policy: str, region: str) -> tuple[str, ...]:
    try:
        mapping = VIEW_POLICIES[policy]
    except KeyError as exc:
        choices = ", ".join(VIEW_POLICIES)
        raise ValueError(f"Unknown view policy {policy!r}; choose: {choices}") from exc
    return mapping.get(region, VIEWS)

