from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .data import crop_with_margin
from .tasks import FACEPART_TO_REGION


REGION_TO_FACEPART = {
    region: facepart
    for facepart, region in FACEPART_TO_REGION.items()
    if region != "full_face"
}


# MediaPipe Face Mesh indices. Eye regions include the eyebrow because the
# targets describe periocular wrinkles rather than only the eyeball contour.
REGION_LANDMARKS: dict[str, tuple[int, ...]] = {
    "forehead": (
        54, 103, 67, 109, 10, 338, 297, 332, 284,
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    ),
    "glabella": (107, 66, 105, 168, 6, 197, 336, 296, 334),
    "right_eye": (
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173,
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    ),
    "left_eye": (
        362, 382, 381, 380, 374, 373, 390, 249, 263,
        466, 388, 387, 386, 385, 384, 398,
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    ),
    "right_cheek": (
        50, 101, 205, 187, 123, 116, 117, 118, 119, 120,
        121, 128, 142, 203, 206, 216, 92, 165, 167, 147,
    ),
    "left_cheek": (
        280, 330, 425, 411, 352, 345, 346, 347, 348, 349,
        350, 357, 371, 423, 426, 436, 322, 391, 393, 376,
    ),
    "lips": (
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
        95, 78, 191, 80, 81, 82, 13, 312, 311, 310,
    ),
    "chin": (
        152, 377, 400, 378, 379, 365, 397, 288, 361, 323,
        454, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
    ),
}


REGION_PADDING = {
    "forehead": (0.08, 0.30),
    "glabella": (0.30, 0.25),
    "right_eye": (0.20, 0.35),
    "left_eye": (0.20, 0.35),
    "right_cheek": (0.15, 0.15),
    "left_cheek": (0.15, 0.15),
    "lips": (0.20, 0.35),
    "chin": (0.08, 0.10),
}


class FaceLandmarkDetector:
    def __init__(self, model_path: Path | None = None) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required. Install with: pip install -e '.[landmark]'"
            ) from exc
        self.mp = mp
        self.legacy = hasattr(mp, "solutions")
        if self.legacy:
            self.detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        else:
            if model_path is None or not model_path.is_file():
                raise RuntimeError(
                    "Current MediaPipe requires --face-landmarker-model "
                    "pointing to face_landmarker.task"
                )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path)
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
            self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )

    def detect(self, rgb_array: np.ndarray):
        if self.legacy:
            result = self.detector.process(rgb_array)
            if not result.multi_face_landmarks:
                return None
            return result.multi_face_landmarks[0].landmark
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_array),
        )
        result = self.detector.detect(image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def boxes_from_landmarks(
    normalized_landmarks,
    width: int,
    height: int,
) -> dict[str, tuple[int, int, int, int]]:
    points = np.asarray(
        [(landmark.x * width, landmark.y * height) for landmark in normalized_landmarks],
        dtype=np.float32,
    )
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for region, indices in REGION_LANDMARKS.items():
        selected = points[list(indices)]
        x1, y1 = selected.min(axis=0)
        x2, y2 = selected.max(axis=0)
        pad_x_ratio, pad_y_ratio = REGION_PADDING[region]
        pad_x = max((x2 - x1) * pad_x_ratio, 2.0)
        pad_y = max((y2 - y1) * pad_y_ratio, 2.0)
        x1 = int(max(0, np.floor(x1 - pad_x)))
        y1 = int(max(0, np.floor(y1 - pad_y)))
        x2 = int(min(width, np.ceil(x2 + pad_x)))
        y2 = int(min(height, np.ceil(y2 + pad_y)))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes[region] = (x1, y1, x2, y2)
    return boxes


def landmark_boxes_for_image(
    image: Image.Image,
    detector: FaceLandmarkDetector,
) -> dict[str, tuple[int, int, int, int]]:
    """Detect the same region boxes used by manifest creation and service code."""
    rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    landmarks = detector.detect(np.asarray(rgb_image))
    if landmarks is None:
        return {}
    return boxes_from_landmarks(landmarks, rgb_image.width, rgb_image.height)


def crop_regions_from_landmarks(
    image: Image.Image,
    detector: FaceLandmarkDetector,
    regions: tuple[str, ...] | None = None,
    margin_ratio: float = 0.10,
) -> dict[str, Image.Image]:
    """Create service crops with the same box and margin rules as training."""
    rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    boxes = landmark_boxes_for_image(rgb_image, detector)
    selected_regions = regions or tuple(REGION_LANDMARKS)
    return {
        region: crop_with_margin(rgb_image, boxes[region], margin_ratio)
        for region in selected_regions
        if region in boxes
    }


def create_landmark_manifest(
    subject_manifest: Path,
    data_root: Path,
    output_path: Path,
    face_landmarker_model: Path | None = None,
) -> pd.DataFrame:
    subjects = pd.read_csv(
        subject_manifest,
        dtype={"subject_id": "string", "device_id": "string"},
    )
    subjects["subject_id"] = subjects["subject_id"].str.zfill(4)
    rows: list[dict[str, object]] = []
    failures = 0

    with FaceLandmarkDetector(face_landmarker_model) as detector:
        for subject in tqdm(
            subjects.itertuples(index=False),
            total=len(subjects),
            desc="Generating landmark boxes",
        ):
            for view in ("F", "L", "R"):
                relative_image_path = getattr(subject, f"image_{view}")
                image_path = data_root / relative_image_path
                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                boxes = landmark_boxes_for_image(image, detector)
                if not boxes:
                    failures += 1

                for region, facepart in REGION_TO_FACEPART.items():
                    bbox = boxes.get(region)
                    rows.append(
                        {
                            "source_split": str(subject.source_split),
                            "subject_id": str(subject.subject_id),
                            "view": view,
                            "facepart": facepart,
                            "bbox": json.dumps(bbox) if bbox is not None else np.nan,
                            "bbox_source": "mediapipe_face_mesh",
                            "image_path": relative_image_path,
                        }
                    )

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Rows: {len(frame)}")
    print(f"Images with no detected face: {failures}")
    print(f"Missing region boxes: {frame['bbox'].isna().sum()}")
    print(f"Saved: {output_path}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate service-style region boxes with MediaPipe Face Mesh."
    )
    parser.add_argument("--subject-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--face-landmarker-model",
        type=Path,
        help="Path to face_landmarker.task for the current MediaPipe Tasks API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_landmark_manifest(
        args.subject_manifest,
        args.data_root,
        args.output,
        args.face_landmarker_model,
    )


if __name__ == "__main__":
    main()

