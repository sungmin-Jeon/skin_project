import unittest

from PIL import Image

from skin_project.landmarks import (
    REGION_LANDMARKS,
    boxes_from_landmarks,
    crop_regions_from_landmarks,
)


class FakeLandmark:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class FakeDetector:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def detect(self, image):
        return self.landmarks


class LandmarkBoxTest(unittest.TestCase):
    def test_glabella_uses_only_inner_brow_and_nose_bridge(self):
        self.assertEqual(REGION_LANDMARKS["glabella"], (107, 168, 6, 336))

    def test_glabella_box_is_centered_above_inner_brows(self):
        landmarks = [FakeLandmark(0.5, 0.5) for _ in range(478)]
        landmarks[107] = FakeLandmark(0.4, 0.5)
        landmarks[336] = FakeLandmark(0.6, 0.5)
        box = boxes_from_landmarks(landmarks, width=200, height=100)[
            "glabella"
        ]
        x1, y1, x2, y2 = box
        self.assertAlmostEqual((x1 + x2) / 2, 100, delta=1)
        self.assertLess((y1 + y2) / 2, 50)
        self.assertLess(x2 - x1, 40)

    def test_boxes_are_inside_image(self):
        landmarks = [FakeLandmark((index % 20) / 20, (index % 25) / 25) for index in range(478)]
        boxes = boxes_from_landmarks(landmarks, width=200, height=100)
        self.assertEqual(len(boxes), 8)
        for box in boxes.values():
            x1, y1, x2, y2 = box
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 200)
            self.assertLessEqual(y2, 100)
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)

    def test_service_crops_use_landmark_boxes(self):
        landmarks = [
            FakeLandmark((index % 20) / 20, (index % 25) / 25)
            for index in range(478)
        ]
        image = Image.new("RGB", (200, 100), "white")
        crops = crop_regions_from_landmarks(
            image,
            FakeDetector(landmarks),
            regions=("forehead", "lips"),
        )
        self.assertEqual(set(crops), {"forehead", "lips"})
        for crop in crops.values():
            self.assertGreater(crop.width, 0)
            self.assertGreater(crop.height, 0)


if __name__ == "__main__":
    unittest.main()

