import unittest

import numpy as np

from skin_project.bbox import has_bbox


class BboxAvailabilityTest(unittest.TestCase):
    def test_numpy_nan_is_missing(self):
        self.assertFalse(has_bbox(np.float64("nan")))

    def test_python_none_is_missing(self):
        self.assertFalse(has_bbox(None))

    def test_string_nan_is_missing(self):
        self.assertFalse(has_bbox("nan"))

    def test_serialized_bbox_is_available(self):
        self.assertTrue(has_bbox("[1, 2, 3, 4]"))

    def test_list_bbox_is_available(self):
        self.assertTrue(has_bbox([1, 2, 3, 4]))


if __name__ == "__main__":
    unittest.main()
