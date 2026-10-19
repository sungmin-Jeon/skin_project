import unittest

import torch

from skin_project.model import OrdinalHead


class OrdinalHeadTest(unittest.TestCase):
    def test_cutpoints_remain_ordered(self):
        head = OrdinalHead(input_dim=4, class_count=7)
        cutpoints = head.cutpoints()
        self.assertEqual(tuple(cutpoints.shape), (6,))
        self.assertTrue(torch.all(cutpoints[1:] > cutpoints[:-1]))

    def test_output_has_one_logit_per_threshold(self):
        head = OrdinalHead(input_dim=4, class_count=5)
        output = head(torch.zeros(3, 4))
        self.assertEqual(tuple(output.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
