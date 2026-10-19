import unittest

import pandas as pd
import torch

from skin_project.losses import ClassificationWeights, GradeFrequencyWeights
from skin_project.tasks import TargetSpec


class GradeFrequencyWeightsTest(unittest.TestCase):
    def setUp(self):
        self.specs = (
            TargetSpec("grade", "forehead", "grade", 0, 2),
            TargetSpec("continuous", "forehead", "continuous"),
        )
        self.train_df = pd.DataFrame(
            {
                "grade": [0, 0, 0, 1],
                "continuous": [10.0, 20.0, 30.0, 40.0],
            }
        )

    def test_rare_grade_receives_more_weight(self):
        weighter = GradeFrequencyWeights(self.specs, self.train_df)
        weights = weighter.for_targets(
            torch.tensor([[0.0, 10.0], [1.0, 20.0]])
        )
        self.assertGreater(weights[1, 0].item(), weights[0, 0].item())
        self.assertEqual(weights[0, 1].item(), 1.0)
        self.assertEqual(weights[1, 1].item(), 1.0)

    def test_zero_power_disables_reweighting(self):
        weighter = GradeFrequencyWeights(
            self.specs, self.train_df, power=0.0
        )
        weights = weighter.for_targets(
            torch.tensor([[0.0, 10.0], [1.0, 20.0]])
        )
        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_invalid_maximum_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            GradeFrequencyWeights(
                self.specs, self.train_df, max_weight=0.5
            )

    def test_class_balancing_upweights_rare_grade(self):
        weighter = ClassificationWeights(
            self.specs,
            self.train_df,
            beta=0.999,
        )
        weights = weighter.class_weights("grade", torch.device("cpu"))
        self.assertGreater(weights[1].item(), weights[0].item())
        self.assertEqual(len(weights), 3)

    def test_ordinal_weights_have_one_value_per_threshold(self):
        weighter = ClassificationWeights(self.specs, self.train_df)
        weights = weighter.ordinal_positive_weights(
            "grade", torch.device("cpu")
        )
        self.assertEqual(len(weights), 2)


if __name__ == "__main__":
    unittest.main()

