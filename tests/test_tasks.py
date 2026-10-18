import unittest

from skin_project.tasks import TASKS, required_regions


class TaskConfigurationTest(unittest.TestCase):
    def test_expected_target_counts(self):
        self.assertEqual(len(TASKS["forehead_1"]), 1)
        self.assertEqual(len(TASKS["grades_11"]), 11)
        self.assertEqual(len(TASKS["core_15"]), 15)

    def test_target_columns_are_unique_within_each_task(self):
        for task_name, specs in TASKS.items():
            with self.subTest(task=task_name):
                columns = [spec.column for spec in specs]
                self.assertEqual(len(columns), len(set(columns)))

    def test_core_task_uses_eight_regions(self):
        self.assertEqual(len(required_regions(TASKS["core_15"])), 8)


if __name__ == "__main__":
    unittest.main()
