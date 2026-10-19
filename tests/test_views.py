import unittest

from skin_project.views import views_for_region


class ViewPolicyTest(unittest.TestCase):
    def test_service_policy_uses_visible_eye_views(self):
        self.assertEqual(views_for_region("service", "left_eye"), ("F", "L"))
        self.assertEqual(views_for_region("service", "right_eye"), ("F", "R"))

    def test_service_policy_uses_front_for_forehead(self):
        self.assertEqual(views_for_region("service", "forehead"), ("F",))

    def test_all_policy_falls_back_to_three_views(self):
        self.assertEqual(views_for_region("all", "chin"), ("F", "L", "R"))


if __name__ == "__main__":
    unittest.main()

