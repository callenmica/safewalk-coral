import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nms import non_max_suppression  # noqa: E402


class NonMaximumSuppressionTests(unittest.TestCase):
    def test_suppresses_overlapping_boxes_of_same_class(self):
        selected = non_max_suppression(
            boxes=np.array(
                [
                    [0, 0, 100, 100],
                    [5, 5, 105, 105],
                    [200, 200, 250, 250],
                ]
            ),
            scores=np.array([0.90, 0.80, 0.70]),
            class_ids=np.array([0, 0, 0]),
            iou_threshold=0.50,
        )
        self.assertEqual(selected.tolist(), [0, 2])

    def test_keeps_overlapping_boxes_of_different_classes(self):
        selected = non_max_suppression(
            boxes=np.array([[0, 0, 100, 100], [0, 0, 100, 100]]),
            scores=np.array([0.80, 0.90]),
            class_ids=np.array([0, 1]),
            iou_threshold=0.50,
        )
        self.assertEqual(selected.tolist(), [1, 0])

    def test_accepts_empty_input(self):
        selected = non_max_suppression([], [], [], iou_threshold=0.50)
        self.assertEqual(selected.tolist(), [])


if __name__ == "__main__":
    unittest.main()
