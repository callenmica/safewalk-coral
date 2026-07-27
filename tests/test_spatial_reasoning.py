import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_reasoning import (  # noqa: E402
    object_position,
    relative_distance,
    select_highest_risk,
)


class SpatialReasoningTests(unittest.TestCase):
    def test_object_positions(self):
        self.assertEqual(object_position((0, 0, 100, 100), 900), "left")
        self.assertEqual(
            object_position((350, 0, 550, 100), 900),
            "centre",
        )
        self.assertEqual(
            object_position((700, 0, 850, 100), 900),
            "right",
        )

    def test_relative_distance(self):
        self.assertEqual(
            relative_distance((0, 0, 500, 500), 1000, 1000),
            "near",
        )
        self.assertEqual(
            relative_distance((0, 0, 300, 300), 1000, 1000),
            "medium",
        )
        self.assertEqual(
            relative_distance((0, 0, 100, 100), 1000, 1000),
            "far",
        )

    def test_risk_selection(self):
        detections = [
            {
                "class": "Road",
                "confidence": 0.99,
                "distance": "near",
                "position": "centre",
            },
            {
                "class": "Car",
                "confidence": 0.80,
                "distance": "near",
                "position": "centre",
            },
            {
                "class": "Person",
                "confidence": 0.95,
                "distance": "far",
                "position": "left",
            },
        ]
        selected = select_highest_risk(
            detections,
            danger_weights={"Car": 1.0, "Person": 0.75},
            silent_classes={"Road"},
        )
        self.assertEqual(selected["class"], "Car")


if __name__ == "__main__":
    unittest.main()

