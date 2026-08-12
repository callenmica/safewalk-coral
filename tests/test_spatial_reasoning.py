import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_reasoning import (  # noqa: E402
    CollisionTracker,
    dynamic_hazard_type,
    escape_route,
    object_position,
    rank_hazards,
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

    def test_rank_hazards_and_dynamic_types(self):
        detections = [
            {
                "class": "Person",
                "confidence": 0.90,
                "distance": "medium",
                "position": "left",
            },
            {
                "class": "Manhole",
                "confidence": 0.75,
                "distance": "near",
                "position": "centre",
            },
        ]
        ranked = rank_hazards(
            detections,
            danger_weights={"Person": 0.75, "Manhole": 1.0},
        )
        self.assertEqual(ranked[0]["class"], "Manhole")
        self.assertEqual(dynamic_hazard_type("Tree"), "head-level")
        self.assertEqual(dynamic_hazard_type("Manhole"), "ground-level")
        self.assertEqual(dynamic_hazard_type("Person"), "obstacle")

    def test_escape_route(self):
        detections = [
            {
                "class": "Car",
                "distance": "near",
                "position": "centre",
            },
            {
                "class": "Person",
                "distance": "far",
                "position": "right",
            },
        ]
        instruction, position, occupancy = escape_route(detections)
        self.assertEqual(instruction, "STEER LEFT")
        self.assertEqual(position, "left")
        self.assertGreater(occupancy["centre"], occupancy["left"])

    def test_collision_tracker(self):
        tracker = CollisionTracker(growth_threshold=0.03)
        first = [
            {
                "class": "Car",
                "position": "centre",
                "distance": "medium",
                "box": (100, 100, 300, 300),
            }
        ]
        second = [
            {
                "class": "Car",
                "position": "centre",
                "distance": "near",
                "box": (50, 50, 450, 450),
            }
        ]
        self.assertFalse(tracker.update(first, 1000, 1000))
        self.assertTrue(tracker.update(second, 1000, 1000))


if __name__ == "__main__":
    unittest.main()
