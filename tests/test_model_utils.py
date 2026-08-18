import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_utils import decode_ssd, quantize_input  # noqa: E402


class ModelUtilsTests(unittest.TestCase):
    def test_quantizes_ssd_minus_one_to_one_input(self):
        image = np.array([[[0, 128, 255]]], dtype=np.uint8)
        detail = {
            "dtype": np.uint8,
            "quantization": (1.0 / 127.5, 127),
        }

        tensor = quantize_input(image, detail)

        self.assertEqual(tensor.shape, (1, 1, 1, 3))
        self.assertEqual(tensor[0, 0, 0].tolist(), [0, 128, 254])

    def test_decodes_normalized_ssd_box_and_zero_based_class(self):
        detections = decode_ssd(
            boxes=np.array([[[0.10, 0.20, 0.60, 0.80]]]),
            class_ids=np.array([[2.0]]),
            scores=np.array([[0.90]]),
            num_detections=np.array([1.0]),
            labels=["Bike", "Building", "Car"],
            confidence_threshold=0.35,
            source_width=100,
            source_height=200,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "Car")
        self.assertEqual(detections[0]["box"], (20, 20, 80, 120))

    def test_filters_low_confidence_ssd_detection(self):
        detections = decode_ssd(
            boxes=np.array([[[0.10, 0.10, 0.50, 0.50]]]),
            class_ids=np.array([[0.0]]),
            scores=np.array([[0.20]]),
            num_detections=np.array([1.0]),
            labels=["Bike"],
            confidence_threshold=0.35,
            source_width=100,
            source_height=100,
        )

        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
