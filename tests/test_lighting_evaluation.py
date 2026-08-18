import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_lighting import lighting_variants, mean_brightness  # noqa: E402


class LightingEvaluationTests(unittest.TestCase):
    def test_variants_have_ordered_brightness(self):
        image = np.full((20, 20, 3), 100, dtype=np.uint8)

        variants = dict(lighting_variants(image))

        self.assertGreater(
            mean_brightness(variants["bright"]),
            mean_brightness(variants["normal"]),
        )
        self.assertGreater(
            mean_brightness(variants["normal"]),
            mean_brightness(variants["low_light"]),
        )


if __name__ == "__main__":
    unittest.main()
