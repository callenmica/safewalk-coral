import sys
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.modules.setdefault("cv2", MagicMock())

from camera_source import LatestFrameCamera  # noqa: E402


class FakeCapture:
    def __init__(self):
        self.released = False
        self.frame_number = 0

    def isOpened(self):
        return True

    def read(self):
        if self.released:
            return False, None
        self.frame_number += 1
        frame = np.full((2, 2, 3), self.frame_number, dtype=np.uint8)
        time.sleep(0.005)
        return True, frame

    def release(self):
        self.released = True


class LatestFrameCameraTests(unittest.TestCase):
    def test_reader_exposes_only_the_latest_frame(self):
        capture = FakeCapture()
        with patch("camera_source.open_camera", return_value=capture):
            camera = LatestFrameCamera("http://camera/stream.mjpg")
            self.assertTrue(camera.start())

            deadline = time.monotonic() + 1.0
            sequence = 0
            while sequence < 3 and time.monotonic() < deadline:
                sequence, frame, fps, running = camera.snapshot()
                time.sleep(0.005)

            camera.stop()

        self.assertGreaterEqual(sequence, 3)
        self.assertEqual(int(frame[0, 0, 0]), sequence)
        self.assertGreater(fps, 0.0)
        self.assertTrue(running)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
