"""Open local V4L2 cameras or network video streams."""

import threading
import time

import cv2


def normalize_camera_source(value):
    value = str(value).strip()
    try:
        return int(value)
    except ValueError:
        return value


def open_camera(source, width=640, height=480, fps=15):
    source = normalize_camera_source(source)

    if isinstance(source, str):
        print("Opening Wi-Fi camera stream:", source)
        return cv2.VideoCapture(source)

    print("Opening local camera index {}.".format(source))
    camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, max(1, width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, max(1, height))
    camera.set(cv2.CAP_PROP_FPS, max(1, fps))
    return camera


class LatestFrameCamera:
    """Drain a camera continuously and retain only its newest frame."""

    def __init__(self, source, width=640, height=480, fps=15):
        self.capture = open_camera(source, width=width, height=height, fps=fps)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._frame = None
        self._sequence = 0
        self._fps = 0.0
        self._running = False

    def start(self):
        if not self.capture.isOpened():
            return False
        with self._lock:
            self._running = True
        self._thread = threading.Thread(target=self._read_loop)
        self._thread.daemon = True
        self._thread.start()
        return True

    @property
    def running(self):
        with self._lock:
            return self._running

    def snapshot(self):
        with self._lock:
            return self._sequence, self._frame, self._fps, self._running

    def _read_loop(self):
        previous_frame_time = None
        failure_started_at = None

        while not self._stop_event.is_set():
            success, frame = self.capture.read()
            if not success:
                if failure_started_at is None:
                    failure_started_at = time.monotonic()
                if time.monotonic() - failure_started_at >= 3.0:
                    print("Camera stream stopped after repeated read failures.")
                    break
                time.sleep(0.02)
                continue

            failure_started_at = None
            now = time.monotonic()
            instantaneous_fps = 0.0
            if previous_frame_time is not None:
                instantaneous_fps = 1.0 / max(
                    0.0001,
                    now - previous_frame_time,
                )
            previous_frame_time = now

            with self._lock:
                self._frame = frame
                self._sequence += 1
                if instantaneous_fps > 0.0:
                    self._fps = (
                        instantaneous_fps
                        if self._fps == 0.0
                        else self._fps * 0.85 + instantaneous_fps * 0.15
                    )

        with self._lock:
            self._running = False

    def stop(self):
        self._stop_event.set()
        self.capture.release()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            self._running = False
