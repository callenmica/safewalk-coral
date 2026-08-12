"""Open local V4L2 cameras or network video streams."""

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

