#!/usr/bin/env python3
"""SafeWalk enhanced dashboard using real Edge TPU inference."""

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from audio_feedback import AudioFeedback
from camera_source import open_camera
from model_utils import load_interpreter, run_inference
from spatial_reasoning import (
    CollisionTracker,
    dynamic_hazard_type,
    escape_route,
    object_position,
    rank_hazards,
    relative_distance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "safewalk_edgetpu.tflite"
DEFAULT_LABELS = PROJECT_ROOT / "models" / "labels.txt"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.json"

WINDOW_NAME = "SafeWalk Coral Dashboard"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 760
TOP_HEIGHT = 64
BOTTOM_HEIGHT = 44
MARGIN = 18
SIDE_WIDTH = 350
CAMERA_X = MARGIN
CAMERA_Y = TOP_HEIGHT + MARGIN
CAMERA_WIDTH = WINDOW_WIDTH - SIDE_WIDTH - MARGIN * 3
CAMERA_HEIGHT = WINDOW_HEIGHT - CAMERA_Y - BOTTOM_HEIGHT - MARGIN
SIDE_X = CAMERA_X + CAMERA_WIDTH + MARGIN
SIDE_Y = CAMERA_Y
SIDE_HEIGHT = CAMERA_HEIGHT

BACKGROUND = (24, 24, 24)
TOP_BAR = (31, 31, 31)
PANEL = (40, 40, 40)
PANEL_DARK = (28, 28, 28)
BORDER = (75, 75, 75)
TEXT = (240, 242, 244)
MUTED = (155, 165, 175)
TEAL = (186, 209, 51)
GREEN = (90, 200, 80)
AMBER = (55, 175, 255)
RED = (75, 75, 245)
BUTTON = (52, 60, 68)

CLASS_COLORS = [
    (0, 220, 255),
    (220, 150, 50),
    (70, 70, 240),
    (70, 210, 100),
    (220, 90, 220),
    (40, 180, 250),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the SafeWalk Edge TPU dashboard.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--camera",
        default="0",
        help="Local camera index or HTTP/RTSP Wi-Fi stream URL.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--no-audio", action="store_true")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_labels(path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def enrich_detections(detections, frame, config):
    frame_height, frame_width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        detection["area"] = max(0, x2 - x1) * max(0, y2 - y1)
        detection["position"] = object_position(
            detection["box"],
            frame_width,
        )
        detection["distance"] = relative_distance(
            detection["box"],
            frame_width,
            frame_height,
            near_area_ratio=config["near_area_ratio"],
            medium_area_ratio=config["medium_area_ratio"],
        )
        detection["hazard_type"] = dynamic_hazard_type(
            detection["class"]
        )
    return detections


def readable_class_name(class_name):
    replacements = {
        "Bike": "bicycle",
        "Teraffic Barrel": "traffic barrel",
    }
    return replacements.get(class_name, class_name.lower())


def spoken_position(position):
    return "ahead" if position == "centre" else position


def hazard_clause(detection):
    name = readable_class_name(detection["class"])
    position = spoken_position(detection["position"])
    hazard_type = detection["hazard_type"]
    if hazard_type == "head-level":
        return "head-level {} {}, {}".format(
            name,
            detection["distance"],
            position,
        )
    if hazard_type == "ground-level":
        return "ground hazard {}, {}".format(name, position)
    return "{} {} {}".format(detection["distance"], name, position)


def warning_phrase(ranked, route_text, low_light, collision):
    parts = []
    if collision:
        parts.append("Imminent collision. Stop immediately")
    if low_light:
        parts.append("Environment too dark for reliable vision assistance")

    highest_confidence = sorted(
        ranked,
        key=lambda item: item["confidence"],
        reverse=True,
    )[:2]
    hazards = [hazard_clause(item) for item in highest_confidence]
    if hazards:
        parts.append("Caution, " + ", and ".join(hazards))

    if route_text.startswith("STEER"):
        parts.append(route_text.lower())
    elif route_text == "STOP AND WAIT" and not collision:
        parts.append("Stop and wait")

    return ". ".join(parts) + ("." if parts else "")


def threat_level(ranked, low_light, collision):
    if collision or any(item["distance"] == "near" for item in ranked):
        return "CRITICAL", RED
    if low_light or any(item["distance"] == "medium" for item in ranked):
        return "WARNING", AMBER
    return "SAFE", GREEN


def draw_text(image, text, position, scale=0.52, color=TEXT, thickness=1):
    cv2.putText(
        image,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_button(image, regions, key, rect, label, active=False):
    regions[key] = rect
    x1, y1, x2, y2 = rect
    fill = TEAL if active else BUTTON
    cv2.rectangle(image, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), BORDER, 1)
    color = (18, 22, 24) if active else TEXT
    size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    text_x = x1 + max(4, (x2 - x1 - size[0]) // 2)
    text_y = y1 + (y2 - y1 + size[1]) // 2
    draw_text(image, label, (text_x, text_y), 0.43, color, 1)


def fit_frame(frame, width, height):
    source_height, source_width = frame.shape[:2]
    scale = min(width / float(source_width), height / float(source_height))
    target = (
        max(1, int(source_width * scale)),
        max(1, int(source_height * scale)),
    )
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def draw_detections(frame, detections, ranked):
    primary = ranked[0] if ranked else None
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = CLASS_COLORS[
            detection["class_id"] % len(CLASS_COLORS)
        ]
        thickness = 4 if detection is primary else 2
        label = "{} {:.0f}% {} {}".format(
            detection["class"],
            detection["confidence"] * 100,
            detection["distance"],
            spoken_position(detection["position"]),
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        draw_text(
            frame,
            label,
            (x1, max(22, y1 - 7)),
            0.52,
            color,
            2,
        )
    return frame


def draw_radar(canvas, detections, route_position, x, y, width, height):
    columns = {"left": 0, "centre": 1, "right": 2}
    rows = {"far": 0, "medium": 1, "near": 2}
    cell_width = width // 3
    cell_height = height // 3
    threats = [[0 for _ in range(3)] for _ in range(3)]

    for detection in detections:
        row = rows[detection["distance"]]
        column = columns[detection["position"]]
        threats[row][column] = max(
            threats[row][column],
            2 if detection["distance"] == "near" else 1,
        )

    for row in range(3):
        for column in range(3):
            x1 = x + column * cell_width
            y1 = y + row * cell_height
            x2 = x + (column + 1) * cell_width
            y2 = y + (row + 1) * cell_height
            value = threats[row][column]
            if value == 2:
                fill = (35, 35, 105)
            elif value == 1:
                fill = (35, 70, 100)
            elif route_position is not None and column == columns[route_position]:
                fill = (35, 78, 42)
            else:
                fill = (34, 55, 38)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), BORDER, 1)

    for detection in detections:
        column = columns[detection["position"]]
        row = rows[detection["distance"]]
        center = (
            x + column * cell_width + cell_width // 2,
            y + row * cell_height + cell_height // 2,
        )
        color = CLASS_COLORS[
            detection["class_id"] % len(CLASS_COLORS)
        ]
        cv2.circle(canvas, center, 5, color, -1)

    draw_text(canvas, "LEFT", (x + 5, y + 13), 0.30, MUTED, 1)
    draw_text(
        canvas,
        "AHEAD",
        (x + cell_width + 5, y + 13),
        0.30,
        MUTED,
        1,
    )
    draw_text(
        canvas,
        "RIGHT",
        (x + cell_width * 2 + 5, y + 13),
        0.30,
        MUTED,
        1,
    )


def draw_latency_graph(canvas, history, x, y, width, height):
    cv2.rectangle(canvas, (x, y), (x + width, y + height), PANEL_DARK, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), BORDER, 1)
    if len(history) < 2:
        return
    maximum = max(1.0, max(history))
    points = []
    step = width / float(max(1, history.maxlen - 1))
    for index, value in enumerate(history):
        point_x = int(x + index * step)
        point_y = int(y + height - 4 - value / maximum * (height - 8))
        points.append((point_x, point_y))
    for index in range(len(points) - 1):
        cv2.line(canvas, points[index], points[index + 1], TEAL, 2)


def build_dashboard(
    frame,
    detections,
    ranked,
    route_text,
    route_position,
    low_light,
    collision,
    inference_ms,
    raw_fps,
    latency_history,
    confidence,
    camera_running,
    audio_enabled,
    source_label,
    regions,
):
    canvas = np.full(
        (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
        BACKGROUND,
        dtype=np.uint8,
    )
    regions.clear()

    cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, TOP_HEIGHT), TOP_BAR, -1)
    draw_text(canvas, "SafeWalk", (24, 39), 0.88, TEXT, 2)
    draw_text(canvas, "REAL EDGE TPU", (172, 38), 0.46, TEAL, 2)
    draw_text(
        canvas,
        datetime.now().strftime("%H:%M:%S"),
        (WINDOW_WIDTH - 150, 38),
        0.62,
        TEXT,
        2,
    )
    if low_light:
        cv2.rectangle(canvas, (330, 17), (510, 48), RED, -1)
        draw_text(canvas, "LOW LIGHT ALERT", (343, 38), 0.45, TEXT, 2)
    if collision:
        cv2.rectangle(canvas, (525, 17), (735, 48), RED, -1)
        draw_text(canvas, "IMMINENT COLLISION", (537, 38), 0.43, TEXT, 2)

    cv2.rectangle(
        canvas,
        (CAMERA_X, CAMERA_Y),
        (CAMERA_X + CAMERA_WIDTH, CAMERA_Y + CAMERA_HEIGHT),
        PANEL_DARK,
        -1,
    )
    if frame is not None:
        annotated = draw_detections(frame.copy(), detections, ranked)
        fitted = fit_frame(annotated, CAMERA_WIDTH, CAMERA_HEIGHT)
        image_height, image_width = fitted.shape[:2]
        offset_x = CAMERA_X + (CAMERA_WIDTH - image_width) // 2
        offset_y = CAMERA_Y + (CAMERA_HEIGHT - image_height) // 2
        canvas[
            offset_y : offset_y + image_height,
            offset_x : offset_x + image_width,
        ] = fitted
    else:
        draw_text(
            canvas,
            "Camera stopped or unavailable",
            (CAMERA_X + 240, CAMERA_Y + CAMERA_HEIGHT // 2),
            0.65,
            MUTED,
            2,
        )
    cv2.rectangle(
        canvas,
        (CAMERA_X, CAMERA_Y),
        (CAMERA_X + CAMERA_WIDTH, CAMERA_Y + CAMERA_HEIGHT),
        BORDER,
        1,
    )

    cv2.rectangle(
        canvas,
        (SIDE_X, SIDE_Y),
        (SIDE_X + SIDE_WIDTH, SIDE_Y + SIDE_HEIGHT),
        PANEL,
        -1,
    )
    x = SIDE_X + 16
    width = SIDE_WIDTH - 32
    level, level_color = threat_level(ranked, low_light, collision)
    draw_text(canvas, "THREAT LEVEL", (x, SIDE_Y + 22), 0.38, MUTED, 1)
    cv2.rectangle(
        canvas,
        (x + 112, SIDE_Y + 6),
        (x + 215, SIDE_Y + 27),
        level_color,
        -1,
    )
    draw_text(canvas, level, (x + 122, SIDE_Y + 22), 0.39, TEXT, 2)
    draw_text(canvas, route_text, (x, SIDE_Y + 55), 0.70, GREEN, 2)

    if ranked:
        primary = ranked[0]
        draw_text(
            canvas,
            "{} {:.0f}%".format(
                primary["class"],
                primary["confidence"] * 100,
            ),
            (x, SIDE_Y + 82),
            0.52,
            TEXT,
            2,
        )
    else:
        draw_text(canvas, "No detected hazard", (x, SIDE_Y + 82), 0.48, MUTED, 1)

    draw_text(canvas, "SPATIAL RADAR", (x, SIDE_Y + 112), 0.38, MUTED, 1)
    draw_radar(
        canvas,
        ranked,
        route_position,
        x,
        SIDE_Y + 120,
        width,
        126,
    )

    metrics_y = SIDE_Y + 274
    draw_text(canvas, "HARDWARE", (x, metrics_y), 0.34, MUTED, 1)
    draw_text(canvas, "INFERENCE", (x + 112, metrics_y), 0.34, MUTED, 1)
    draw_text(canvas, "RAW FPS", (x + 230, metrics_y), 0.34, MUTED, 1)
    draw_text(canvas, "EDGE TPU", (x, metrics_y + 24), 0.48, TEAL, 2)
    draw_text(
        canvas,
        "{:.1f} ms".format(inference_ms),
        (x + 112, metrics_y + 24),
        0.48,
        TEXT,
        2,
    )
    draw_text(
        canvas,
        "{:.1f}".format(raw_fps),
        (x + 230, metrics_y + 24),
        0.48,
        TEXT,
        2,
    )
    draw_text(canvas, "LATENCY HISTORY", (x, metrics_y + 51), 0.34, MUTED, 1)
    draw_latency_graph(canvas, latency_history, x, metrics_y + 58, width, 54)

    list_y = metrics_y + 138
    draw_text(canvas, "TOP HAZARDS", (x, list_y), 0.38, MUTED, 1)
    for index, detection in enumerate(ranked[:4]):
        row_y = list_y + 25 + index * 25
        text = "{}  {:.0f}%  {} {}".format(
            detection["class"][:15],
            detection["confidence"] * 100,
            detection["distance"],
            spoken_position(detection["position"]),
        )
        draw_text(canvas, text, (x, row_y), 0.40, TEXT, 1)

    controls_y = SIDE_Y + SIDE_HEIGHT - 76
    draw_text(
        canvas,
        "CONFIDENCE {:.0f}%".format(confidence * 100),
        (x, controls_y - 9),
        0.38,
        MUTED,
        1,
    )
    draw_button(canvas, regions, "confidence_down", (x + 190, controls_y - 30, x + 230, controls_y), "-")
    draw_button(canvas, regions, "confidence_up", (x + 238, controls_y - 30, x + 278, controls_y), "+")
    draw_button(
        canvas,
        regions,
        "camera",
        (x, controls_y + 8, x + 98, controls_y + 40),
        "Stop" if camera_running else "Retry",
        active=camera_running,
    )
    draw_button(
        canvas,
        regions,
        "audio",
        (x + 108, controls_y + 8, x + 218, controls_y + 40),
        "Mute" if audio_enabled else "Audio on",
        active=audio_enabled,
    )
    draw_button(
        canvas,
        regions,
        "quit",
        (x + 228, controls_y + 8, x + width, controls_y + 40),
        "Quit",
    )

    bottom_y = WINDOW_HEIGHT - BOTTOM_HEIGHT
    cv2.rectangle(canvas, (0, bottom_y), (WINDOW_WIDTH, WINDOW_HEIGHT), TOP_BAR, -1)
    source_text = source_label
    if len(source_text) > 105:
        source_text = source_text[:102] + "..."
    draw_text(canvas, "SOURCE: " + source_text, (20, bottom_y + 27), 0.38, MUTED, 1)
    return canvas


def main():
    args = parse_args()
    for required_path in (args.model, args.labels, args.config):
        if not required_path.exists():
            print("Missing required file: {}".format(required_path))
            return 1

    config = load_json(args.config)
    labels = load_labels(args.labels)
    confidence = (
        args.confidence
        if args.confidence is not None
        else config["confidence_threshold"]
    )

    print("Loading real Edge TPU model:", args.model)
    interpreter = load_interpreter(args.model)
    camera = open_camera(
        args.camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    camera_running = camera.isOpened()
    if not camera_running:
        camera.release()
        camera = None

    audio = AudioFeedback(
        cooldown_seconds=config["warning_cooldown_seconds"],
        speech_rate=config["speech_rate"],
        enabled=not args.no_audio,
    )
    collision_tracker = CollisionTracker()
    latency_history = deque(maxlen=40)
    regions = {}
    pending_action = [None]
    last_frame = None
    detections = []
    ranked = []
    route_text = "PATH CLEAR"
    route_position = "centre"
    low_light = False
    collision = False
    inference_ms = 0.0
    raw_fps = 0.0
    previous_frame_time = time.monotonic()
    failed_reads = 0
    quit_requested = False

    def on_mouse(event, mouse_x, mouse_y, flags, userdata):
        del flags, userdata
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        get_window_rect = getattr(cv2, "getWindowImageRect", None)
        if get_window_rect is not None:
            try:
                _, _, display_width, display_height = get_window_rect(
                    WINDOW_NAME
                )
                if display_width > 0 and display_height > 0:
                    mouse_x = int(mouse_x * WINDOW_WIDTH / display_width)
                    mouse_y = int(mouse_y * WINDOW_HEIGHT / display_height)
            except cv2.error:
                pass
        for action, rect in regions.items():
            x1, y1, x2, y2 = rect
            if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
                pending_action[0] = action
                return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while not quit_requested:
            action = pending_action[0]
            pending_action[0] = None
            if action == "quit":
                quit_requested = True
            elif action == "audio":
                audio.enabled = not audio.enabled
            elif action == "confidence_down":
                confidence = max(0.10, confidence - 0.05)
            elif action == "confidence_up":
                confidence = min(0.90, confidence + 0.05)
            elif action == "camera":
                if camera_running:
                    camera.release()
                    camera = None
                    camera_running = False
                else:
                    camera = open_camera(
                        args.camera,
                        width=args.camera_width,
                        height=args.camera_height,
                        fps=args.camera_fps,
                    )
                    camera_running = camera.isOpened()
                    if not camera_running:
                        camera.release()
                        camera = None
                    failed_reads = 0

            if camera_running and camera is not None:
                success, frame = camera.read()
                if success:
                    failed_reads = 0
                    now = time.monotonic()
                    instantaneous_fps = 1.0 / max(
                        0.0001,
                        now - previous_frame_time,
                    )
                    raw_fps = (
                        instantaneous_fps
                        if raw_fps == 0.0
                        else raw_fps * 0.85 + instantaneous_fps * 0.15
                    )
                    previous_frame_time = now
                    low_light = (
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() < 40.0
                    )

                    inference_start = time.perf_counter()
                    detections = run_inference(
                        interpreter=interpreter,
                        frame=frame,
                        labels=labels,
                        confidence_threshold=confidence,
                        iou_threshold=config["iou_threshold"],
                    )
                    inference_ms = (
                        time.perf_counter() - inference_start
                    ) * 1000.0
                    latency_history.append(inference_ms)
                    detections = enrich_detections(
                        detections,
                        frame,
                        config,
                    )
                    ranked = rank_hazards(
                        detections,
                        danger_weights=config["danger_weights"],
                        silent_classes=config["silent_classes"],
                    )
                    frame_height, frame_width = frame.shape[:2]
                    collision = collision_tracker.update(
                        ranked,
                        frame_width,
                        frame_height,
                    )
                    route_text, route_position, _ = escape_route(
                        ranked,
                        silent_classes=config["silent_classes"],
                    )
                    audio.speak_phrase(
                        warning_phrase(
                            ranked,
                            route_text,
                            low_light,
                            collision,
                        )
                    )
                    last_frame = frame
                else:
                    failed_reads += 1
                    if failed_reads >= 10:
                        print("Camera stream stopped after repeated read failures.")
                        camera.release()
                        camera = None
                        camera_running = False

            dashboard = build_dashboard(
                last_frame if camera_running else None,
                detections if camera_running else [],
                ranked if camera_running else [],
                route_text,
                route_position,
                low_light,
                collision,
                inference_ms,
                raw_fps,
                latency_history,
                confidence,
                camera_running,
                audio.enabled,
                str(args.camera),
                regions,
            )
            cv2.imshow(WINDOW_NAME, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                quit_requested = True
            elif key in (ord("m"), ord("M")):
                pending_action[0] = "audio"
            elif key == ord(" "):
                pending_action[0] = "camera"
            elif key in (ord("-"), ord("_")):
                pending_action[0] = "confidence_down"
            elif key in (ord("+"), ord("=")):
                pending_action[0] = "confidence_up"
    except KeyboardInterrupt:
        print("\nStopping SafeWalk dashboard.")
    finally:
        if camera is not None:
            camera.release()
        audio.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
