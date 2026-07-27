#!/usr/bin/env python3
"""SafeWalk live inference application for Coral Dev Board Mini."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

from audio_feedback import AudioFeedback
from model_utils import load_interpreter, run_inference
from spatial_reasoning import (
    object_position,
    relative_distance,
    select_highest_risk,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "safewalk_edgetpu.tflite"
DEFAULT_LABELS = PROJECT_ROOT / "models" / "labels.txt"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.json"

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
        description="Run SafeWalk on the Coral Dev Board Mini.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--display", action="store_true")
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
    return detections


def draw_detections(frame, detections, risk, fps):
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = CLASS_COLORS[detection["class_id"] % len(CLASS_COLORS)]
        thickness = 4 if detection is risk else 2
        label = "{} {:.0f}% {} {}".format(
            detection["class"],
            detection["confidence"] * 100,
            detection["distance"],
            detection["position"],
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "SafeWalk  {:.1f} FPS".format(fps),
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    args = parse_args()

    for required_path in (args.model, args.labels, args.config):
        if not required_path.exists():
            print("Missing required file: {}".format(required_path))
            return 1

    if not args.model.name.endswith("_edgetpu.tflite"):
        print(
            "Model filename must end with _edgetpu.tflite: {}".format(
                args.model.name
            )
        )
        return 1

    config = load_json(args.config)
    labels = load_labels(args.labels)
    confidence = (
        args.confidence
        if args.confidence is not None
        else config["confidence_threshold"]
    )

    print("Loading Edge TPU model:", args.model)
    interpreter = load_interpreter(args.model)
    print("Loaded {} classes.".format(len(labels)))

    audio = AudioFeedback(
        cooldown_seconds=config["warning_cooldown_seconds"],
        speech_rate=config["speech_rate"],
        enabled=not args.no_audio,
    )
    if audio.enabled and not audio.available:
        print("espeak was not found; warnings will be printed only.")

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print("Could not open camera index {}.".format(args.camera))
        audio.close()
        return 1

    print("SafeWalk is running. Press Ctrl+C to stop.")
    previous_time = time.monotonic()
    fps = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Camera frame could not be read.")
                break

            detections = run_inference(
                interpreter=interpreter,
                frame=frame,
                labels=labels,
                confidence_threshold=confidence,
                iou_threshold=config["iou_threshold"],
            )
            detections = enrich_detections(detections, frame, config)
            risk = select_highest_risk(
                detections,
                danger_weights=config["danger_weights"],
                silent_classes=config["silent_classes"],
            )
            audio.speak(risk)

            now = time.monotonic()
            instantaneous_fps = 1.0 / max(0.0001, now - previous_time)
            fps = (
                instantaneous_fps
                if fps == 0.0
                else 0.85 * fps + 0.15 * instantaneous_fps
            )
            previous_time = now

            if risk is not None:
                print(
                    "\rRisk: {:<22} {:<6} {:<7} {:>5.1f}%  FPS {:>4.1f}".format(
                        risk["class"],
                        risk["distance"],
                        risk["position"],
                        risk["confidence"] * 100,
                        fps,
                    ),
                    end="",
                )
                sys.stdout.flush()

            if args.display:
                preview = draw_detections(
                    frame.copy(),
                    detections,
                    risk,
                    fps,
                )
                cv2.imshow("SafeWalk Coral", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("\nStopping SafeWalk.")
    finally:
        camera.release()
        audio.close()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())

