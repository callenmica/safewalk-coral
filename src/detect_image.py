#!/usr/bin/env python3
"""Run SafeWalk Edge TPU detection on one image and save the result."""

import argparse
from pathlib import Path
import sys
import time

import cv2

from model_utils import load_interpreter, run_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT / "models" / "safewalk_ssd_mobilenet_v2_edgetpu.tflite"
)
DEFAULT_LABELS = PROJECT_ROOT / "models" / "labels.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect obstacles in one still image on the Edge TPU."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--confidence", type=float, default=0.35)
    return parser.parse_args()


def load_labels(path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    args = parse_args()
    for path in (args.image, args.model, args.labels):
        if not path.exists():
            raise FileNotFoundError(str(path))

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise RuntimeError("Unable to read image: {}".format(args.image))

    labels = load_labels(args.labels)
    interpreter = load_interpreter(args.model)

    started = time.perf_counter()
    detections = run_inference(
        interpreter=interpreter,
        frame=frame,
        labels=labels,
        confidence_threshold=args.confidence,
        iou_threshold=0.45,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = "{} {:.1f}%".format(
            detection["class"],
            detection["confidence"] * 100.0,
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 90), 3)
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 220, 90),
            2,
            cv2.LINE_AA,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError("Unable to save output: {}".format(args.output))

    print("SafeWalk Still-Image Detection")
    print("------------------------------")
    print("Model:       {}".format(args.model))
    print("Input:       {}".format(args.image))
    print("Output:      {}".format(args.output))
    print("Latency:     {:.3f} ms".format(latency_ms))
    print("Detections:  {}".format(len(detections)))
    for detection in detections:
        print(
            "- {:<24} {:>6.2f}%  {}".format(
                detection["class"],
                detection["confidence"] * 100.0,
                detection["box"],
            )
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("Detection failed: {}".format(error), file=sys.stderr)
        sys.exit(1)
