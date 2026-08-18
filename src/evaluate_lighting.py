#!/usr/bin/env python3
"""Evaluate SafeWalk detections under controlled lighting transformations."""

import argparse
import csv
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from model_utils import load_interpreter, run_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT / "models" / "safewalk_ssd_mobilenet_v2_edgetpu.tflite"
)
DEFAULT_LABELS = PROJECT_ROOT / "models" / "labels.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare detections under normal, bright, and low light."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def load_labels(path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def lighting_variants(image):
    pixels = image.astype(np.float32)
    return [
        ("normal", image.copy()),
        ("bright", np.clip(pixels * 1.25 + 35, 0, 255).astype(np.uint8)),
        ("low_light", np.clip(pixels * 0.35, 0, 255).astype(np.uint8)),
    ]


def mean_brightness(image):
    return float(np.mean(image))


def annotate(image, detections):
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = "{} {:.1f}%".format(
            detection["class"],
            detection["confidence"] * 100.0,
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), (60, 220, 90), 3)
        cv2.putText(
            output,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 220, 90),
            2,
            cv2.LINE_AA,
        )
    return output


def detect(interpreter, image, labels, confidence, runs):
    latencies = []
    detections = []
    for _ in range(runs):
        started = time.perf_counter()
        detections = run_inference(
            interpreter=interpreter,
            frame=image,
            labels=labels,
            confidence_threshold=confidence,
            iou_threshold=0.45,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
    return detections, float(np.mean(latencies))


def main():
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    for path in (args.image, args.model, args.labels):
        if not path.exists():
            raise FileNotFoundError(str(path))

    source = cv2.imread(str(args.image))
    if source is None:
        raise RuntimeError("Unable to read image: {}".format(args.image))

    labels = load_labels(args.labels)
    interpreter = load_interpreter(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Warm the Edge TPU parameter cache before collecting measurements.
    run_inference(
        interpreter=interpreter,
        frame=source,
        labels=labels,
        confidence_threshold=args.confidence,
        iou_threshold=0.45,
    )

    rows = []
    for scenario, image in lighting_variants(source):
        detections, latency_ms = detect(
            interpreter,
            image,
            labels,
            args.confidence,
            args.runs,
        )
        confidences = [item["confidence"] for item in detections]
        classes = sorted(set(item["class"] for item in detections))

        input_path = args.output_dir / "{}_input.jpg".format(scenario)
        output_path = args.output_dir / "{}_detected.jpg".format(scenario)
        cv2.imwrite(str(input_path), image)
        cv2.imwrite(str(output_path), annotate(image, detections))

        rows.append(
            {
                "scenario": scenario,
                "mean_brightness": mean_brightness(image),
                "detections": len(detections),
                "classes": ", ".join(classes),
                "average_confidence_pct": (
                    float(np.mean(confidences)) * 100.0
                    if confidences
                    else 0.0
                ),
                "maximum_confidence_pct": (
                    float(np.max(confidences)) * 100.0
                    if confidences
                    else 0.0
                ),
                "average_pipeline_latency_ms": latency_ms,
            }
        )

    csv_path = args.output_dir / "lighting_comparison.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("SafeWalk Controlled Lighting Evaluation")
    print("---------------------------------------")
    print(
        "{:<12} {:>10} {:>11} {:>11} {:>12}".format(
            "Scenario", "Brightness", "Detections", "Avg conf", "Latency"
        )
    )
    for row in rows:
        print(
            "{:<12} {:>10.1f} {:>11d} {:>10.2f}% {:>10.3f} ms".format(
                row["scenario"],
                row["mean_brightness"],
                row["detections"],
                row["average_confidence_pct"],
                row["average_pipeline_latency_ms"],
            )
        )
        print("  Classes: {}".format(row["classes"] or "none"))
    print("CSV saved: {}".format(csv_path))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("Lighting evaluation failed: {}".format(error), file=sys.stderr)
        sys.exit(1)
