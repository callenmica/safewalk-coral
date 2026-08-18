#!/usr/bin/env python3
"""Benchmark pure TensorFlow Lite inference on CPU or Edge TPU."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from model_utils import letterbox, quantize_input


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a controlled still-image inference benchmark."
    )
    parser.add_argument("--model", required=True, help="Path to a TFLite model.")
    parser.add_argument("--image", required=True, help="Path to the test image.")
    parser.add_argument(
        "--device",
        choices=("cpu", "edgetpu"),
        required=True,
        help="Interpreter backend to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--output",
        help="CSV output path. Defaults to benchmark_<device>.csv.",
    )
    return parser.parse_args()


def load_interpreter(model_path, device):
    if device == "edgetpu":
        from pycoral.utils.edgetpu import make_interpreter

        return make_interpreter(str(model_path))

    from tflite_runtime.interpreter import Interpreter

    return Interpreter(model_path=str(model_path))


def prepare_input(interpreter, image):
    input_detail = interpreter.get_input_details()[0]
    input_shape = input_detail["shape"]
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])

    padded, _, _, _ = letterbox(image, input_width, input_height)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = quantize_input(rgb, input_detail)
    interpreter.set_tensor(input_detail["index"], tensor)
    return tuple(int(value) for value in input_shape)


def percentile(values, percentage):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentage))


def main():
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output or "benchmark_{}.csv".format(args.device))

    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("--warmup must be non-negative and --runs must be positive.")
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Unable to read test image: {}".format(image_path))

    interpreter = load_interpreter(model_path, args.device)
    interpreter.allocate_tensors()
    input_shape = prepare_input(interpreter, image)

    for _ in range(args.warmup):
        interpreter.invoke()

    latencies_ms = []
    for _ in range(args.runs):
        started = time.perf_counter()
        interpreter.invoke()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    average_ms = float(np.mean(latencies_ms))
    minimum_ms = float(np.min(latencies_ms))
    maximum_ms = float(np.max(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p95_ms = percentile(latencies_ms, 95)
    fps = 1000.0 / average_ms if average_ms > 0 else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("run", "device", "latency_ms"))
        for run_number, latency_ms in enumerate(latencies_ms, start=1):
            writer.writerow((run_number, args.device, "{:.6f}".format(latency_ms)))

    print("\nSafeWalk Controlled Inference Benchmark")
    print("---------------------------------------")
    print("Timestamp:     {}".format(datetime.now().isoformat(timespec="seconds")))
    print("Device:        {}".format(args.device))
    print("Model:         {}".format(model_path))
    print("Image:         {}".format(image_path))
    print("Input shape:   {}".format(input_shape))
    print("Warm-up runs:  {}".format(args.warmup))
    print("Timed runs:    {}".format(args.runs))
    print("Timing scope:  interpreter.invoke() only")
    print("Average:       {:.3f} ms".format(average_ms))
    print("Minimum:       {:.3f} ms".format(minimum_ms))
    print("Maximum:       {:.3f} ms".format(maximum_ms))
    print("Median:        {:.3f} ms".format(median_ms))
    print("95th pct:      {:.3f} ms".format(p95_ms))
    print("Throughput:    {:.2f} FPS".format(fps))
    print("CSV saved:     {}".format(output_path))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("Benchmark failed: {}".format(error), file=sys.stderr)
        sys.exit(1)
