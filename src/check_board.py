#!/usr/bin/env python3
"""Check the Dev Board Mini runtime before starting SafeWalk."""

from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "safewalk_edgetpu.tflite"


def check(label, action):
    try:
        detail = action()
        print("[PASS] {}{}".format(label, ": " + detail if detail else ""))
        return True
    except Exception as error:
        print("[FAIL] {}: {}".format(label, error))
        return False


def main():
    results = []
    results.append(
        check(
            "Python",
            lambda: sys.version.split()[0],
        )
    )
    results.append(
        check(
            "OpenCV",
            lambda: __import__("cv2").__version__,
        )
    )
    results.append(
        check(
            "PyCoral",
            lambda: __import__("pycoral").__file__,
        )
    )
    results.append(
        check(
            "Audio",
            lambda: shutil.which("espeak")
            or (_ for _ in ()).throw(RuntimeError("espeak not installed")),
        )
    )
    results.append(
        check(
            "Compiled model",
            lambda: str(MODEL_PATH)
            if MODEL_PATH.exists()
            else (_ for _ in ()).throw(
                FileNotFoundError(str(MODEL_PATH))
            ),
        )
    )

    if MODEL_PATH.exists():
        def allocate_model():
            from pycoral.utils.edgetpu import make_interpreter

            interpreter = make_interpreter(str(MODEL_PATH))
            interpreter.allocate_tensors()
            return "Edge TPU interpreter allocated"

        results.append(check("Edge TPU", allocate_model))

    if all(results):
        print("\nSafeWalk board check passed.")
        return 0

    print("\nOne or more checks failed. See docs/DEPLOYMENT.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

