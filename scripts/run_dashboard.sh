#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_DIR}"
exec python3 src/coral_dashboard.py \
  --model models/safewalk_ssd_mobilenet_v2_edgetpu.tflite \
  --labels models/labels.txt \
  --config config/config.json \
  "$@"
