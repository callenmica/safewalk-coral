#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Updating Mendel Linux packages..."
sudo apt-get update

echo "Installing camera and audio dependencies..."
sudo apt-get install -y git python3-opencv espeak alsa-utils

if ! python3 -c "import pycoral" >/dev/null 2>&1; then
  echo "PyCoral is missing. Installing the Mendel package..."
  sudo apt-get install -y python3-pycoral
fi

chmod +x "${PROJECT_DIR}/scripts/run_safewalk.sh"
chmod +x "${PROJECT_DIR}/src/coral_inference.py"
chmod +x "${PROJECT_DIR}/src/check_board.py"

echo
echo "Board dependencies are ready."
echo "Next: place safewalk_edgetpu.tflite in ${PROJECT_DIR}/models/"
echo "Then run: python3 src/check_board.py"

