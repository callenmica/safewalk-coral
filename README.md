# SafeWalk Coral

SafeWalk is an offline obstacle-detection and audio-guidance prototype for the
Coral Dev Board Mini. A USB camera supplies frames, the on-board Edge TPU runs
the object detector, and the application announces the highest-priority hazard.

## Important model requirement

The Coral Edge TPU cannot run the PyTorch `best.pt` checkpoint directly. Export
it on Google Colab as a fully quantized Edge TPU model first:

1. Open `notebooks/export_edgetpu_colab.ipynb` in Google Colab.
2. Upload `best.pt` when prompted.
3. Make sure the ROD dataset and `data.yaml` are available for INT8 calibration.
4. Run all cells and download the generated file.
5. Rename the output to `safewalk_edgetpu.tflite` while keeping the
   `_edgetpu.tflite` suffix.
6. Place it in `models/` and commit it to this repository.

Edge TPU export must run on x86 Linux. Google Colab is suitable; the ARM-based
Dev Board Mini is not an export machine.

## Dev Board Mini setup

The board should be running the latest available Mendel Linux release and be
connected to Wi-Fi.

```bash
git clone https://github.com/callenmica/safewalk-coral.git
cd safewalk-coral
bash scripts/setup_board.sh
```

Copy the exported model into `models/safewalk_edgetpu.tflite`, then verify the
board:

```bash
python3 src/check_board.py
```

Connect a USB webcam and check that it appears:

```bash
ls /dev/video*
```

Run SafeWalk:

```bash
bash scripts/run_safewalk.sh
```

For a connected HDMI display:

```bash
bash scripts/run_safewalk.sh --display
```

Press `Ctrl+C` to stop the headless application, or `Q` in the display window.

## Runtime options

```bash
python3 src/coral_inference.py --help
```

Common examples:

```bash
# Use a second camera.
bash scripts/run_safewalk.sh --camera 1

# Disable spoken warnings.
bash scripts/run_safewalk.sh --no-audio

# Use a different confidence threshold.
bash scripts/run_safewalk.sh --confidence 0.45
```

## Repository layout

```text
config/       Runtime thresholds and class danger weights
docs/         Export and board deployment guide
models/       Labels and the exported Edge TPU model
notebooks/    Google Colab Edge TPU export notebook
scripts/      Board setup and launcher scripts
src/          Coral inference and guidance modules
tests/        Hardware-independent unit tests
```

## Hardware notes

- Use a stable 5 V / 2 A USB-C supply.
- The Dev Board Mini is normally used headlessly through MDT or SSH.
- Connect earphones or a speaker supported by Mendel Linux/ALSA.
- SafeWalk's near/medium/far values are bounding-box estimates, not physical
  distance measurements.

## References

- [Coral Dev Board Mini setup](https://coral.ai/docs/dev-board-mini/get-started/)
- [Coral Edge TPU model requirements](https://coral.ai/docs/edgetpu/models-intro/)
- [Ultralytics Edge TPU export](https://docs.ultralytics.com/integrations/edge-tpu/)

