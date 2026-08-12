# Coral Dev Board Mini deployment

## 1. Prepare the board

The Dev Board Mini ships with Mendel Linux and is normally controlled
headlessly. Follow the official setup guide to connect power, connect through
MDT or SSH, join Wi-Fi, and update Mendel:

```bash
sudo apt-get update
sudo apt-get dist-upgrade
sudo reboot now
```

Use a stable 5 V / 2 A supply. Shut down with `sudo shutdown now`; do not remove
power while Mendel is running.

## 2. Export the YOLO model

The Edge TPU accepts TensorFlow Lite models that are fully INT8 quantized and
compiled for the Edge TPU. The PyTorch `best.pt` file does not meet that
requirement.

Run `notebooks/export_edgetpu_colab.ipynb` on Google Colab. The export uses the
ROD dataset as representative calibration data and should produce a file whose
name ends in `_edgetpu.tflite`.

Read the compiler output. If a substantial part of the graph remains on the
CPU, real-time performance may be below the project target. Keep the compiler
log for the project report.

## 3. Add the compiled model

On the Mac:

```bash
git clone git@github.com:callenmica/safewalk-coral.git
cd safewalk-coral
cp ~/Downloads/*_edgetpu.tflite models/safewalk_edgetpu.tflite
git add -f models/safewalk_edgetpu.tflite
git commit -m "Add compiled SafeWalk Edge TPU model"
git push
```

The `_edgetpu.tflite` suffix is intentional and must be preserved.

## 4. Clone onto the board

From the Dev Board Mini shell:

```bash
git clone https://github.com/callenmica/safewalk-coral.git
cd safewalk-coral
bash scripts/setup_board.sh
python3 src/check_board.py
```

## 5. Test the camera

Connect a UVC-compatible USB webcam and check:

```bash
ls -l /dev/video*
```

If the camera is `/dev/video1`, use `--camera 1`.

To use an iPhone instead, start an HTTP MJPEG or RTSP camera-server app and
connect the phone and board to the same Wi-Fi. Use the app's direct video URL:

```bash
bash scripts/run_dashboard.sh --camera "http://192.168.1.50:8080/video"
```

If the app displays a browser control page, find its direct MJPEG/RTSP stream
URL; an HTML page cannot be decoded as video.

## 6. Test audio

Connect earphones or a supported speaker, then run:

```bash
espeak "SafeWalk audio test"
```

If no sound is heard:

```bash
aplay -l
alsamixer
```

## 7. Start SafeWalk

Headless mode:

```bash
bash scripts/run_safewalk.sh
```

HDMI preview:

```bash
bash scripts/run_safewalk.sh --display
```

The application announces one prioritized warning at a time and applies a
cooldown so speech does not repeat every frame.

Enhanced Coral dashboard with two-hazard speech and spatial guidance:

```bash
bash scripts/run_dashboard.sh --camera 0
```

## Troubleshooting

### The model file is missing

Confirm this exact path exists:

```bash
ls -lh models/safewalk_edgetpu.tflite
```

### Edge TPU allocation fails

The model was not compiled correctly, has the wrong filename, or the board's
Mendel/Edge TPU packages are outdated. Re-run the Colab export and board update.

### Unsupported YOLO output shape

Keep the error text and the export log. The exported graph layout differs from
the raw YOLOv8 detection output expected by `src/model_utils.py`.

### Detection is slow

Review the Edge TPU compiler output. Unsupported operations run on the CPU and
can sharply reduce FPS. A Coral-native detector such as EfficientDet-Lite is
the fallback if YOLOv8 cannot be mapped sufficiently to the TPU.
