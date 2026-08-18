# SafeWalk Coral

SafeWalk is an offline obstacle-detection and audio-guidance prototype for the
Coral Dev Board Mini. A USB camera supplies frames, the on-board Edge TPU runs
the object detector, and the application announces the highest-priority hazard.

## Deployment model

The original YOLOv8n model remains the project baseline. Its partial Edge TPU
mapping made it unsuitable for responsive deployment, so the optimized runtime
uses a custom SSD MobileNet V2 detector trained on the same 25-class ROD
dataset. The SSD model uses a 300 x 300 UINT8 input and maps 99 of 102
operations to one Edge TPU subgraph.

The Coral Edge TPU cannot run the PyTorch `best.pt` checkpoint directly. The
deployment artifact is the fully quantized and compiled model at:

```text
models/safewalk_ssd_mobilenet_v2_edgetpu.tflite
```

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

Verify the board and optimized model:

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

Run a controlled still-image detection and save an annotated result:

```bash
python3 src/detect_image.py \
  --image test_images/benchmark.jpg \
  --output results/benchmark_detected.jpg
```

Create controlled normal, bright, and low-light variants of the same image and
save annotated results plus a CSV summary:

```bash
python3 src/evaluate_lighting.py \
  --image test_images/benchmark.jpg \
  --output-dir results/lighting
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

# Use an iPhone or other direct HTTP/RTSP video stream.
bash scripts/run_safewalk.sh --camera "http://192.168.1.50:8080/video"

# Disable spoken warnings.
bash scripts/run_safewalk.sh --no-audio

# Use a different confidence threshold.
bash scripts/run_safewalk.sh --confidence 0.45
```

## Coral dashboard

The enhanced dashboard runs the real compiled TFLite model on the Edge TPU:

```bash
bash scripts/run_dashboard.sh --camera "http://192.168.1.50:8080/video"
```

The dashboard includes measured inference latency and frame rate, two-hazard
spoken warnings, a spatial radar and threat level, escape-route guidance,
head-level and ground-level hazard classification, collision growth tracking,
safe-zone highlighting, low-light alerts, and a live latency graph.

The camera value may be a local index such as `1` or a direct HTTP/RTSP stream
URL. For a Wi-Fi camera, keep the iPhone streaming app open and connect the
iPhone and Coral board to the same network. Resolution and FPS for a network
stream are controlled in the iPhone app.

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
- Wi-Fi video can lag or disconnect; SafeWalk remains a prototype and should
  not be relied on as a sole mobility aid.
- SafeWalk's near/medium/far values are bounding-box estimates, not physical
  distance measurements.

## References

- [Coral Dev Board Mini setup](https://coral.ai/docs/dev-board-mini/get-started/)
- [Coral Edge TPU model requirements](https://coral.ai/docs/edgetpu/models-intro/)
- [Ultralytics Edge TPU export](https://docs.ultralytics.com/integrations/edge-tpu/)
