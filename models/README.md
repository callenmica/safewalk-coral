# Models

SafeWalk keeps the original YOLOv8 Edge TPU export as a baseline and the
SSD MobileNet V2 export as the Coral-optimized deployment model:

```text
models/safewalk_edgetpu.tflite                       # YOLOv8 baseline
models/safewalk_ssd_mobilenet_v2_int8_300.tflite    # uncompiled CPU benchmark
models/safewalk_ssd_mobilenet_v2_edgetpu.tflite     # optimized deployment
```

The optimized SSD MobileNet V2 model uses a 300 x 300 UINT8 input. Its compiler
report mapped 99 of 102 operations to one Edge TPU subgraph. The remaining
three CPU operations perform output dequantization and SSD post-processing.

The compiled filenames retain the `_edgetpu.tflite` suffix to make it clear
that they target the Edge TPU. The uncompiled INT8 model provides a controlled
CPU baseline using the same architecture, weights, input image, and input size.

Do not copy `best.pt` to the Coral board. It is the PyTorch training checkpoint,
not the board runtime model.

The `.gitignore` excludes `.tflite` files by default to prevent accidental
commits before export is verified. To add a finished model intentionally:

```bash
git add -f models/safewalk_ssd_mobilenet_v2_edgetpu.tflite
git commit -m "Add compiled SafeWalk Edge TPU model"
git push
```
