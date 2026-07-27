# Edge TPU compiler report

## Export configuration

- Source checkpoint: SafeWalk YOLOv8n `best.pt`
- Input size: 320 x 320
- Quantization: full integer INT8
- Calibration fraction: 1% of the ROD training dataset
- Edge TPU Compiler: 16.0.384591198
- Compiled model: `models/safewalk_edgetpu.tflite`
- Compiled model size: 3,346,653 bytes
- SHA-256:
  `1d476a80c70809dcec054b90834eb9aec826ef18f7ad53936ff80228bdd11b75`

## Mapping summary

The compiler produced one Edge TPU subgraph. Based on the operation table:

- 106 operations were mapped to the Edge TPU.
- 149 operations were not mapped to the Edge TPU.
- Approximately 41.6% of operations mapped by operation count.
- 27 of 64 `CONV_2D` operations mapped to the Edge TPU.

The compiler reported `More than one subgraph is not supported` for most
unmapped operations. Two `TRANSPOSE` operations were also not mapped because
of an unspecified compiler limitation.

Operation count is not an exact measure of runtime because different operations
have different computational costs. Actual camera FPS and latency must be
measured on the Dev Board Mini.

## Deployment decision

This model is suitable for the first SafeWalk deployment test. It will execute
as a mixed Edge TPU and CPU graph, so performance may be lower than a model that
maps completely to the Edge TPU.

If measured performance is not sufficient, the recommended fallback is a
Coral-native object detector such as EfficientDet-Lite, followed by retraining
or class adaptation for the SafeWalk dataset.
