# Models

Place the exported Edge TPU model here with this exact name:

```text
models/safewalk_edgetpu.tflite
```

The filename must retain the `_edgetpu.tflite` suffix so runtimes can recognize
that the model targets the Edge TPU.

Do not copy `best.pt` to the Coral board. It is the PyTorch training checkpoint,
not the board runtime model.

The `.gitignore` excludes `.tflite` files by default to prevent accidental
commits before export is verified. To add the finished model intentionally:

```bash
git add -f models/safewalk_edgetpu.tflite
git commit -m "Add compiled SafeWalk Edge TPU model"
git push
```

