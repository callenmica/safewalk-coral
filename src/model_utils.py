"""Edge TPU preprocessing and object-detection output decoding."""

import cv2
import numpy as np

from nms import non_max_suppression


def load_interpreter(model_path):
    from pycoral.utils.edgetpu import make_interpreter

    interpreter = make_interpreter(str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def letterbox(frame, target_width, target_height):
    source_height, source_width = frame.shape[:2]
    scale = min(
        target_width / float(source_width),
        target_height / float(source_height),
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    output = np.full(
        (target_height, target_width, 3),
        114,
        dtype=np.uint8,
    )
    output[
        pad_y : pad_y + resized_height,
        pad_x : pad_x + resized_width,
    ] = resized
    return output, scale, pad_x, pad_y


def quantize_input(rgb_image, input_detail):
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        image = rgb_image.astype(np.float32) / 255.0
        return image[np.newaxis, ...]

    scale, zero_point = input_detail["quantization"]
    if scale <= 0:
        raise ValueError("Model input tensor has invalid quantization metadata.")

    limits = np.iinfo(dtype)
    real_min = (limits.min - zero_point) * scale
    real_max = (limits.max - zero_point) * scale

    if real_min <= -0.9 and real_max >= 0.9:
        image = rgb_image.astype(np.float32) / 127.5 - 1.0
    else:
        image = rgb_image.astype(np.float32) / 255.0

    quantized = np.round(image / scale + zero_point)
    quantized = np.clip(quantized, limits.min, limits.max).astype(dtype)
    return quantized[np.newaxis, ...]


def dequantize_output(tensor, output_detail):
    if np.issubdtype(tensor.dtype, np.floating):
        return tensor.astype(np.float32)

    scale, zero_point = output_detail["quantization"]
    if scale <= 0:
        return tensor.astype(np.float32)
    return (tensor.astype(np.float32) - zero_point) * scale


def _raw_prediction_output(interpreter):
    candidates = []
    for detail in interpreter.get_output_details():
        tensor = interpreter.get_tensor(detail["index"])
        tensor = dequantize_output(tensor, detail)
        candidates.append((tensor.size, tensor))

    if not candidates:
        raise RuntimeError("The model returned no output tensors.")

    return max(candidates, key=lambda item: item[0])[1]


def model_family(interpreter):
    details = interpreter.get_output_details()
    if len(details) == 4:
        has_boxes = any(
            len(detail["shape"]) >= 2 and int(detail["shape"][-1]) == 4
            for detail in details
        )
        has_count = any(int(np.prod(detail["shape"])) == 1 for detail in details)
        if has_boxes and has_count:
            return "ssd"
    return "yolov8"


def _ssd_output_tensors(interpreter):
    outputs = []
    for detail in interpreter.get_output_details():
        tensor = interpreter.get_tensor(detail["index"])
        tensor = dequantize_output(tensor, detail)
        outputs.append((detail, tensor))

    boxes = None
    count = None
    vectors = []
    for detail, tensor in outputs:
        shape = tensor.shape
        if tensor.ndim >= 2 and shape[-1] == 4:
            boxes = tensor
        elif tensor.size == 1:
            count = tensor
        else:
            vectors.append((detail, tensor))

    if boxes is None or count is None or len(vectors) != 2:
        raise RuntimeError("Unsupported SSD output tensor layout.")

    classes = None
    scores = None
    for detail, tensor in vectors:
        name = detail.get("name", "").lower()
        if "class" in name or name.endswith(":1"):
            classes = tensor
        elif "score" in name or name.endswith(":2"):
            scores = tensor

    if classes is None or scores is None:
        first = vectors[0][1]
        second = vectors[1][1]
        first_integer_error = float(np.mean(np.abs(first - np.round(first))))
        second_integer_error = float(np.mean(np.abs(second - np.round(second))))
        if first_integer_error <= second_integer_error:
            classes, scores = first, second
        else:
            classes, scores = second, first

    return boxes, classes, scores, count


def decode_ssd(
    boxes,
    class_ids,
    scores,
    num_detections,
    labels,
    confidence_threshold,
    source_width,
    source_height,
):
    boxes = np.asarray(boxes).reshape(-1, 4)
    class_ids = np.asarray(class_ids).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    count = min(
        int(round(float(np.asarray(num_detections).reshape(-1)[0]))),
        len(boxes),
        len(class_ids),
        len(scores),
    )

    detections = []
    for index in range(max(0, count)):
        confidence = float(scores[index])
        if confidence < confidence_threshold:
            continue

        class_id = int(round(float(class_ids[index])))
        if class_id < 0 or class_id >= len(labels):
            continue

        ymin, xmin, ymax, xmax = boxes[index]
        x1 = max(0, min(source_width - 1, int(round(xmin * source_width))))
        y1 = max(0, min(source_height - 1, int(round(ymin * source_height))))
        x2 = max(0, min(source_width - 1, int(round(xmax * source_width))))
        y2 = max(0, min(source_height - 1, int(round(ymax * source_height))))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "class_id": class_id,
                "class": labels[class_id],
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
            }
        )

    return detections


def _xywh_to_xyxy(boxes):
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def decode_yolov8(
    output,
    labels,
    confidence_threshold,
    iou_threshold,
    input_width,
    input_height,
    source_width,
    source_height,
    scale,
    pad_x,
    pad_y,
):
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        raise RuntimeError(
            "Unsupported YOLO output shape: {}".format(output.shape)
        )

    expected_columns = 4 + len(labels)
    if predictions.shape[0] == expected_columns:
        predictions = predictions.T
    elif predictions.shape[1] != expected_columns:
        raise RuntimeError(
            "Expected {} YOLO values per prediction, received shape {}.".format(
                expected_columns,
                predictions.shape,
            )
        )

    boxes_xywh = predictions[:, :4].copy()
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[
        np.arange(class_scores.shape[0]),
        class_ids,
    ]

    keep = confidences >= confidence_threshold
    if not np.any(keep):
        return []

    boxes_xywh = boxes_xywh[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    # Some TFLite exports use normalized box coordinates.
    if boxes_xywh.size and np.max(np.abs(boxes_xywh)) <= 2.0:
        boxes_xywh[:, [0, 2]] *= input_width
        boxes_xywh[:, [1, 3]] *= input_height

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    selected = non_max_suppression(
        boxes_xyxy,
        confidences,
        class_ids,
        iou_threshold,
    )

    detections = []
    for index in selected:
        box = boxes_xyxy[int(index)]
        x1 = int(round((box[0] - pad_x) / scale))
        y1 = int(round((box[1] - pad_y) / scale))
        x2 = int(round((box[2] - pad_x) / scale))
        y2 = int(round((box[3] - pad_y) / scale))

        x1 = max(0, min(source_width - 1, x1))
        y1 = max(0, min(source_height - 1, y1))
        x2 = max(0, min(source_width - 1, x2))
        y2 = max(0, min(source_height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(class_ids[int(index)])
        detections.append(
            {
                "class_id": class_id,
                "class": labels[class_id],
                "confidence": float(confidences[int(index)]),
                "box": (x1, y1, x2, y2),
            }
        )

    return detections


def run_inference(
    interpreter,
    frame,
    labels,
    confidence_threshold,
    iou_threshold,
):
    input_detail = interpreter.get_input_details()[0]
    input_shape = input_detail["shape"]
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])
    source_height, source_width = frame.shape[:2]

    family = model_family(interpreter)
    if family == "ssd":
        resized = cv2.resize(
            frame,
            (input_width, input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = quantize_input(rgb, input_detail)
        interpreter.set_tensor(input_detail["index"], input_tensor)
        interpreter.invoke()
        boxes, class_ids, scores, count = _ssd_output_tensors(interpreter)
        return decode_ssd(
            boxes=boxes,
            class_ids=class_ids,
            scores=scores,
            num_detections=count,
            labels=labels,
            confidence_threshold=confidence_threshold,
            source_width=source_width,
            source_height=source_height,
        )

    padded, scale, pad_x, pad_y = letterbox(
        frame,
        input_width,
        input_height,
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    input_tensor = quantize_input(rgb, input_detail)
    interpreter.set_tensor(input_detail["index"], input_tensor)
    interpreter.invoke()
    output = _raw_prediction_output(interpreter)

    return decode_yolov8(
        output=output,
        labels=labels,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        input_width=input_width,
        input_height=input_height,
        source_width=source_width,
        source_height=source_height,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
