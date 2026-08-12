"""Board-compatible non-maximum suppression for object detections."""

import numpy as np


def non_max_suppression(boxes, scores, class_ids, iou_threshold):
    """Return score-sorted indices after class-aware box suppression."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    class_ids = np.asarray(class_ids)

    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)

    selected = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]

        while order.size:
            current = int(order[0])
            selected.append(current)
            if order.size == 1:
                break

            remaining = order[1:]
            current_box = boxes[current]
            other_boxes = boxes[remaining]

            overlap_x1 = np.maximum(current_box[0], other_boxes[:, 0])
            overlap_y1 = np.maximum(current_box[1], other_boxes[:, 1])
            overlap_x2 = np.minimum(current_box[2], other_boxes[:, 2])
            overlap_y2 = np.minimum(current_box[3], other_boxes[:, 3])

            overlap_width = np.maximum(0.0, overlap_x2 - overlap_x1)
            overlap_height = np.maximum(0.0, overlap_y2 - overlap_y1)
            intersection = overlap_width * overlap_height

            current_area = max(
                0.0,
                (current_box[2] - current_box[0])
                * (current_box[3] - current_box[1]),
            )
            other_areas = np.maximum(
                0.0,
                (other_boxes[:, 2] - other_boxes[:, 0])
                * (other_boxes[:, 3] - other_boxes[:, 1]),
            )
            union = current_area + other_areas - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[iou <= iou_threshold]

    return np.asarray(
        sorted(selected, key=lambda index: scores[index], reverse=True),
        dtype=np.int64,
    )
