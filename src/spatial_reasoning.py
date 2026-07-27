"""Position, distance, and hazard-priority helpers for SafeWalk."""


DISTANCE_SCORES = {
    "far": 0.25,
    "medium": 0.65,
    "near": 1.0,
}

POSITION_SCORES = {
    "left": 0.65,
    "centre": 1.0,
    "right": 0.65,
}


def object_position(box, frame_width):
    """Return left, centre, or right from an xyxy bounding box."""
    x1, _, x2, _ = box
    center_x = (x1 + x2) / 2.0

    if center_x < frame_width / 3.0:
        return "left"
    if center_x < frame_width * 2.0 / 3.0:
        return "centre"
    return "right"


def relative_distance(
    box,
    frame_width,
    frame_height,
    near_area_ratio=0.20,
    medium_area_ratio=0.08,
):
    """Estimate near, medium, or far from bounding-box area."""
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = max(1.0, float(frame_width * frame_height))
    ratio = box_area / frame_area

    if ratio >= near_area_ratio:
        return "near"
    if ratio >= medium_area_ratio:
        return "medium"
    return "far"


def hazard_score(detection, danger_weights=None):
    """Score a detection by class danger, distance, position, and confidence."""
    danger_weights = danger_weights or {}
    class_weight = danger_weights.get(detection["class"], 0.55)
    distance_weight = DISTANCE_SCORES[detection["distance"]]
    position_weight = POSITION_SCORES[detection["position"]]
    confidence = detection["confidence"]

    return (
        0.35 * class_weight
        + 0.35 * distance_weight
        + 0.20 * position_weight
        + 0.10 * confidence
    )


def select_highest_risk(detections, danger_weights=None, silent_classes=None):
    """Return the highest-priority warning candidate."""
    silent_classes = set(silent_classes or [])
    candidates = [
        detection
        for detection in detections
        if detection["class"] not in silent_classes
    ]

    if not candidates:
        return None

    for detection in candidates:
        detection["hazard_score"] = hazard_score(
            detection,
            danger_weights=danger_weights,
        )

    return max(candidates, key=lambda item: item["hazard_score"])

