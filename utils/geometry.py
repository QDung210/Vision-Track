from typing import Tuple

Bbox = Tuple[int, int, int, int]  # (x, y, w, h)


def box_center(bbox: Bbox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def compute_iou(b1: Bbox, b2: Bbox) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def box_to_xyxy(bbox: Bbox) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def clamp_bbox(bbox: Bbox, frame_w: int, frame_h: int) -> Bbox:
    x, y, w, h = bbox
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h
