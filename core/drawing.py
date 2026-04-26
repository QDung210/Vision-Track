import cv2
import numpy as np

from core.tracker import ObjectTracker

# BGR versions of theme colors (OpenCV uses BGR, not RGB)
_COLOR_MAP = {
    "tracking":  (16,  185, 129),   # green
    "recovered": (217,  70, 239),   # magenta
    "predicted": ( 6,  182, 212),   # cyan
    "crowded":   (245, 158,  11),   # yellow
    "lost":      (239,  68,  68),   # red
}
_WHITE = (240, 240, 255)
_BLUE  = ( 29, 106, 229)


def annotate_frame(
    frame: np.ndarray,
    trackers: list[ObjectTracker],
    fps: float,
    algorithm: str,
) -> np.ndarray:
    out = frame.copy()

    for t in trackers:
        color = _COLOR_MAP.get(t.status, _WHITE)
        x, y, w, h = t.bbox

        # Bounding box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        # Kalman center dot
        kx, ky = t.kalman_center
        cv2.circle(out, (int(kx), int(ky)), 4, _BLUE, -1)

        # ID + status label
        label = f"#{t.id} {t.status.upper()}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x, y - lh - 6), (x + lw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Speed
        spd_txt = f"{t.speed:.1f}px/f"
        cv2.putText(out, spd_txt, (x + 2, y + h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    # HUD overlay (top-left)
    hud = [
        f"ALG : {algorithm}",
        f"FPS : {fps:.1f}",
        f"OBJ : {len(trackers)}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(out, line, (10, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, _WHITE, 1, cv2.LINE_AA)

    return out


def draw_roi_preview(
    frame: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    current: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()

    for (x, y, w, h) in rois:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), _BLUE, -1)
        cv2.rectangle(out,     (x, y), (x + w, y + h), _BLUE, 2)
        cv2.putText(out, f"ROI #{rois.index((x,y,w,h))+1}", (x+4, y+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if current:
        x, y, w, h = current
        cv2.rectangle(out, (x, y), (x + w, y + h), (6, 182, 212), 2)

    cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

    guide = "Drag to draw ROI  |  Right-click: undo  |  SPACE: confirm  |  ESC: cancel"
    cv2.putText(out, guide, (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1, cv2.LINE_AA)
    return out
