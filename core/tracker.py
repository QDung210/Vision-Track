import cv2
import numpy as np
from math import hypot

from core.kalman_filter import KalmanFilter
from utils.geometry import Bbox, box_center


_MAX_FAILURES      = 8
_MAX_PREDICT_FRAMES = 20

# Tracking state labels
TRACKING  = "tracking"
LOST      = "lost"
PREDICTED = "predicted"
RECOVERED = "recovered"
CROWDED   = "crowded"


def _create_cv_tracker(algorithm: str) -> cv2.Tracker:
    if algorithm == "CSRT":
        return cv2.TrackerCSRT_create()
    if algorithm == "KCF":
        return cv2.TrackerKCF_create()
    raise ValueError(f"Unsupported algorithm: {algorithm!r}")


class ObjectTracker:
    """
    Wraps an OpenCV tracker with a Kalman filter for smooth,
    failure-resilient single-object tracking.
    """

    def __init__(
        self,
        track_id: int,
        algorithm: str,
        frame: np.ndarray,
        bbox: Bbox,
        process_noise: float = 1.0,
        meas_noise: float = 6.0,
    ) -> None:
        self.id        = track_id
        self.algorithm = algorithm
        self.bbox      = bbox
        self.status    = TRACKING
        self.speed     = 0.0
        self.fail_count    = 0
        self.predict_count = 0

        cx, cy = box_center(bbox)
        self._prev_center: tuple[float, float] = (cx, cy)

        self.kalman = KalmanFilter()
        self.kalman.init_state(cx, cy)
        self.kalman.set_process_noise(process_noise)
        self.kalman.set_measurement_noise(meas_noise)

        self._cv_tracker = _create_cv_tracker(algorithm)
        self._cv_tracker.init(frame, bbox)

    def update(self, frame: np.ndarray) -> bool:
        ok, raw_box = self._cv_tracker.update(frame)

        if ok:
            self.bbox = tuple(int(v) for v in raw_box)  # type: ignore[assignment]
            cx, cy    = box_center(self.bbox)
            self.speed = hypot(cx - self._prev_center[0], cy - self._prev_center[1])

            self.kalman.set_process_noise(self.speed)
            self.kalman.update(cx, cy)
            self.kalman.predict()

            was_lost = self.fail_count > 0
            self.fail_count    = 0
            self.predict_count = 0
            self._prev_center  = (cx, cy)
            self.status = RECOVERED if was_lost else TRACKING
            return True

        # ── Tracker failed ──────────────────────────────────────────
        self.fail_count += 1
        if self.fail_count >= _MAX_FAILURES:
            # Reinitialise tracker at predicted position
            self._cv_tracker = _create_cv_tracker(self.algorithm)
            self._cv_tracker.init(frame, self.bbox)
            self.fail_count = 0

        px, py = self.kalman.predict()
        w, h   = self.bbox[2], self.bbox[3]
        self.bbox = (int(px - w / 2), int(py - h / 2), w, h)
        self.predict_count += 1
        self.status = LOST if self.predict_count > _MAX_PREDICT_FRAMES else PREDICTED
        return False

    def set_noise(self, process_noise: float, meas_noise: float) -> None:
        self.kalman.set_process_noise(process_noise)
        self.kalman.set_measurement_noise(meas_noise)

    @property
    def kalman_center(self) -> tuple[float, float]:
        return float(self.kalman.state[0, 0]), float(self.kalman.state[1, 0])
