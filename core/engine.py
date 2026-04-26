import time
import numpy as np

from core.tracker import ObjectTracker, TRACKING, CROWDED
from utils.geometry import Bbox, compute_iou

_CROWDING_IOU = 0.10
_FPS_ALPHA    = 0.2


class TrackingEngine:
    """
    Manages multiple ObjectTrackers, computes per-frame metrics,
    and detects crowding between tracked objects.
    """

    def __init__(self, algorithm: str = "CSRT") -> None:
        self.algorithm     = algorithm
        self.trackers:  list[ObjectTracker] = []
        self.fps            = 0.0
        self.frame_skip     = 0
        self._next_id       = 0
        self._process_noise = 1.0
        self._meas_noise    = 6.0

    # ── Configuration ─────────────────────────────────────────────────

    def set_algorithm(self, algorithm: str) -> None:
        self.algorithm = algorithm

    def set_noise(self, process_noise: float, meas_noise: float) -> None:
        self._process_noise = process_noise
        self._meas_noise    = meas_noise
        for t in self.trackers:
            t.set_noise(process_noise, meas_noise)

    # ── Tracker management ────────────────────────────────────────────

    def add_tracker(self, frame: np.ndarray, bbox: Bbox) -> None:
        t = ObjectTracker(
            self._next_id,
            self.algorithm,
            frame,
            bbox,
            self._process_noise,
            self._meas_noise,
        )
        self.trackers.append(t)
        self._next_id += 1

    def clear(self) -> None:
        self.trackers.clear()
        self._next_id   = 0
        self.fps        = 0.0
        self.frame_skip = 0

    # ── Per-frame update ──────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> list[ObjectTracker]:
        if not self.trackers:
            return []

        t0 = time.perf_counter()

        for tracker in self.trackers:
            tracker.update(frame)

        self._resolve_crowding()

        elapsed = time.perf_counter() - t0
        inst_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self.fps = _FPS_ALPHA * inst_fps + (1 - _FPS_ALPHA) * self.fps

        return self.trackers

    def _resolve_crowding(self) -> None:
        n = len(self.trackers)
        for i in range(n):
            for j in range(i + 1, n):
                if compute_iou(self.trackers[i].bbox, self.trackers[j].bbox) > _CROWDING_IOU:
                    if self.trackers[i].status == TRACKING:
                        self.trackers[i].status = CROWDED
                    if self.trackers[j].status == TRACKING:
                        self.trackers[j].status = CROWDED

    # ── Metrics ───────────────────────────────────────────────────────

    @property
    def max_speed(self) -> float:
        return max((t.speed for t in self.trackers), default=0.0)

    @property
    def object_count(self) -> int:
        return len(self.trackers)
