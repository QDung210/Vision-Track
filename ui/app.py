"""
VisionTrack – main application window.

Layout (1280 × 800 minimum):
  ┌──────────────────────────────────────────┐
  │            HEADER BAR                    │
  ├────────┬─────────────────────┬───────────┤
  │ LEFT   │     VIDEO CANVAS    │  RIGHT    │
  │ PANEL  │  (expands freely)   │  PANEL    │
  ├────────┴─────────────────────┴───────────┤
  │            STATUS BAR                    │
  └──────────────────────────────────────────┘
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog
from typing import Any

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from core.drawing import annotate_frame, draw_roi_preview
from core.engine import TrackingEngine
from utils.geometry import clamp_bbox
from ui.theme import (
    BG, SURFACE, SURFACE2, BORDER,
    BLUE, BLUE_DARK, CYAN, GREEN, RED, YELLOW, TEXT, TEXT_MUTED,
    TRACK_COLORS,
    FONT_TITLE, FONT_HEADING, FONT_LABEL, FONT_MONO,
    FONT_METRIC, FONT_BTN, FONT_MONO_SM,
    PANEL_LEFT, PANEL_RIGHT, CORNER_R, CARD_PAD,
    setup_theme,
)


# ─────────────────────────────────────────────────────────────────────────────
# App state machine
# ─────────────────────────────────────────────────────────────────────────────

class _S:
    IDLE       = "idle"
    READY      = "ready"
    ROI_SELECT = "roi_select"
    TRACKING   = "tracking"
    DONE       = "done"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _card(parent: Any, title: str | None = None, **kw) -> ctk.CTkFrame:
    frame = ctk.CTkFrame(
        parent,
        fg_color=SURFACE2,
        border_color=BORDER,
        border_width=1,
        corner_radius=CORNER_R,
        **kw,
    )
    if title:
        ctk.CTkLabel(
            frame,
            text=title.upper(),
            font=("Segoe UI", 9, "bold"),
            text_color=TEXT_MUTED,
            anchor="w",
        ).pack(anchor="w", padx=CARD_PAD, pady=(CARD_PAD, 4))
    return frame


def _divider(parent: Any) -> None:
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=8, pady=6)


# ─────────────────────────────────────────────────────────────────────────────
# Metric card widget
# ─────────────────────────────────────────────────────────────────────────────

class MetricCard(ctk.CTkFrame):
    def __init__(self, parent: Any, label: str, unit: str = "", color: str = CYAN) -> None:
        super().__init__(parent, fg_color=SURFACE, border_color=BORDER,
                         border_width=1, corner_radius=CORNER_R)
        ctk.CTkLabel(self, text=label, font=FONT_MONO_SM,
                     text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(8, 0))
        self._var = tk.StringVar(value="—")
        ctk.CTkLabel(self, textvariable=self._var, font=FONT_METRIC,
                     text_color=color).pack(anchor="w", padx=10)
        if unit:
            ctk.CTkLabel(self, text=unit, font=FONT_MONO_SM,
                         text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(0, 8))
        else:
            ctk.CTkLabel(self, text="", font=FONT_MONO_SM).pack(pady=(0, 6))

    def set(self, value: str) -> None:
        self._var.set(value)


# ─────────────────────────────────────────────────────────────────────────────
# Status badge
# ─────────────────────────────────────────────────────────────────────────────

_BADGE_COLORS = {
    _S.IDLE:       (TEXT_MUTED, SURFACE2),
    _S.READY:      (BLUE,       SURFACE2),
    _S.ROI_SELECT: (CYAN,       SURFACE2),
    _S.TRACKING:   (GREEN,      SURFACE2),
    _S.DONE:       (TEXT_MUTED, SURFACE2),
}
_BADGE_LABELS = {
    _S.IDLE:       "● IDLE",
    _S.READY:      "● READY",
    _S.ROI_SELECT: "● SELECT ROI",
    _S.TRACKING:   "● TRACKING",
    _S.DONE:       "● DONE",
}


# ─────────────────────────────────────────────────────────────────────────────
# Background tracking thread
# ─────────────────────────────────────────────────────────────────────────────

class _TrackingThread(threading.Thread):
    def __init__(
        self,
        video_path: str,
        engine: TrackingEngine,
        frame_q: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self._path      = video_path
        self._engine    = engine
        self._frame_q   = frame_q
        self._stop      = stop_event

    def run(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            return

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            trackers = self._engine.update(frame)
            annotated = annotate_frame(frame, trackers,
                                       self._engine.fps,
                                       self._engine.algorithm)

            # Drop frame if queue full (non-blocking) to stay real-time
            try:
                self._frame_q.put_nowait(annotated)
            except queue.Full:
                self._engine.frame_skip += 1

        cap.release()
        try:
            self._frame_q.put_nowait(None)   # sentinel → done
        except queue.Full:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class VisionTrackApp(ctk.CTk):
    def __init__(self) -> None:
        setup_theme()
        super().__init__()

        self._state      = _S.IDLE
        self._video_path = ""
        self._engine     = TrackingEngine()
        self._thread: _TrackingThread | None = None
        self._stop_event = threading.Event()
        self._frame_q: queue.Queue = queue.Queue(maxsize=4)

        # Video properties
        self._first_frame: np.ndarray | None = None
        self._cap_w = 640
        self._cap_h = 480

        # ROI drawing
        self._rois: list[tuple[int, int, int, int]] = []
        self._roi_start: tuple[int, int] | None = None
        self._roi_cur:   tuple[int, int, int, int] | None = None

        # Keep PhotoImage alive
        self._photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._refresh_state()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.title("VisionTrack")
        self.geometry("1280x800")
        self.minsize(1000, 680)
        self.configure(fg_color=BG)

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_header()
        self._build_content()
        self._build_statusbar()

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        hdr = ctk.CTkFrame(self, fg_color=SURFACE, border_color=BORDER,
                           border_width=1, corner_radius=0, height=60)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)

        # Logo dot + title
        dot = ctk.CTkLabel(hdr, text="◈", font=("Segoe UI", 24, "bold"),
                           text_color=BLUE, width=40)
        dot.grid(row=0, column=0, padx=(16, 4), pady=10)

        title_frame = ctk.CTkFrame(hdr, fg_color="transparent")
        title_frame.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(title_frame, text="VisionTrack",
                     font=FONT_TITLE, text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(title_frame, text="CSRT / KCF + Kalman Filter Tracking",
                     font=("Segoe UI", 10), text_color=TEXT_MUTED).pack(anchor="w")

        # Status badge (right side)
        self._badge_var = tk.StringVar(value=_BADGE_LABELS[_S.IDLE])
        self._badge_lbl = ctk.CTkLabel(
            hdr, textvariable=self._badge_var,
            font=("Consolas", 11, "bold"),
            text_color=TEXT_MUTED,
            fg_color=SURFACE2,
            corner_radius=12,
            padx=14, pady=4,
        )
        self._badge_lbl.grid(row=0, column=2, padx=16, pady=10)

    # ── Content area ─────────────────────────────────────────────────────────

    def _build_content(self) -> None:
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        self._build_left_panel(content)
        self._build_canvas_area(content)
        self._build_right_panel(content)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self, parent: Any) -> None:
        lp = ctk.CTkScrollableFrame(
            parent, width=PANEL_LEFT,
            fg_color=SURFACE, border_color=BORDER, border_width=1,
            corner_radius=0, scrollbar_button_color=BORDER,
        )
        lp.grid(row=0, column=0, sticky="nsew")

        # ── Video file ─────────────────────────────────────
        vc = _card(lp, "Video Source")
        vc.pack(fill="x", padx=8, pady=(8, 4))

        self._path_var = tk.StringVar(value="No file selected")
        path_entry = ctk.CTkEntry(
            vc, textvariable=self._path_var,
            state="readonly", fg_color=SURFACE, border_color=BORDER,
            text_color=TEXT_MUTED, font=FONT_MONO_SM,
        )
        path_entry.pack(fill="x", padx=CARD_PAD, pady=(0, 6))
        ctk.CTkButton(
            vc, text="Browse…", font=FONT_BTN,
            fg_color=BLUE, hover_color=BLUE_DARK,
            corner_radius=CORNER_R, command=self._browse,
        ).pack(fill="x", padx=CARD_PAD, pady=(0, CARD_PAD))

        # ── Algorithm selector ─────────────────────────────
        ac = _card(lp, "Algorithm")
        ac.pack(fill="x", padx=8, pady=4)

        self._algo_var = tk.StringVar(value="CSRT")
        algo_row = ctk.CTkFrame(ac, fg_color="transparent")
        algo_row.pack(fill="x", padx=CARD_PAD, pady=(0, CARD_PAD))
        algo_row.grid_columnconfigure((0, 1), weight=1)

        for col, algo in enumerate(("CSRT", "KCF")):
            btn = ctk.CTkButton(
                algo_row, text=algo, font=FONT_BTN,
                corner_radius=CORNER_R,
                command=lambda a=algo: self._select_algo(a),
            )
            btn.grid(row=0, column=col, padx=(0 if col else 0, 4 if col == 0 else 0), sticky="ew")
            setattr(self, f"_btn_{algo.lower()}", btn)
        self._select_algo("CSRT")  # set initial visual state

        # ── Kalman filter ──────────────────────────────────
        kc = _card(lp, "Kalman Filter")
        kc.pack(fill="x", padx=8, pady=4)

        self._proc_noise_var = tk.DoubleVar(value=1.0)
        self._meas_noise_var = tk.DoubleVar(value=6.0)

        for label, var, lo, hi in [
            ("Process Noise", self._proc_noise_var, 0.1, 20.0),
            ("Measurement Noise", self._meas_noise_var, 0.1, 20.0),
        ]:
            row = ctk.CTkFrame(kc, fg_color="transparent")
            row.pack(fill="x", padx=CARD_PAD, pady=(0, 4))
            top = ctk.CTkFrame(row, fg_color="transparent")
            top.pack(fill="x")
            ctk.CTkLabel(top, text=label, font=FONT_LABEL,
                         text_color=TEXT).pack(side="left")
            val_lbl = ctk.CTkLabel(top, textvariable=var, font=FONT_MONO_SM,
                                   text_color=CYAN, width=36)
            val_lbl.pack(side="right")
            ctk.CTkSlider(
                row, from_=lo, to=hi, variable=var,
                button_color=BLUE, progress_color=BLUE,
                command=lambda _v: self._on_noise_change(),
            ).pack(fill="x", pady=(2, 0))

        ctk.CTkLabel(kc, text="").pack(pady=2)  # bottom spacing

        # ── Controls ───────────────────────────────────────
        cc = _card(lp, "Controls")
        cc.pack(fill="x", padx=8, pady=(4, 8))

        self._btn_start = ctk.CTkButton(
            cc, text="▶  Start", font=FONT_BTN,
            fg_color=GREEN, hover_color="#0d9e6e",
            corner_radius=CORNER_R, command=self._on_start,
        )
        self._btn_start.pack(fill="x", padx=CARD_PAD, pady=(0, 6))

        self._btn_stop = ctk.CTkButton(
            cc, text="■  Stop", font=FONT_BTN,
            fg_color=RED, hover_color="#b91c1c",
            corner_radius=CORNER_R, command=self._on_stop,
        )
        self._btn_stop.pack(fill="x", padx=CARD_PAD, pady=(0, 6))

        self._btn_reset = ctk.CTkButton(
            cc, text="↺  Reset", font=FONT_BTN,
            fg_color=SURFACE2, hover_color=BORDER,
            border_color=BORDER, border_width=1,
            corner_radius=CORNER_R, command=self._on_reset,
        )
        self._btn_reset.pack(fill="x", padx=CARD_PAD, pady=(0, CARD_PAD))

    # ── Canvas area ───────────────────────────────────────────────────────────

    def _build_canvas_area(self, parent: Any) -> None:
        container = ctk.CTkFrame(parent, fg_color=BG, corner_radius=0)
        container.grid(row=0, column=1, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            container, bg="#060d1a",
            highlightthickness=1, highlightbackground=BORDER,
        )
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Draw placeholder text
        self._draw_placeholder()

        # Mouse events for ROI drawing
        self._canvas.bind("<ButtonPress-1>",   self._roi_press)
        self._canvas.bind("<B1-Motion>",        self._roi_drag)
        self._canvas.bind("<ButtonRelease-1>",  self._roi_release)
        self._canvas.bind("<Button-3>",         self._roi_undo)
        self._canvas.bind("<Configure>",        self._on_canvas_resize)

    def _draw_placeholder(self) -> None:
        self._canvas.delete("placeholder")
        w = self._canvas.winfo_width()  or 640
        h = self._canvas.winfo_height() or 480
        cx, cy = w // 2, h // 2
        self._canvas.create_text(
            cx, cy - 16,
            text="◈  VisionTrack",
            fill=BLUE, font=("Segoe UI", 22, "bold"),
            tags="placeholder",
        )
        self._canvas.create_text(
            cx, cy + 20,
            text="Browse a video file, then click  ▶ Start",
            fill=TEXT_MUTED, font=("Segoe UI", 12),
            tags="placeholder",
        )

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self, parent: Any) -> None:
        rp = ctk.CTkScrollableFrame(
            parent, width=PANEL_RIGHT,
            fg_color=SURFACE, border_color=BORDER, border_width=1,
            corner_radius=0, scrollbar_button_color=BORDER,
        )
        rp.grid(row=0, column=2, sticky="nsew")

        ctk.CTkLabel(rp, text="METRICS", font=("Segoe UI", 9, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(
            anchor="w", padx=10, pady=(10, 4))

        # 2 × 2 metric grid
        grid = ctk.CTkFrame(rp, fg_color="transparent")
        grid.pack(fill="x", padx=6, pady=2)
        grid.grid_columnconfigure((0, 1), weight=1)

        self._mc_fps   = MetricCard(grid, "FPS",     "fps",   CYAN)
        self._mc_obj   = MetricCard(grid, "OBJECTS",  "objs",  BLUE)
        self._mc_speed = MetricCard(grid, "MAX SPD",  "px/f",  YELLOW)
        self._mc_skip  = MetricCard(grid, "SKIPPED",  "frms",  RED)

        for i, mc in enumerate((self._mc_fps, self._mc_obj,
                                 self._mc_speed, self._mc_skip)):
            mc.grid(row=i // 2, column=i % 2, sticky="ew",
                    padx=3, pady=3)

        _divider(rp)

        # Object list header
        ctk.CTkLabel(rp, text="TRACKED OBJECTS", font=("Segoe UI", 9, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(
            anchor="w", padx=10, pady=(2, 4))

        self._obj_list_frame = ctk.CTkFrame(rp, fg_color="transparent")
        self._obj_list_frame.pack(fill="x", padx=6)

        _divider(rp)

        # Legend
        ctk.CTkLabel(rp, text="LEGEND", font=("Segoe UI", 9, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(
            anchor="w", padx=10, pady=(2, 4))

        legend_data = [
            ("tracking",  "Tracking OK"),
            ("predicted", "Predicted"),
            ("crowded",   "Crowded"),
            ("recovered", "Recovered"),
            ("lost",      "Lost"),
        ]
        for state, label in legend_data:
            row = ctk.CTkFrame(rp, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)
            dot = ctk.CTkLabel(row, text="●", font=("Segoe UI", 12),
                               text_color=TRACK_COLORS[state], width=20)
            dot.pack(side="left")
            ctk.CTkLabel(row, text=label, font=FONT_MONO_SM,
                         text_color=TEXT).pack(side="left", padx=4)

        ctk.CTkLabel(rp, text="").pack(pady=4)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self) -> None:
        sb = ctk.CTkFrame(self, fg_color=SURFACE, border_color=BORDER,
                          border_width=1, corner_radius=0, height=30)
        sb.grid(row=2, column=0, sticky="ew")
        sb.grid_propagate(False)

        self._status_var = tk.StringVar(value="Ready — browse a video to begin.")
        ctk.CTkLabel(sb, textvariable=self._status_var,
                     font=FONT_MONO_SM, text_color=TEXT_MUTED,
                     anchor="w").pack(side="left", padx=12, pady=4)

    # ── State machine helpers ─────────────────────────────────────────────────

    def _refresh_state(self) -> None:
        s = self._state

        badge_txt = _BADGE_LABELS.get(s, "●")
        badge_col = _BADGE_COLORS.get(s, (TEXT_MUTED, SURFACE2))[0]
        self._badge_var.set(badge_txt)
        self._badge_lbl.configure(text_color=badge_col)

        is_idle      = s == _S.IDLE
        is_ready     = s == _S.READY
        is_roi       = s == _S.ROI_SELECT
        is_tracking  = s == _S.TRACKING
        is_done      = s == _S.DONE

        can_start = is_ready or is_done
        can_stop  = is_tracking or is_roi
        can_reset = not is_idle

        self._btn_start.configure(state="normal" if can_start else "disabled")
        self._btn_stop.configure(state="normal"  if can_stop  else "disabled")
        self._btn_reset.configure(state="normal" if can_reset else "disabled")

    def _set_state(self, state: str, status_msg: str = "") -> None:
        self._state = state
        self._refresh_state()
        if status_msg:
            self._status_var.set(status_msg)

    # ── Algorithm toggle ──────────────────────────────────────────────────────

    def _select_algo(self, algo: str) -> None:
        self._algo_var.set(algo)
        self._engine.set_algorithm(algo)
        selected   = dict(fg_color=BLUE,   hover_color=BLUE_DARK, text_color=TEXT)
        unselected = dict(fg_color=SURFACE2, hover_color=BORDER,
                          border_color=BORDER, text_color=TEXT_MUTED)
        for name in ("csrt", "kcf"):
            btn = getattr(self, f"_btn_{name}")
            btn.configure(**(selected if name.upper() == algo else unselected))

    # ── Noise change ──────────────────────────────────────────────────────────

    def _on_noise_change(self) -> None:
        self._engine.set_noise(
            round(self._proc_noise_var.get(), 1),
            round(self._meas_noise_var.get(), 1),
        )

    # ── Browse ────────────────────────────────────────────────────────────────

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")],
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._status_var.set(f"Cannot open: {path}")
            cap.release()
            return

        ok, frame = cap.read()
        self._cap_h, self._cap_w = (frame.shape[:2] if ok else (480, 640))
        cap.release()

        if ok:
            self._first_frame = frame.copy()

        self._video_path = path
        short = path.split("/")[-1]
        self._path_var.set(short)
        self._set_state(_S.READY, f"Loaded: {short}  —  Click ▶ Start to select ROIs")

    # ── ROI drawing (on canvas) ───────────────────────────────────────────────

    def _roi_press(self, event: tk.Event) -> None:
        if self._state != _S.ROI_SELECT:
            return
        self._roi_start = (event.x, event.y)

    def _roi_drag(self, event: tk.Event) -> None:
        if self._state != _S.ROI_SELECT or not self._roi_start:
            return
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        self._roi_cur = (x, y, w, h)
        self._redraw_roi_frame()

    def _roi_release(self, event: tk.Event) -> None:
        if self._state != _S.ROI_SELECT or not self._roi_start:
            return
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w > 10 and h > 10:
            # Convert canvas coords → frame coords
            sx, sy = self._canvas_to_frame_scale()
            bx = int(min(x0, x1) * sx)
            by = int(min(y0, y1) * sy)
            bw, bh = int(w * sx), int(h * sy)
            bbox = clamp_bbox((bx, by, bw, bh), self._cap_w, self._cap_h)
            self._rois.append(bbox)
            n = len(self._rois)
            self._status_var.set(
                f"{n} ROI(s) drawn — right-click to undo, SPACE to start, ESC to cancel"
            )
        self._roi_start = None
        self._roi_cur   = None
        self._redraw_roi_frame()

    def _roi_undo(self, _event: tk.Event) -> None:
        if self._state == _S.ROI_SELECT and self._rois:
            self._rois.pop()
            self._redraw_roi_frame()

    def _redraw_roi_frame(self) -> None:
        if self._first_frame is None:
            return
        preview = draw_roi_preview(self._first_frame, self._rois, self._roi_cur)
        self._show_frame(preview)

    def _canvas_to_frame_scale(self) -> tuple[float, float]:
        cw = self._canvas.winfo_width()  or self._cap_w
        ch = self._canvas.winfo_height() or self._cap_h
        # Compute displayed size (letterbox)
        scale = min(cw / self._cap_w, ch / self._cap_h)
        dw    = self._cap_w * scale
        dh    = self._cap_h * scale
        ox    = (cw - dw) / 2
        oy    = (ch - dh) / 2
        # The canvas coords include offset; return scale for inside region
        return self._cap_w / dw, self._cap_h / dh

    # ── Key bindings for ROI phase ────────────────────────────────────────────

    def _bind_roi_keys(self) -> None:
        self.bind("<space>", self._roi_confirm)
        self.bind("<Escape>", self._roi_cancel)

    def _unbind_roi_keys(self) -> None:
        self.unbind("<space>")
        self.unbind("<Escape>")

    def _roi_confirm(self, _event: tk.Event | None = None) -> None:
        if self._state != _S.ROI_SELECT:
            return
        if not self._rois:
            self._status_var.set("Draw at least one ROI first.")
            return
        self._unbind_roi_keys()
        self._start_tracking()

    def _roi_cancel(self, _event: tk.Event | None = None) -> None:
        if self._state != _S.ROI_SELECT:
            return
        self._unbind_roi_keys()
        self._rois.clear()
        self._set_state(_S.READY, "ROI selection cancelled.")
        if self._first_frame is not None:
            self._show_frame(self._first_frame)

    # ── Start / Stop / Reset ─────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._state == _S.DONE:
            self._on_reset()
            return
        if self._state != _S.READY or not self._first_frame is not None:
            return
        self._rois.clear()
        self._roi_cur = None
        self._set_state(_S.ROI_SELECT, "Draw ROI(s) on the video — SPACE to confirm, ESC to cancel")
        self._show_frame(self._first_frame)
        self._bind_roi_keys()

    def _on_stop(self) -> None:
        if self._state == _S.ROI_SELECT:
            self._roi_cancel()
            return
        self._stop_event.set()
        self._set_state(_S.DONE, "Stopped by user.")

    def _on_reset(self) -> None:
        self._stop_event.set()
        self._engine.clear()
        self._rois.clear()
        self._first_frame = None
        self._video_path  = ""
        self._path_var.set("No file selected")
        self._set_state(_S.IDLE, "Ready — browse a video to begin.")
        self._draw_placeholder()
        self._mc_fps.set("—")
        self._mc_obj.set("—")
        self._mc_speed.set("—")
        self._mc_skip.set("—")
        for w in self._obj_list_frame.winfo_children():
            w.destroy()

    def _start_tracking(self) -> None:
        if not self._rois or self._first_frame is None:
            return

        self._engine.clear()
        for bbox in self._rois:
            self._engine.add_tracker(self._first_frame, bbox)
        self._engine.set_noise(
            round(self._proc_noise_var.get(), 1),
            round(self._meas_noise_var.get(), 1),
        )

        self._stop_event.clear()
        self._frame_q.queue.clear()

        self._thread = _TrackingThread(
            self._video_path, self._engine, self._frame_q, self._stop_event
        )
        self._thread.start()
        self._set_state(_S.TRACKING,
                        f"Tracking {len(self._rois)} object(s) — ESC/Stop to end")
        self._poll_frames()

    # ── Frame polling (UI thread) ─────────────────────────────────────────────

    def _poll_frames(self) -> None:
        if self._state != _S.TRACKING:
            return
        try:
            frame = self._frame_q.get_nowait()
        except queue.Empty:
            self.after(15, self._poll_frames)
            return

        if frame is None:
            self._set_state(_S.DONE, "Video finished.")
            self._update_metrics()
            return

        self._show_frame(frame)
        self._update_metrics()
        self.after(1, self._poll_frames)

    # ── Frame display ─────────────────────────────────────────────────────────

    def _show_frame(self, frame: np.ndarray) -> None:
        cw = max(self._canvas.winfo_width(),  1)
        ch = max(self._canvas.winfo_height(), 1)

        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)

        self._canvas.delete("all")
        self._canvas.create_image(ox, oy, anchor="nw", image=self._photo)

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._state == _S.IDLE:
            self._draw_placeholder()
        elif self._state == _S.ROI_SELECT and self._first_frame is not None:
            self._redraw_roi_frame()

    # ── Metrics update ────────────────────────────────────────────────────────

    def _update_metrics(self) -> None:
        e = self._engine
        self._mc_fps.set(f"{e.fps:.1f}")
        self._mc_obj.set(str(e.object_count))
        self._mc_speed.set(f"{e.max_speed:.1f}")
        self._mc_skip.set(str(e.frame_skip))

        # Rebuild object list
        for child in self._obj_list_frame.winfo_children():
            child.destroy()

        for t in e.trackers:
            color = TRACK_COLORS.get(t.status, TEXT_MUTED)
            row = ctk.CTkFrame(self._obj_list_frame, fg_color=SURFACE,
                               border_color=color, border_width=1,
                               corner_radius=6)
            row.pack(fill="x", pady=2)

            ctk.CTkLabel(row, text=f"#{t.id}", font=("Consolas", 11, "bold"),
                         text_color=color, width=28).pack(side="left", padx=6, pady=4)
            ctk.CTkLabel(row, text=t.status, font=FONT_MONO_SM,
                         text_color=color).pack(side="left")
            ctk.CTkLabel(row, text=f"{t.speed:.1f}px/f", font=FONT_MONO_SM,
                         text_color=TEXT_MUTED).pack(side="right", padx=6)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop_event.set()
        self.destroy()
