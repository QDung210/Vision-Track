"""
VisionTrack – main application window.

  ┌──────────────────────────────────────────┐
  │            HEADER BAR                    │
  ├────────┬─────────────────────┬───────────┤
  │ LEFT   │     VIDEO CANVAS    │  RIGHT    │
  │ PANEL  │                     │  PANEL    │
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
    BG, SURFACE, SURFACE2, BORDER, TEXT, TEXT_MUTED,
    BLUE, BLUE_DARK, CYAN, GREEN, RED, YELLOW,
    TRACK_COLORS, CANVAS_BG,
    FONT_TITLE, FONT_LABEL, FONT_MONO, FONT_METRIC, FONT_BTN, FONT_MONO_SM,
    PANEL_LEFT, PANEL_RIGHT, CORNER_R, CARD_PAD,
    setup_theme,
)


# ─────────────────────────────────────────────────────────────────────────────
# App states
# ─────────────────────────────────────────────────────────────────────────────

class _S:
    IDLE       = "idle"
    READY      = "ready"
    ROI_SELECT = "roi_select"
    TRACKING   = "tracking"
    DONE       = "done"


_BADGE_LABELS = {
    _S.IDLE:       "● IDLE",
    _S.READY:      "● READY",
    _S.ROI_SELECT: "● SELECT ROI",
    _S.TRACKING:   "● TRACKING",
    _S.DONE:       "● DONE",
}
_BADGE_COLORS = {
    _S.IDLE:       TEXT_MUTED,
    _S.READY:      BLUE,
    _S.ROI_SELECT: CYAN,
    _S.TRACKING:   GREEN,
    _S.DONE:       TEXT_MUTED,
}


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
            frame, text=title.upper(),
            font=("Segoe UI", 9, "bold"), text_color=TEXT_MUTED, anchor="w",
        ).pack(anchor="w", padx=CARD_PAD, pady=(CARD_PAD, 4))
    return frame


def _divider(parent: Any) -> None:
    ctk.CTkFrame(parent, height=1, fg_color=BORDER, corner_radius=0).pack(
        fill="x", padx=8, pady=6
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric card
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
        ctk.CTkLabel(self, text=unit if unit else " ", font=FONT_MONO_SM,
                     text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(0, 8))

    def set(self, value: str) -> None:
        self._var.set(value)


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
        done_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self._path  = video_path
        self._engine = engine
        self._q     = frame_q
        self._stop  = stop_event
        self._done  = done_event

    def run(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self._done.set()
            return

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            trackers  = self._engine.update(frame)
            annotated = annotate_frame(frame, trackers,
                                       self._engine.fps, self._engine.algorithm)
            try:
                self._q.put_nowait(annotated)
            except queue.Full:
                self._engine.frame_skip += 1

        cap.release()
        self._done.set()   # always signal, even if stopped early


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class VisionTrackApp(ctk.CTk):
    def __init__(self) -> None:
        self._dark_mode = True
        setup_theme(dark=True)
        super().__init__()

        # ── App state ─────────────────────────────────────────────────────
        self._state      = _S.IDLE
        self._video_path = ""
        self._engine     = TrackingEngine()
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._frame_q: queue.Queue = queue.Queue(maxsize=8)
        self._thread: _TrackingThread | None = None

        # ── Video metadata ─────────────────────────────────────────────────
        self._first_frame: np.ndarray | None = None
        self._cap_w = 640
        self._cap_h = 480

        # ── ROI drawing ────────────────────────────────────────────────────
        self._rois: list[tuple[int, int, int, int]] = []
        self._roi_start: tuple[int, int] | None = None

        # ── Keep PhotoImage alive (GC protection) ──────────────────────────
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

        ctk.CTkLabel(hdr, text="◈", font=("Segoe UI", 24, "bold"),
                     text_color=BLUE, width=40).grid(row=0, column=0, padx=(16, 4))

        title_box = ctk.CTkFrame(hdr, fg_color="transparent")
        title_box.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(title_box, text="VisionTrack",
                     font=FONT_TITLE, text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(title_box, text="CSRT / KCF + Kalman Filter Tracking",
                     font=("Segoe UI", 10), text_color=TEXT_MUTED).pack(anchor="w")

        # Right side: theme toggle + status badge
        right_box = ctk.CTkFrame(hdr, fg_color="transparent")
        right_box.grid(row=0, column=2, padx=16)

        self._theme_btn = ctk.CTkButton(
            right_box, text="☀ Light", width=90,
            font=("Segoe UI", 11), fg_color=SURFACE2,
            hover_color=BORDER, text_color=TEXT_MUTED,
            border_color=BORDER, border_width=1,
            corner_radius=CORNER_R,
            command=self._toggle_theme,
        )
        self._theme_btn.pack(side="left", padx=(0, 10))

        self._badge_var = tk.StringVar(value=_BADGE_LABELS[_S.IDLE])
        self._badge_lbl = ctk.CTkLabel(
            right_box, textvariable=self._badge_var,
            font=("Consolas", 11, "bold"),
            text_color=TEXT_MUTED,
            fg_color=SURFACE2,
            corner_radius=12,
            padx=14, pady=4,
        )
        self._badge_lbl.pack(side="left")

    # ── Content ───────────────────────────────────────────────────────────────

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

        # Video source
        vc = _card(lp, "Video Source")
        vc.pack(fill="x", padx=8, pady=(8, 4))
        self._path_var = tk.StringVar(value="No file selected")
        ctk.CTkEntry(
            vc, textvariable=self._path_var, state="readonly",
            fg_color=SURFACE, border_color=BORDER,
            text_color=TEXT_MUTED, font=FONT_MONO_SM,
        ).pack(fill="x", padx=CARD_PAD, pady=(0, 6))
        ctk.CTkButton(
            vc, text="Browse…", font=FONT_BTN,
            fg_color=BLUE, hover_color=BLUE_DARK,
            corner_radius=CORNER_R, command=self._browse,
        ).pack(fill="x", padx=CARD_PAD, pady=(0, CARD_PAD))

        # Algorithm
        ac = _card(lp, "Algorithm")
        ac.pack(fill="x", padx=8, pady=4)
        algo_row = ctk.CTkFrame(ac, fg_color="transparent")
        algo_row.pack(fill="x", padx=CARD_PAD, pady=(0, CARD_PAD))
        algo_row.grid_columnconfigure((0, 1), weight=1)
        for col, algo in enumerate(("CSRT", "KCF")):
            btn = ctk.CTkButton(
                algo_row, text=algo, font=FONT_BTN,
                corner_radius=CORNER_R,
                command=lambda a=algo: self._select_algo(a),
            )
            btn.grid(row=0, column=col, padx=(0, 4) if col == 0 else (0, 0), sticky="ew")
            setattr(self, f"_btn_{algo.lower()}", btn)
        self._select_algo("CSRT")

        # Kalman
        kc = _card(lp, "Kalman Filter")
        kc.pack(fill="x", padx=8, pady=4)
        self._proc_noise_var = tk.DoubleVar(value=1.0)
        self._meas_noise_var = tk.DoubleVar(value=6.0)
        for label, var, lo, hi in [
            ("Process Noise",     self._proc_noise_var, 0.1, 20.0),
            ("Measurement Noise", self._meas_noise_var, 0.1, 20.0),
        ]:
            row = ctk.CTkFrame(kc, fg_color="transparent")
            row.pack(fill="x", padx=CARD_PAD, pady=(0, 4))
            top = ctk.CTkFrame(row, fg_color="transparent")
            top.pack(fill="x")
            ctk.CTkLabel(top, text=label, font=FONT_LABEL, text_color=TEXT).pack(side="left")
            ctk.CTkLabel(top, textvariable=var, font=FONT_MONO_SM,
                         text_color=CYAN, width=36).pack(side="right")
            ctk.CTkSlider(
                row, from_=lo, to=hi, variable=var,
                button_color=BLUE, progress_color=BLUE,
                command=lambda _v: self._on_noise_change(),
            ).pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(kc, text="").pack(pady=2)

        # Controls
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
            container,
            bg=CANVAS_BG["dark"],
            highlightthickness=1,
            highlightbackground="#1a2d4d",
        )
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self._canvas.bind("<ButtonPress-1>",  self._roi_press)
        self._canvas.bind("<B1-Motion>",       self._roi_drag)
        self._canvas.bind("<ButtonRelease-1>", self._roi_release)
        self._canvas.bind("<Button-3>",        self._roi_undo)
        self._canvas.bind("<Configure>",       self._on_canvas_resize)

        self.after(100, self._draw_placeholder)

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self, parent: Any) -> None:
        rp = ctk.CTkScrollableFrame(
            parent, width=PANEL_RIGHT,
            fg_color=SURFACE, border_color=BORDER, border_width=1,
            corner_radius=0, scrollbar_button_color=BORDER,
        )
        rp.grid(row=0, column=2, sticky="nsew")

        ctk.CTkLabel(rp, text="METRICS", font=("Segoe UI", 9, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(anchor="w", padx=10, pady=(10, 4))

        grid = ctk.CTkFrame(rp, fg_color="transparent")
        grid.pack(fill="x", padx=6, pady=2)
        grid.grid_columnconfigure((0, 1), weight=1)

        self._mc_fps   = MetricCard(grid, "FPS",     "fps",  CYAN)
        self._mc_obj   = MetricCard(grid, "OBJECTS", "objs", BLUE)
        self._mc_speed = MetricCard(grid, "MAX SPD", "px/f", YELLOW)
        self._mc_skip  = MetricCard(grid, "SKIPPED", "frms", RED)
        for i, mc in enumerate((self._mc_fps, self._mc_obj, self._mc_speed, self._mc_skip)):
            mc.grid(row=i // 2, column=i % 2, sticky="ew", padx=3, pady=3)

        _divider(rp)
        ctk.CTkLabel(rp, text="TRACKED OBJECTS", font=("Segoe UI", 9, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(anchor="w", padx=10, pady=(2, 4))
        self._obj_list_frame = ctk.CTkFrame(rp, fg_color="transparent")
        self._obj_list_frame.pack(fill="x", padx=6)

        _divider(rp)
        ctk.CTkLabel(rp, text="LEGEND", font=("Segoe UI", 9, "bold"),
                     text_color=TEXT_MUTED, anchor="w").pack(anchor="w", padx=10, pady=(2, 4))
        for state, label in [
            ("tracking",  "Tracking OK"),
            ("predicted", "Predicted"),
            ("crowded",   "Crowded"),
            ("recovered", "Recovered"),
            ("lost",      "Lost"),
        ]:
            row = ctk.CTkFrame(rp, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)
            ctk.CTkLabel(row, text="●", font=("Segoe UI", 12),
                         text_color=TRACK_COLORS[state], width=20).pack(side="left")
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

    # ── Theme toggle ──────────────────────────────────────────────────────────

    def _toggle_theme(self) -> None:
        self._dark_mode = not self._dark_mode
        mode = "dark" if self._dark_mode else "light"
        ctk.set_appearance_mode(mode)
        # Update elements that CTk can't auto-update (plain tk widgets)
        self._canvas.configure(bg=CANVAS_BG[mode])
        self._theme_btn.configure(text="☀ Light" if self._dark_mode else "◑ Dark")
        # Redraw placeholder if needed
        if self._state == _S.IDLE:
            self._draw_placeholder()

    # ── State machine ─────────────────────────────────────────────────────────

    def _refresh_state(self) -> None:
        s = self._state
        self._badge_var.set(_BADGE_LABELS.get(s, "●"))
        self._badge_lbl.configure(text_color=_BADGE_COLORS.get(s, TEXT_MUTED))

        can_start = s in (_S.READY, _S.DONE)
        can_stop  = s in (_S.TRACKING, _S.ROI_SELECT)
        can_reset = s != _S.IDLE

        self._btn_start.configure(state="normal" if can_start else "disabled")
        self._btn_stop.configure(state="normal"  if can_stop  else "disabled")
        self._btn_reset.configure(state="normal" if can_reset else "disabled")

    def _set_state(self, state: str, msg: str = "") -> None:
        self._state = state
        self._refresh_state()
        if msg:
            self._status_var.set(msg)

    # ── Algorithm ─────────────────────────────────────────────────────────────

    def _select_algo(self, algo: str) -> None:
        self._engine.set_algorithm(algo)
        sel   = dict(fg_color=BLUE, hover_color=BLUE_DARK, text_color=TEXT)
        unsel = dict(fg_color=SURFACE2, hover_color=BORDER,
                     border_color=BORDER, text_color=TEXT_MUTED)
        for name in ("csrt", "kcf"):
            getattr(self, f"_btn_{name}").configure(
                **(sel if name.upper() == algo else unsel)
            )

    def _on_noise_change(self) -> None:
        self._engine.set_noise(
            round(self._proc_noise_var.get(), 1),
            round(self._meas_noise_var.get(), 1),
        )

    # ── Browse ────────────────────────────────────────────────────────────────

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")],
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._status_var.set(f"Cannot open: {path}")
            cap.release()
            return

        ok, frame = cap.read()
        cap.release()
        if not ok:
            self._status_var.set("Could not read first frame.")
            return

        self._cap_h, self._cap_w = frame.shape[:2]
        self._first_frame = frame.copy()
        self._video_path  = path
        short = path.replace("\\", "/").split("/")[-1]
        self._path_var.set(short)
        self._set_state(_S.READY, f"Loaded: {short}  —  Click ▶ Start to draw ROIs")

        # FIX 3: Show first frame immediately after browse
        self.after(50, lambda: self._show_frame(self._first_frame))

    # ── ROI drawing ───────────────────────────────────────────────────────────

    def _roi_press(self, event: tk.Event) -> None:
        if self._state != _S.ROI_SELECT:
            return
        self._roi_start = (event.x, event.y)

    def _roi_drag(self, event: tk.Event) -> None:
        # FIX 2: Draw directly on canvas — no coordinate conversion needed for preview
        if self._state != _S.ROI_SELECT or not self._roi_start:
            return
        x0, y0 = self._roi_start
        self._canvas.delete("roi_drag")
        self._canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline=CYAN, width=2, dash=(5, 3), tags="roi_drag",
        )

    def _roi_release(self, event: tk.Event) -> None:
        if self._state != _S.ROI_SELECT or not self._roi_start:
            return
        self._canvas.delete("roi_drag")
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        self._roi_start = None

        if abs(x1 - x0) > 10 and abs(y1 - y0) > 10:
            # FIX 2: Correct letterbox-aware conversion
            fx0, fy0 = self._canvas_to_frame(min(x0, x1), min(y0, y1))
            fx1, fy1 = self._canvas_to_frame(max(x0, x1), max(y0, y1))
            bbox = clamp_bbox((fx0, fy0, fx1 - fx0, fy1 - fy0), self._cap_w, self._cap_h)
            if bbox[2] > 10 and bbox[3] > 10:
                self._rois.append(bbox)
                n = len(self._rois)
                self._status_var.set(
                    f"{n} ROI(s) selected  ·  right-click: undo  ·  SPACE: confirm  ·  ESC: cancel"
                )
                self._redraw_confirmed_rois()

    def _roi_undo(self, _event: tk.Event) -> None:
        if self._state == _S.ROI_SELECT and self._rois:
            self._rois.pop()
            self._redraw_confirmed_rois()
            n = len(self._rois)
            self._status_var.set(f"{n} ROI(s)  ·  SPACE to confirm  ·  ESC to cancel")

    def _canvas_to_frame(self, cx: float, cy: float) -> tuple[int, int]:
        """Convert canvas pixel → video frame pixel, accounting for letterbox offset."""
        cw = max(self._canvas.winfo_width(), 1)
        ch = max(self._canvas.winfo_height(), 1)
        if self._cap_w == 0 or self._cap_h == 0:
            return 0, 0
        scale = min(cw / self._cap_w, ch / self._cap_h)
        ox = (cw - self._cap_w * scale) / 2.0
        oy = (ch - self._cap_h * scale) / 2.0
        fx = int((cx - ox) / scale)
        fy = int((cy - oy) / scale)
        return (
            max(0, min(fx, self._cap_w - 1)),
            max(0, min(fy, self._cap_h - 1)),
        )

    def _redraw_confirmed_rois(self) -> None:
        if self._first_frame is None:
            return
        preview = draw_roi_preview(self._first_frame, self._rois)
        self._show_frame(preview)

    # ── ROI key bindings ──────────────────────────────────────────────────────

    def _bind_roi_keys(self) -> None:
        self.bind("<space>",  self._roi_confirm)
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
            # Restart from the same video
            self._engine.clear()
            self._rois.clear()
            self._stop_event.clear()
            self._done_event.clear()
            self._set_state(_S.READY, "Click ▶ Start to draw ROIs")
            if self._first_frame is not None:
                self._show_frame(self._first_frame)
            return
        if self._state != _S.READY or self._first_frame is None:
            return
        self._rois.clear()
        self._set_state(_S.ROI_SELECT, "Drag to draw ROIs  ·  SPACE to confirm  ·  ESC to cancel")
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
        self._done_event.clear()
        # Drain stale frames
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break

        self._thread = _TrackingThread(
            self._video_path, self._engine,
            self._frame_q, self._stop_event, self._done_event,
        )
        self._thread.start()
        self._set_state(_S.TRACKING,
                        f"Tracking {len(self._rois)} object(s)  ·  Stop or ESC to end")
        self.after(16, self._poll_frames)

    # ── Frame polling ─────────────────────────────────────────────────────────

    def _poll_frames(self) -> None:
        # FIX 1: Always reschedule; wrap display in try/except so errors don't kill the loop
        if self._state != _S.TRACKING:
            return

        try:
            frame = self._frame_q.get_nowait()
            self._show_frame(frame)
            self._update_metrics()
        except queue.Empty:
            # Check if thread finished and queue is fully drained
            if self._done_event.is_set():
                self._set_state(_S.DONE, "Video finished.")
                self._update_metrics()
                return
        except Exception:
            pass  # Display error — keep polling

        self.after(16, self._poll_frames)  # ~60 fps

    # ── Frame display ─────────────────────────────────────────────────────────

    def _show_frame(self, frame: np.ndarray) -> None:
        cw = max(self._canvas.winfo_width(), 50)
        ch = max(self._canvas.winfo_height(), 50)
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2

        resized    = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.delete("all")
        self._canvas.create_image(ox, oy, anchor="nw", image=self._photo)

    def _draw_placeholder(self) -> None:
        self._canvas.delete("all")
        w = max(self._canvas.winfo_width(),  640)
        h = max(self._canvas.winfo_height(), 480)
        cx, cy = w // 2, h // 2
        self._canvas.create_text(cx, cy - 18, text="◈  VisionTrack",
                                  fill=BLUE, font=("Segoe UI", 22, "bold"))
        self._canvas.create_text(cx, cy + 18,
                                  text="Browse a video file, then click  ▶ Start",
                                  fill="#4a6aaa", font=("Segoe UI", 12))

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._state == _S.IDLE:
            self._draw_placeholder()
        elif self._state == _S.ROI_SELECT and self._first_frame is not None:
            self._redraw_confirmed_rois()

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _update_metrics(self) -> None:
        e = self._engine
        self._mc_fps.set(f"{e.fps:.1f}")
        self._mc_obj.set(str(e.object_count))
        self._mc_speed.set(f"{e.max_speed:.1f}")
        self._mc_skip.set(str(e.frame_skip))

        for child in self._obj_list_frame.winfo_children():
            child.destroy()
        for t in e.trackers:
            color = TRACK_COLORS.get(t.status, "#5a7aaa")
            row = ctk.CTkFrame(self._obj_list_frame, fg_color=SURFACE,
                               border_color=color, border_width=1, corner_radius=6)
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
