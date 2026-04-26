import customtkinter as ctk

# ── Palette ──────────────────────────────────────────────────────────
BG        = "#0a0f1a"   # deep navy-black background
SURFACE   = "#0d1526"   # card / panel surface
SURFACE2  = "#111d33"   # slightly lighter surface
BORDER    = "#1a2d4d"   # subtle blue-tinted border
BLUE      = "#1d6ae5"   # primary accent blue
BLUE_DARK = "#1550b0"   # pressed / darker blue
CYAN      = "#06b6d4"   # secondary accent cyan
GREEN     = "#10b981"   # success / tracking OK
RED       = "#ef4444"   # error / lost
YELLOW    = "#f59e0b"   # warning / crowded
MAGENTA   = "#d946ef"   # recovered state
TEXT      = "#e8f0ff"   # primary text (blue-tinted white)
TEXT_MUTED = "#5a7aaa"  # secondary text

# Tracking state → UI color
TRACK_COLORS = {
    "tracking":  GREEN,
    "recovered": MAGENTA,
    "predicted": CYAN,
    "crowded":   YELLOW,
    "lost":      RED,
}

# ── Typography ────────────────────────────────────────────────────────
FONT_TITLE   = ("Segoe UI", 20, "bold")
FONT_HEADING = ("Segoe UI", 13, "bold")
FONT_LABEL   = ("Segoe UI", 11)
FONT_MONO    = ("Consolas", 12, "bold")
FONT_MONO_SM = ("Consolas", 10)
FONT_METRIC  = ("Consolas", 26, "bold")
FONT_BTN     = ("Segoe UI", 12, "bold")

# ── Sizes ─────────────────────────────────────────────────────────────
PANEL_LEFT  = 270
PANEL_RIGHT = 240
HEADER_H    = 60
STATUSBAR_H = 30
CORNER_R    = 8   # widget corner radius
CARD_PAD    = 12  # inner padding for cards


def setup_theme() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
