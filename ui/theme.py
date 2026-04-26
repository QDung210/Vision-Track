import customtkinter as ctk

# ── Dual-mode tuples: (light_value, dark_value) ───────────────────────────
# CTk auto-switches between them when appearance mode changes.
BG         = ("#eef3fc", "#0a0f1a")
SURFACE    = ("#ffffff",  "#0d1526")
SURFACE2   = ("#e0ebff",  "#111d33")
BORDER     = ("#b8d0f0",  "#1a2d4d")
TEXT       = ("#0a1a3e",  "#e8f0ff")
TEXT_MUTED = ("#4a6aaa",  "#5a7aaa")

# ── Accent colors (fixed — readable on both backgrounds) ──────────────────
BLUE      = "#1d6ae5"
BLUE_DARK = "#1550b0"
CYAN      = "#06b6d4"
GREEN     = "#10b981"
RED       = "#ef4444"
YELLOW    = "#f59e0b"
MAGENTA   = "#d946ef"

# ── Canvas bg (plain tk.Canvas, no tuple support) ─────────────────────────
CANVAS_BG = {"dark": "#060d1a", "light": "#ccddf5"}

# ── Tracking state → accent color ─────────────────────────────────────────
TRACK_COLORS = {
    "tracking":  GREEN,
    "recovered": MAGENTA,
    "predicted": CYAN,
    "crowded":   YELLOW,
    "lost":      RED,
}

# ── Typography ────────────────────────────────────────────────────────────
FONT_TITLE   = ("Segoe UI", 20, "bold")
FONT_LABEL   = ("Segoe UI", 11)
FONT_MONO    = ("Consolas", 12, "bold")
FONT_MONO_SM = ("Consolas", 10)
FONT_METRIC  = ("Consolas", 26, "bold")
FONT_BTN     = ("Segoe UI", 12, "bold")

# ── Layout ────────────────────────────────────────────────────────────────
PANEL_LEFT  = 270
PANEL_RIGHT = 240
CORNER_R    = 8
CARD_PAD    = 12


def setup_theme(dark: bool = True) -> None:
    ctk.set_appearance_mode("dark" if dark else "light")
    ctk.set_default_color_theme("blue")
