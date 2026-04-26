"""
VisionTrack — entry point.

Usage:
    python main.py
"""

from ui.app import VisionTrackApp


def main() -> None:
    app = VisionTrackApp()
    app.mainloop()


if __name__ == "__main__":
    main()
