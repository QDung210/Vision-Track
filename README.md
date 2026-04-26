# VisionTrack

Real-time multi-object video tracking using **CSRT / KCF** algorithms combined with a custom **Kalman Filter**, built with Python and CustomTkinter.

---

## Features

- **Dual-algorithm support** — switch between CSRT (accurate) and KCF (fast)
- **Kalman Filter fusion** — smooth position prediction even when the tracker loses the object
- **Multi-object tracking** — draw as many ROIs as needed before starting
- **Live metrics** — FPS, object count, max speed, frame skip counter
- **Failure recovery** — automatically reinitialises the tracker after 8 consecutive failures
- **Crowding detection** — detects overlapping objects via IoU and switches colour
- **Modern dark UI** — blue/black themed CustomTkinter interface

---

## Tracking States

| Colour  | State     | Meaning                                       |
|---------|-----------|-----------------------------------------------|
| Green   | tracking  | Tracker OK, no failures                       |
| Cyan    | predicted | Tracker skipped this frame, Kalman predicting |
| Magenta | recovered | Object found again after being lost           |
| Yellow  | crowded   | Overlapping with another tracked object       |
| Red     | lost      | Tracker failed for too many frames            |

---

## Project Structure

```
OpenCV_project/
├── main.py                 # Entry point
├── requirements.txt
├── core/
│   ├── kalman_filter.py    # 4-state Kalman filter [x, y, vx, vy]
│   ├── tracker.py          # ObjectTracker (OpenCV + Kalman)
│   ├── engine.py           # TrackingEngine (multi-object manager)
│   └── drawing.py          # Frame annotation (bounding boxes, HUD)
├── ui/
│   ├── app.py              # Main CustomTkinter application
│   └── theme.py            # Colours, fonts, sizing constants
└── utils/
    └── geometry.py         # IoU, bounding box helpers
```

---

## Installation

```bash
pip install -r requirements.txt
```

> Python 3.11+ recommended.

---

## Usage

```bash
python main.py
```

1. Click **Browse…** to select a video file (`.mp4`, `.avi`, `.mov`, `.mkv`).
2. Click **▶ Start** — the first frame appears in the canvas.
3. **Click and drag** to draw a bounding box around each object to track.
4. **Right-click** to undo the last ROI.
5. Press **SPACE** to confirm ROIs and begin tracking.
6. Press **■ Stop** or **ESC** to halt tracking.
7. Click **↺ Reset** to clear everything and start over.

---

## Configuration

Adjust **Process Noise** and **Measurement Noise** sliders in the left panel before starting:

- Higher **process noise** → Kalman trusts velocity more (tracks fast/erratic objects)
- Higher **measurement noise** → Kalman trusts the visual tracker less (smoother but laggier)

---

## Dependencies

| Package         | Purpose                          |
|-----------------|----------------------------------|
| opencv-python   | Video capture + CSRT/KCF trackers|
| customtkinter   | Modern dark-mode UI              |
| numpy           | Matrix operations (Kalman filter)|
| Pillow          | Convert OpenCV frames for display|
