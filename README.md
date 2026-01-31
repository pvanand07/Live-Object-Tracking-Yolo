# Object Tracking with YOLO

Real-time object tracking using Ultralytics YOLOv8 and webcam video feed.

## Features

- Real-time object detection and tracking
- Webcam video input
- ByteTrack algorithm for persistent object IDs
- Visual annotations with bounding boxes and tracking IDs

## Setup

This project uses `uv` for dependency management.

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the tracking script:**
   ```bash
   uv run track_webcam.py
   ```

## Usage

- The script will automatically download the YOLOv8 nano model on first run
- Press `q` to quit the application
- The tracking window shows:
  - Bounding boxes around detected objects
  - Tracking IDs (persistent across frames)
  - Object class labels and confidence scores
  - Total number of tracked objects

## Model Options

You can change the YOLO model in `track_webcam.py`:
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

## Tracker Options

Available trackers:
- `bytetrack.yaml` (default) - ByteTrack algorithm
- `botsort.yaml` - BoT-SORT algorithm

## Documentation

For more information, visit the [Ultralytics Tracking Documentation](https://docs.ultralytics.com/modes/track/)
