# Squash Video Analysis

**Developer:** tubakhxn

## Project Overview
This project is a squash video analysis tool that uses computer vision to detect and track players and the ball in a squash game video. It overlays player bounding boxes, a yellow ball trail, and displays player scores and hit locations. The tool is designed for offline analysis of uploaded squash match videos (not webcam), and does not use MediaPipe.

## Features
- Detects and tracks two main players
- Detects and tracks the squash ball
- Draws a yellow trail showing the ball's motion
- Overlays player bounding boxes and labels
- Displays player scores and hit locations (front wall, ground, in play)
- Bold, clear overlays for easy analysis

## Folder Structure
```
squash/
├── .venv/                  # Python virtual environment (auto-generated)
├── squash_video_analysis/   # Main source code folder
│   └── squash_video_analysis.py  # Main script
├── video.mp4               # Input squash video file (place your video here)
├── output_annotated.mp4    # Output video with overlays (auto-generated)
├── yolov8n.pt              # YOLOv8 model weights file
└── README.md               # Project documentation (this file)
```

## How to Use
1. **Install Python 3.11** (or compatible version).
2. **Clone or download** this project folder.
3. **Set up the virtual environment:**
   - Open a terminal in the project folder.
   - Run: `python -m venv .venv`
   - Activate the environment:
     - Windows: `.venv\Scripts\activate`
     - macOS/Linux: `source .venv/bin/activate`
4. **Install dependencies:**
   - Run: `pip install opencv-python ultralytics numpy`
5. **Add your squash video:**
   - Place your squash video file in the project folder as `video.mp4` (or update the path in the script).
6. **Download YOLOv8 weights:**
   - Place `yolov8n.pt` in the project folder (download from Ultralytics if needed).
7. **Run the analysis:**
   - Run: `python squash_video_analysis/squash_video_analysis.py`
8. **View results:**
   - The output video with overlays will be saved as `output_annotated.mp4`.

## Notes
- The script is designed for offline video analysis, not real-time webcam use.
- Only two main players are tracked for clarity.
- The overlays are designed to match a reference style: bold, clear, and without coordinate numbers.

---
**Developer:** tubakhxn
