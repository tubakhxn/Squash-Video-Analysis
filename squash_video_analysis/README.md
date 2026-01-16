# Squash Video Analysis Project

This project analyzes a squash game video to detect players and the ball using AI models (YOLO via Ultralytics) and OpenCV. No webcam or MediaPipe required.

## How to Use
1. Place your squash game video file (e.g., `input_squash_game.mp4`) in this folder.
2. Run `squash_video_analysis.py`.
3. The script will output an annotated video (`output_annotated.mp4`) with detected players and ball.

## Dependencies
- Python 3.8+
- OpenCV
- Ultralytics (YOLO)
- numpy

## Installation
```bash
pip install opencv-python ultralytics numpy
```

## Notes
- The script uses a pre-trained YOLO model (`yolov8n.pt`). The model will be downloaded automatically by Ultralytics if not present.
- You can change the input video filename in the script.

## Project Files
- `squash_video_analysis.py`: Main script
- `input_squash_game.mp4`: Your squash video (add this file)
- `output_annotated.mp4`: Output video with detections
