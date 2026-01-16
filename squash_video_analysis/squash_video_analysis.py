

import cv2
import numpy as np
from collections import deque

# Load YOLOv5 model (using Ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Path to input video
VIDEO_PATH = r"c:\Users\Tuba Khan\Downloads\squash\video.mp4"  # Correct absolute path for video file
OUTPUT_PATH = 'output_annotated.mp4'

# Load model (YOLOv8 or YOLOv5)
def load_model():
    if YOLO is None:
        raise ImportError('Ultralytics YOLO not installed. Please install with: pip install ultralytics')
    # Use a pre-trained model (e.g., yolov8n.pt)
    model = YOLO('yolov8n.pt')
    return model

# Detect objects in frame
def detect_objects(model, frame):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
    classes = results[0].boxes.cls.cpu().numpy() if results else []
    confs = results[0].boxes.conf.cpu().numpy() if results else []
    return boxes, classes, confs

# Main processing loop
def process_video():
    model = load_model()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}. Please check the file path and format.")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # Ball tracking
    ball_positions = deque(maxlen=30)
    last_bounce = None
    last_hit = None
    player_hits = [1, 0]
    last_player = -1
    last_ball_center = None

    # Class IDs (YOLOv8: 32=sports ball, 0=person)
    BALL_CLASS = 32
    PLAYER_CLASS = 0

    frame_idx = 0
    max_ball_speed = 80  # pixels per frame, adjust as needed for realism
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        boxes, classes, confs = detect_objects(model, frame)
        # Find ball and players
        ball_candidates = []
        player_boxes = []
        player_areas = []
        for box, cls, conf in zip(boxes, classes, confs):
            if int(cls) == BALL_CLASS and conf > 0.3:
                bx1, by1, bx2, by2 = map(int, box)
                center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                ball_candidates.append((conf, center, box))
            elif int(cls) == PLAYER_CLASS and conf > 0.3:
                x1, y1, x2, y2 = map(int, box)
                area = (x2-x1)*(y2-y1)
                player_boxes.append(box)
                player_areas.append(area)
        # Only keep the two largest players (likely the real players)
        if len(player_boxes) > 2:
            idxs = np.argsort(player_areas)[-2:]
            player_boxes = [player_boxes[i] for i in idxs]

        # Ball detection: pick highest-confidence, filter out jumps
        ball_center = None
        if ball_candidates:
            ball_candidates.sort(reverse=True)  # highest confidence first
            conf, center, box = ball_candidates[0]
            if last_ball_center is not None:
                dist = np.linalg.norm(np.array(center) - np.array(last_ball_center))
                if dist < max_ball_speed:
                    ball_center = center
                else:
                    ball_center = last_ball_center  # ignore jump, keep last
            else:
                ball_center = center
            last_ball_center = ball_center
            ball_positions.append(ball_center)
        else:
            # If ball not detected, use last known position for consistency
            ball_center = last_ball_center
            if ball_center is not None:
                ball_positions.append(ball_center)

        # Draw yellow sharp trail (like reference), no red ball
        if ball_center is not None:
            # Draw sharp yellow trail from previous to current
            if len(ball_positions) > 1 and ball_positions[-2] is not None:
                cv2.line(frame, ball_positions[-2], ball_center, (0, 255, 255), 8)  # yellow, thick

        # Draw player boxes (thicker)
        for i, pbox in enumerate(player_boxes):
            px1, py1, px2, py2 = map(int, pbox)
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 7)
            label = f'Player-{i+1}'
            cv2.putText(frame, label, (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4)


        # Improved bounce/hit detection and ball location text
        bounce = False
        hit = False
        ball_location_text = ""
        if len(ball_positions) > 2:
            prev = ball_positions[-2]
            curr = ball_positions[-1]
            # Bounce: ball y near bottom of frame (ground)
            if curr[1] > height * 0.85 and (last_bounce is None or frame_idx - last_bounce > 10):
                bounce = True
                last_bounce = frame_idx
                ball_location_text = "GROUND"
            # Hit: ball y near top of frame (front wall)
            elif curr[1] < height * 0.15 and (last_hit is None or frame_idx - last_hit > 10):
                hit = True
                last_hit = frame_idx
                ball_location_text = "FRONT WALL"
            else:
                # In play
                ball_location_text = "IN PLAY"
        else:
            ball_location_text = ""

        # Assign hit to closest player
        if bounce or hit:
            if ball_center and player_boxes:
                dists = [np.linalg.norm(np.array(ball_center) - np.array([(b[0]+b[2])//2, (b[1]+b[3])//2])) for b in player_boxes]
                player_idx = int(np.argmin(dists))
                if player_idx != last_player:
                    player_hits[player_idx] += 1
                    last_player = player_idx


        # Draw overlays (bolder, no numbers, just location text if needed)
        cv2.putText(frame, 'FRONT WALL', (width//2-180, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,0,255), 7)
        cv2.putText(frame, 'IN', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 7)
        cv2.putText(frame, f'Total hits:', (width-400, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,0,255), 5)
        cv2.putText(frame, f'Player 1: {player_hits[0]}', (width-400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,0,255), 5)
        cv2.putText(frame, f'Player 2: {player_hits[1]}', (width-400, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,0,255), 5)
        # Ball location text (center top, larger, but no numbers)
        if ball_location_text:
            cv2.putText(frame, ball_location_text, (width//2-180, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 8)

        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print(f'Processing complete. Output saved to {OUTPUT_PATH}')
    print('Analysis finished. Check the output video for detected players, ball, and scores.')

if __name__ == '__main__':
    process_video()

