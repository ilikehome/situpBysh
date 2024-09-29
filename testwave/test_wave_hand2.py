import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("pre-trained_yolo/yolov8n-pose.pt").to(device)

shoulder_width = None
hand_coords = []
initial_hand_x = None
is_hand_above_shoulder = False
has_waved = False
wave_direction = 0

def is_valid_wave(coords, shoulder_width):
    if shoulder_width is None:
        return False
    global initial_hand_x, has_waved, wave_direction
    if len(coords) < 2:
        return False
    current_hand_x, current_hand_y = coords[-1]
    if is_hand_above_shoulder:
        if initial_hand_x is None:
            initial_hand_x = current_hand_x
        if current_hand_y >= 0:  # Assuming shoulder y-coordinate is 0 for simplicity
            displacement = current_hand_x - initial_hand_x
            if displacement >= shoulder_width*2 and initial_hand_x < current_hand_x:
                if not has_waved:
                    has_waved = True
                    wave_direction = 1
                    return True
                else:
                    return False
            elif displacement <= -shoulder_width*2 and initial_hand_x > current_hand_x:
                if not has_waved:
                    has_waved = True
                    wave_direction = 2
                    return True
                else:
                    return False
    return False

def process_frame(frame):
    global shoulder_width, is_hand_above_shoulder, has_waved, initial_hand_x, wave_direction
    results = model(frame)
    if results:
        for result in results:
            keypoints = result.keypoints.data[0] if result.keypoints is not None and len(result.keypoints.data) > 0 else None
            if keypoints is not None:
                if keypoints.size(0) == 0:
                    return

                # 获取左肩和左手的 keypoints
                left_shoulder = keypoints[5]
                left_wrist = keypoints[10]
                if left_wrist[2].item() > 0.5 and left_shoulder[2].item() > 0.5:
                    # Calculate shoulder width on the fly if not already calculated
                    if shoulder_width is None:
                        right_shoulder = keypoints[6]
                        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
                    if left_wrist[1] < left_shoulder[1]:
                        is_hand_above_shoulder = True
                        hand_coords.append((left_wrist[0].item(), left_wrist[1].item()))
                    else:
                        is_hand_above_shoulder = False
                        hand_coords.clear()
                        initial_hand_x = None
                        has_waved = False
                        wave_direction = 0

def main():
    cap = cv2.VideoCapture('noperson.mp4')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        process_frame(frame)
        wave = is_valid_wave(hand_coords, shoulder_width)
        if wave:
            print("Detected a valid wave!")
            print(wave_direction)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()