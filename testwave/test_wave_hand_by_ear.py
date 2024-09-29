import torch
import cv2
import numpy as np
from ultralytics import YOLO
import logging

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("pre-trained_yolo/yolov8n-pose.pt").to(device)

def process_frame(frame):
    results = model(frame)
    left_wrist_pos = None
    left_shoulder_pos = None
    right_shoulder_pos = None
    left_ear_pos = None
    right_ear_pos = None
    is_hand_above_shoulder = False
    has_started_slide = False
    slide_direction = None
    if results:
        for result in results:
            keypoints = result.keypoints.data[0] if result.keypoints is not None and len(result.keypoints.data) > 0 else None
            if keypoints is not None:
                if keypoints.size(0) == 0:
                    return None, None, None, None, None, False, False, None
                left_shoulder_pos = keypoints[5][:2] if keypoints[5][2].item() > 0.5 else None
                left_wrist_pos = keypoints[10][:2] if keypoints[10][2].item() > 0.5 else None
                right_shoulder_pos = keypoints[6][:2] if keypoints[6][2].item() > 0.5 else None
                head_pos = keypoints[0][:2] if keypoints[0][2].item() > 0.5 else None
                if head_pos is not None:
                    left_ear_pos = head_pos + np.array([0.1 * (right_shoulder_pos[0] - head_pos[0]), -0.1 * (head_pos[1] - left_shoulder_pos[1])]) if right_shoulder_pos is not None and left_shoulder_pos is not None else None
                    right_ear_pos = head_pos - np.array([0.1 * (head_pos[0] - left_shoulder_pos[0]), -0.1 * (head_pos[1] - right_shoulder_pos[1])]) if right_shoulder_pos is not None and left_shoulder_pos is not None else None
                if left_wrist_pos is not None and left_shoulder_pos is not None:
                    is_hand_above_shoulder = left_wrist_pos[1] < left_shoulder_pos[1]
    return left_wrist_pos, left_shoulder_pos, right_shoulder_pos, left_ear_pos, right_ear_pos, is_hand_above_shoulder, has_started_slide, slide_direction

def is_valid_slide(left_wrist_pos, left_ear_pos, right_ear_pos, is_hand_above_shoulder, has_started_slide, slide_direction):
    if left_wrist_pos is None or left_ear_pos is None or right_ear_pos is None:
        return False, has_started_slide, slide_direction
    if is_hand_above_shoulder:
        if not has_started_slide:
            if left_wrist_pos[0] < left_ear_pos[0]:
                has_started_slide = True
                slide_direction = None
        else:
            if slide_direction is None:
                if left_wrist_pos[0] > right_ear_pos[0]:
                    slide_direction = "right"
                elif left_wrist_pos[0] < left_ear_pos[0]:
                    has_started_slide = False
            elif slide_direction == "right":
                if left_wrist_pos[0] < left_ear_pos[0]:
                    has_started_slide = False
    else:
        has_started_slide = False
        slide_direction = None
    return has_started_slide, slide_direction

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    cap = cv2.VideoCapture('cam.mp4')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        left_wrist_pos, left_shoulder_pos, right_shoulder_pos, left_ear_pos, right_ear_pos, is_hand_above_shoulder, has_started_slide, slide_direction = process_frame(frame)
        has_started_slide, slide_direction = is_valid_slide(left_wrist_pos, left_ear_pos, right_ear_pos, is_hand_above_shoulder, has_started_slide, slide_direction)
        if slide_direction:
            logging.info(f"Detected a valid slide! Direction: {slide_direction}")
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()