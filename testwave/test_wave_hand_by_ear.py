import torch
import cv2
import numpy as np
from ultralytics import YOLO
import logging

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("pre-trained_yolo/yolov8n-pose.pt").to(device)

class SlideDetector:
    def __init__(self):
        self.left_hand_left_of_left_ear_observed = False
        self.right_hand_right_of_right_ear_observed = False
        self.has_started_slide = False
        self.slide_direction = None
        self.ret_result = 0

    def clear(self):
        self.left_hand_left_of_left_ear_observed = False
        self.right_hand_right_of_right_ear_observed = False
        self.has_started_slide = False
        self.slide_direction = None
        self.ret_result = 0

def process_frame(frame, detector):
    is_hand_above_shoulder = False
    results = model(frame)
    selected_person = None
    min_distance_from_center = float('inf')
    for result in results:
        keypoints = result.keypoints.data[0] if result.keypoints is not None and len(result.keypoints.data) > 0 else None
        if keypoints is not None:
            if keypoints.size(0) == 0:
                continue
            # Check if tensors are on GPU and move to CPU if needed
            if keypoints.is_cuda:
                keypoints = keypoints.cpu()
            left_shoulder_pos = keypoints[5][:2] if keypoints[5][2].item() > 0.5 else None
            left_wrist_pos = keypoints[10][:2] if keypoints[10][2].item() > 0.5 else None
            right_shoulder_pos = keypoints[6][:2] if keypoints[6][2].item() > 0.5 else None
            head_pos = keypoints[0][:2] if keypoints[0][2].item() > 0.5 else None
            if left_shoulder_pos is not None and left_wrist_pos is not None and right_shoulder_pos is not None and head_pos is not None:
                distance_from_center = abs(head_pos[0] - frame.shape[1] / 2)
                if distance_from_center < min_distance_from_center:
                    min_distance_from_center = distance_from_center
                    selected_person = {
                        'left_shoulder_pos': left_shoulder_pos,
                        'left_wrist_pos': left_wrist_pos,
                        'right_shoulder_pos': right_shoulder_pos,
                        'head_pos': head_pos
                    }
    if selected_person:
        is_hand_above_shoulder = selected_person['left_wrist_pos'][1] < selected_person['left_shoulder_pos'][1]
        if not is_hand_above_shoulder:
            detector.clear()
            return selected_person, is_hand_above_shoulder

        left_ear_pos = selected_person['head_pos'] + np.array([0.1 * (selected_person['right_shoulder_pos'][0] - selected_person['head_pos'][0]), -0.1 * (selected_person['head_pos'][1] - selected_person['left_shoulder_pos'][1])])
        right_ear_pos = selected_person['head_pos'] - np.array([0.1 * (selected_person['head_pos'][0] - selected_person['left_shoulder_pos'][0]), -0.1 * (selected_person['head_pos'][1] - selected_person['right_shoulder_pos'][1])])
        if not detector.has_started_slide:
            if selected_person['left_wrist_pos'][0] < left_ear_pos[0]:
                detector.left_hand_left_of_left_ear_observed = True
            elif selected_person['left_wrist_pos'][0] > right_ear_pos[0]:
                detector.right_hand_right_of_right_ear_observed = True
        elif detector.has_started_slide and detector.slide_direction == "right":
            if selected_person['left_wrist_pos'][0] < left_ear_pos[0]:
                detector.has_started_slide = False
                detector.slide_direction = None
        elif detector.has_started_slide and detector.slide_direction == "left":
            if selected_person['left_wrist_pos'][0] > right_ear_pos[0]:
                detector.has_started_slide = False
                detector.slide_direction = None
        if selected_person['left_wrist_pos'][0] > right_ear_pos[0] and detector.left_hand_left_of_left_ear_observed:
            if not detector.has_started_slide:
                detector.ret_result = 1
                detector.has_started_slide = True
                detector.slide_direction = "right"
                logging.info("Detected potential start of right slide.")
            else:
                detector.ret_result = 0
        if selected_person['left_wrist_pos'][0] < left_ear_pos[0] and detector.right_hand_right_of_right_ear_observed:
            if not detector.has_started_slide:
                detector.ret_result = 2
                detector.has_started_slide = True
                detector.slide_direction = "left"
                logging.info("Detected potential start of left slide.")
            else:
                detector.ret_result = 0
        if not is_hand_above_shoulder and detector.has_started_slide:
            detector.has_started_slide = False
            detector.slide_direction = None
            logging.info("Hand is below shoulder. Resetting slide.")
    return selected_person, is_hand_above_shoulder

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    detector = SlideDetector()
    cap = cv2.VideoCapture('noperson.mp4')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 调整视频大小
        height, width, _ = frame.shape
        new_width = 640
        new_height = int((new_width / width) * height)
        frame = cv2.resize(frame, (new_width, new_height))

        selected_person, is_hand_above_shoulder = process_frame(frame, detector)
        logging.info(f"Detected a valid slide! Direction Result: {detector.ret_result}")
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()