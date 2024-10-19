import time
import traceback

import cv2
import numpy as np
import torch
#from flask import request, jsonify

import os
import sys
# 设置日志
import logging
from ultralytics import YOLO

sys.path.append('..'+ os.sep +'integrated')
# from integrated_flask import app, device, model

# app = Flask(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

model = YOLO('pre-trained_yolo/yolov8n-pose.pt').to(device)


class PersonWavingStatus:
    def __init__(self):
        self.hand_raised = False
        self.hand_raised_start_time = None
        self.waving = False
        self.waving_direction = None
        self.prev_wrist_pos = None
        self.continuous_waving = False
        self.shoulder_width = None

    def reset(self):
        self.hand_raised = False
        self.hand_raised_start_time = None
        self.waving = False
        self.waving_direction = None
        self.prev_wrist_pos = None
        self.continuous_waving = False
        self.shoulder_width = None


global_five_person_status = [PersonWavingStatus() for _ in range(5)]

# 截取一个区域内的人脸来识别，避免四周过多干扰
def clipped(image, x1, y1, x2, y2):
    assert x2 > x1
    assert y2 > y1
    h, w, _ = image.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, w)
    y2 = min(y2, h)
    # 截取指定区域
    clipped_image = image[y1:y2, x1:x2]
    return clipped_image


# 将 frame 从右到左等宽切成 5 份
def split_frame_right_to_left(frame):
    height, width, _ = frame.shape
    slice_width = width // 5
    slices = []
    start_x = width
    for _ in range(4):
        end_x = start_x - slice_width
        sliced_frame = frame[:, end_x:start_x]
        slices.append(sliced_frame)
        start_x = end_x
    slices.append(frame[:, :start_x])
    return slices



def is_raise(keypoints, one_person_status):
    one_person_valid_raise_hand = None
    left_wrist = keypoints[10]
    right_wrist = keypoints[11]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    # 获取左耳关键点（假设头部关键点索引为 0，左耳相对头部的偏移可根据实际模型输出调整）
    head_key = keypoints[0]
    left_ear_x = head_key[0] + 0.1 * (right_shoulder[0] - head_key[0])
    left_ear_y = head_key[1] - 0.1 * (head_key[1] - left_shoulder[1])
    right_ear_x = head_key[0] - 0.1 * (head_key[0] - left_shoulder[0])
    right_ear_y = head_key[1] - 0.1 * (head_key[1] - right_shoulder[1])
    if left_wrist[2].item() > 0.5 and right_wrist[2].item() > 0.5 and left_shoulder[2].item() > 0.5 and right_shoulder[
        2].item() > 0.5:
        # 计算肩宽
        if one_person_status.shoulder_width is None:
            # 将张量移动到 CPU 上再进行处理
            right_shoulder = right_shoulder.cpu()
            left_shoulder = left_shoulder.cpu()
            one_person_status.shoulder_width = np.linalg.norm(right_shoulder[:2] - left_shoulder[:2])

        # 判断举手并在正确一侧
        if (left_wrist[1] < left_shoulder[1] and left_wrist[0] < left_ear_x) or (
                right_wrist[1] < right_shoulder[1] and right_wrist[0] > right_ear_x):
            if not one_person_status.hand_raised:
                one_person_status.hand_raised_start_time = time.time()
                one_person_status.hand_raised = True  # 这个变量只是手举起来了，并不一定是两秒了
            else:
                current_time = time.time()
                if current_time - one_person_status.hand_raised_start_time > 2:
                    one_person_valid_raise_hand = True
        else:
            one_person_status.hand_raised = False
    return one_person_valid_raise_hand


if __name__ == "__main__":
    cap = cv2.VideoCapture("rasie_hand_fail_1.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        raise_result = [False, False, False, False, False]
        h, w, _ = frame.shape
        frame = clipped(frame, 0, 0, w, h)
        sub_frames = split_frame_right_to_left(frame)
        results = model(sub_frames)  # 5 个小帧的返回
        if results:
            for index, result in enumerate(results):
                keypoints_in_one_pic = result.keypoints.data if len(result.keypoints) > 0 else None
                if keypoints_in_one_pic is not None:
                    one_person_keypoints = results[index].keypoints.data[0]
                    if one_person_keypoints.size(0) > 0:
                        # 将张量移动到 CPU 上再进行处理
                        one_person_keypoints = one_person_keypoints.cpu()
                        if is_raise(one_person_keypoints, global_five_person_status[index]):
                            raise_result[index] = True
                        if not global_five_person_status[index].hand_raised:
                            global_five_person_status[index].reset()
                    else:
                        global_five_person_status[index].reset()
                else:
                    global_five_person_status[index].reset()
        else:
            for person_status in global_five_person_status:
                person_status.reset()

        sport_type = 2
        raise_result_dict = {}
        if sport_type == "1":  # 跳绳
            raise_result_dict = {i + 1: value for i, value in enumerate(raise_result)}
        elif sport_type == "2":  # 仰卧起坐
            raise_result_dict = {i + 1: raise_result[i + 1] for i in range(3)}

        print(raise_result_dict)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

