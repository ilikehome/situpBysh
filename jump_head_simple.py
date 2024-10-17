import time
from threading import Thread
import logging

import cv2
from ultralytics import YOLO

# 设置日志级别为 WARNING 或更高，以抑制 INFO 级别的日志输出
logging.getLogger('ultralytics').setLevel(logging.WARNING)

model = YOLO('yolov8n-pose.pt')

video_path = 'jump12.mp4'
cap = cv2.VideoCapture(video_path)

jump_counter = 0
prev_head_y = None
prev_foot_y = None
prev_hand_head_distance = None
is_jumping_up = False

frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 缩小视频帧大小为原来的 1/2
    new_width = int(frame.shape[1] * 0.5)
    new_height = int(frame.shape[0] * 0.5)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    results = model(resized_frame)

    for result in results:
        threshold_head_ok = None
        threshold_foot_ok = None
        head_foot_same_direction = None
        keypoints = result.keypoints.xyn[0]
        confidences = result.keypoints.conf[0]
        if len(keypoints) > 0:
            # 检查关键点置信度
            if confidences[0] < 0.5 or confidences[11] < 0.5 or confidences[12] < 0.5 or confidences[15] < 0.5 or confidences[16] < 0.5:
                continue

            # 假设头部和脚部关键点的 y 坐标之差为近似身高
            head_y = keypoints[0][1]
            # 计算左右脚部平均 y 坐标
            left_foot_y = keypoints[15][1]
            right_foot_y = keypoints[16][1]
            foot_y = (left_foot_y + right_foot_y) / 2
            height = abs(foot_y - head_y)

            if prev_head_y is not None and prev_foot_y is not None:
                # 计算阈值 # 设置阈值，动作需要超过这个幅度
                threshold_head = height * 0.01
                threshold_foot = height * 0.01

                threshold_head_ok = abs(head_y - prev_head_y) > threshold_head
                threshold_foot_ok = abs(foot_y - prev_foot_y) > threshold_foot
                head_foot_same_direction = (head_y < prev_head_y and foot_y < prev_foot_y) or (head_y > prev_head_y and foot_y > prev_foot_y)

                if threshold_head_ok and threshold_foot_ok and head_foot_same_direction:
                    # 判断两脚之间高度差异
                    if head_y < prev_head_y:
                        # 判断为向上跳起阶段
                        is_jumping_up = True
                    elif is_jumping_up and head_y > prev_head_y:
                        # 判断为落下阶段且之前有向上跳起，认为跳了一次绳
                        jump_counter += 1
                        is_jumping_up = False

            prev_foot_y = foot_y
            prev_head_y = head_y
        print(f"头阈值:{threshold_head_ok}，脚阈值:{threshold_foot_ok}，头脚一致:{head_foot_same_direction}，方向up:{is_jumping_up}，跳绳次数：{jump_counter}")

    # 显示视频帧（可选）
    cv2.imshow('Frame', resized_frame)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()