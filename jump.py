import time
from threading import Thread

import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

video_path = 'jump6.mp4'
cap = cv2.VideoCapture(video_path)

jump_counter = 0
prev_hip_y = None
prev_foot_y = None #脚需要和髋动作一致
prev_hand_head_distance = None #手上提时，髋必定在上提
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
        keypoints = result.keypoints.xyn[0]
        if len(keypoints) > 0:
            # 假设头部和脚部关键点的 y 坐标之差为近似身高
            head_y = keypoints[0][1]
            foot_y = keypoints[15][1]
            height = abs(foot_y - head_y)

            hip_y = keypoints[8][1]
            foot_y = keypoints[15][1]
            head_y = keypoints[0][1]
            left_hand_y = keypoints[7][1]
            right_hand_y = keypoints[9][1]

            if prev_hip_y is not None:
                # 计算阈值 # 设置阈值，动作需要超过这个幅度
                threshold_hip = height * 0.005
                threshold_foot = height * 0.005
                threshold_hand = height * 0.005

                if abs(hip_y - prev_hip_y) > threshold_hip:
                    # 判断手向上的时候，髋部是否向上
                    cur_left_hand_head_distance = abs(left_hand_y - head_y) # 判断手相对头部的运动方向
                    cur_right_hand_head_distance = abs(right_hand_y - head_y)
                    if prev_hand_head_distance is not None:
                        cur_avg_hand_head_distance = (cur_left_hand_head_distance + cur_right_hand_head_distance) / 2
                        is_hand_up = cur_avg_hand_head_distance < prev_hand_head_distance
                        prev_hand_head_distance = (cur_left_hand_head_distance + cur_right_hand_head_distance) / 2
                        if is_hand_up and not (hip_y < prev_hip_y):
                            continue
                    else:
                        prev_hand_head_distance = (cur_left_hand_head_distance + cur_right_hand_head_distance) / 2

                    if prev_foot_y is not None:
                        if (hip_y < prev_hip_y and foot_y < prev_foot_y and abs(foot_y - prev_foot_y) > threshold_foot) or (hip_y > prev_hip_y and foot_y > prev_foot_y and abs(foot_y - prev_foot_y) > threshold_foot):# 判断脚与髋的运动一致性
                            if hip_y < prev_hip_y:
                                # 判断为向上跳起阶段
                                is_jumping_up = True
                            elif is_jumping_up and hip_y > prev_hip_y:
                                # 判断为落下阶段且之前有向上跳起，认为跳了一次绳
                                jump_counter += 1
                                is_jumping_up = False


            prev_foot_y = foot_y
            prev_hip_y = hip_y
            print(f"当前髋部高度：{hip_y}，跳绳次数：{jump_counter}")

    # 显示视频帧（可选）
    cv2.imshow('Frame', resized_frame)
    time.sleep(0.2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()