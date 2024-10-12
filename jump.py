import time
from threading import Thread

import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

video_path = 'jump5.mp4'
cap = cv2.VideoCapture(video_path)

jump_counter = 0
prev_hip_y = None
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
        keypoints = result.keypoints.data[0] if len(result.keypoints) > 0 else None
        if keypoints is not None:
            # 获取各个关键点的置信度
            head_confidence = keypoints[0][2].item()
            foot_confidence = keypoints[15][2].item()
            hip_confidence = keypoints[8][2].item()
            left_hand_confidence = keypoints[7][2].item()
            right_hand_confidence = keypoints[9][2].item()

            # 假设头部和脚部关键点的 y 坐标之差为近似身高
            head_y = keypoints[0][1].item()
            foot_y = keypoints[15][1].item()
            height = abs(foot_y - head_y)

            hip_y = keypoints[8][1].item()
            foot_y = keypoints[15][1].item()
            head_y = keypoints[0][1].item()
            left_hand_y = keypoints[7][1].item()
            right_hand_y = keypoints[9][1].item()

            # 如果任何一个关键点置信度过低，则跳过此次循环
            if head_confidence < 0.5 or foot_confidence < 0.5 or hip_confidence < 0.5 or left_hand_confidence < 0.5 or right_hand_confidence < 0.5:
                continue

            if prev_hip_y is not None:
                # 计算阈值 # 设置阈值，动作需要超过这个幅度
                threshold_hip = height * 0.01
                threshold_foot = height * 0.01
                threshold_hand = height * 0.01
                jump_left_right_foot_diff_threshold = height * 0.02

                if abs(hip_y - prev_hip_y) > threshold_hip:
                    print(f"abs(hip_y - prev_hip_y) - threshold_hip：{abs(hip_y - prev_hip_y) - threshold_hip}")
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
                            # 判断两脚之间高度差异
                            left_foot_y = keypoints[16][1].item()
                            right_foot_y = keypoints[15][1].item()
                            foot_diff = abs(left_foot_y - right_foot_y)
                            if hip_y < prev_hip_y and foot_diff < jump_left_right_foot_diff_threshold:
                                # 判断为向上跳起阶段
                                is_jumping_up = True
                            elif is_jumping_up and hip_y > prev_hip_y:
                                # 判断为落下阶段且之前有向上跳起，认为跳了一次绳
                                jump_counter += 1
                                is_jumping_up = False



            prev_foot_y = foot_y
            prev_hip_y = hip_y
            print(f"is_jumping_up：{is_jumping_up}")
            print(f"当前髋部高度：{hip_y}，跳绳次数：{jump_counter}")

    # 显示视频帧（可选）
    cv2.imshow('Frame', resized_frame)
    time.sleep(0.01)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()