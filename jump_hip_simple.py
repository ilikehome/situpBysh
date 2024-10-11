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
        threshold_hip_ok = None
        threshold_foot_ok = None
        hip_foot_same = None
        foot_diff_ok = None
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

            if prev_hip_y is not None and prev_foot_y is not None:
                # 计算阈值 # 设置阈值，动作需要超过这个幅度
                threshold_hip = height * 0.01
                threshold_foot = height * 0.01
                jump_left_right_foot_diff_threshold = height * 0.02

                threshold_hip_ok = abs(hip_y - prev_hip_y) > threshold_hip
                threshold_foot_ok = abs(foot_y - prev_foot_y) > threshold_foot
                hip_foot_same = (hip_y < prev_hip_y and foot_y < prev_foot_y) or (hip_y > prev_hip_y and foot_y > prev_foot_y)

                left_foot_y = keypoints[16][1]
                right_foot_y = keypoints[15][1]
                foot_diff = abs(left_foot_y - right_foot_y)
                foot_diff_ok = foot_diff < jump_left_right_foot_diff_threshold

                if threshold_hip_ok and threshold_foot_ok and hip_foot_same:# 判断脚与髋的运动一致性
                    # 判断两脚之间高度差异
                    left_foot_y = keypoints[16][1]
                    right_foot_y = keypoints[15][1]
                    foot_diff = abs(left_foot_y - right_foot_y)
                    if hip_y < prev_hip_y:
                        # 判断为向上跳起阶段
                        is_jumping_up = True
                        if not foot_diff_ok:
                            is_jumping_up = None
                    elif is_jumping_up and hip_y > prev_hip_y:
                        # 判断为落下阶段且之前有向上跳起，认为跳了一次绳
                        jump_counter += 1
                        is_jumping_up = False


            prev_foot_y = foot_y
            prev_hip_y = hip_y
        print(f"髋阈值:{threshold_hip_ok}，脚阈值:{threshold_foot_ok}，髋脚一致:{hip_foot_same}，左右脚:{foot_diff_ok}，方向up:{is_jumping_up}，跳绳次数：{jump_counter}")

    # 显示视频帧（可选）
    cv2.imshow('Frame', resized_frame)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()