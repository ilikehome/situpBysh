import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

video_path = 'jump2.mp4'
cap = cv2.VideoCapture(video_path)

jump_counter = 0
prev_hip_y = None
prev_foot_y = None
prev_hand_head_distance = None
is_jumping_up = False
is_hand_correct = False
is_valid_movement = False

# 设置调整后的阈值
threshold_hip = 0.003
threshold_foot = 0.004  # 脚的阈值
threshold_hand = 0.003  # 手相对头的阈值

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
            hip_y = keypoints[8][1]
            foot_y = keypoints[15][1]
            head_y = keypoints[0][1]
            left_hand_y = keypoints[7][1]
            right_hand_y = keypoints[9][1]

            if prev_hip_y is not None:
                if abs(hip_y - prev_hip_y) > threshold_hip:
                    # 判断脚与髋的运动一致性
                    if prev_foot_y is not None:
                        if (hip_y < prev_hip_y and foot_y < prev_foot_y and abs(foot_y - prev_foot_y) > threshold_foot) or (hip_y > prev_hip_y and foot_y > prev_foot_y and abs(foot_y - prev_foot_y) > threshold_foot):
                            is_valid_movement = True
                        else:
                            is_valid_movement = False

                    # 判断手相对头部的运动方向
                    cur_left_hand_head_distance = abs(left_hand_y - head_y)
                    cur_right_hand_head_distance = abs(right_hand_y - head_y)
                    if prev_hand_head_distance is not None:
                        cur_avg_hand_head_distance = (cur_left_hand_head_distance + cur_right_hand_head_distance) / 2
                        prev_avg_hand_head_distance = prev_hand_head_distance
                        if hip_y < prev_hip_y and cur_avg_hand_head_distance > prev_avg_hand_head_distance:
                            is_hand_correct = True
                        elif hip_y > prev_hip_y and cur_avg_hand_head_distance < prev_avg_hand_head_distance:
                            is_hand_correct = True
                        else:
                            is_hand_correct = False
                    else:
                        prev_hand_head_distance = (cur_left_hand_head_distance + cur_right_hand_head_distance) / 2

                    if is_valid_movement and is_hand_correct:
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()