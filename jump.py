import time

import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

video_path = 'jump1.mp4'
cap = cv2.VideoCapture(video_path)

jump_counter = 0
prev_head_waist_distance = None
person_height_percentage = 0.005
is_jumping_up = False  # 初始化变量

while cap.isOpened():
    time.sleep(0.3)
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        keypoints = result.keypoints.xyn[0]
        if len(keypoints) > 0:
            head_y = keypoints[0][1]
            waist_y = keypoints[2][1]
            current_head_waist_distance = abs(head_y - waist_y)
            if prev_head_waist_distance is not None:
                height = result.boxes.xywh[0][3]
                if current_head_waist_distance < prev_head_waist_distance and \
                        prev_head_waist_distance - current_head_waist_distance > height * person_height_percentage:
                    # 判断为向上跳起阶段
                    is_jumping_up = True
                elif is_jumping_up and current_head_waist_distance > prev_head_waist_distance:
                    # 判断为落下阶段且之前有向上跳起，认为跳了一次绳
                    jump_counter += 1
                    is_jumping_up = False
            prev_head_waist_distance = current_head_waist_distance
            print(f"当前头部与腰部相对距离：{current_head_waist_distance}，跳绳次数：{jump_counter}")

    # 显示视频帧（可选）
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()