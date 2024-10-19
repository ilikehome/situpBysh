import logging

import cv2
from ultralytics import YOLO
logging.getLogger('ultralytics').setLevel(logging.WARNING)

model = YOLO('yolov8n-pose.pt')
video_path = 'rasie_hand_fail_1.mp4'
cap = cv2.VideoCapture(video_path)
n = 5  # 假设 n 为 5，可根据实际情况调整

def filter_people(results):
    filtered_results = []
    for result in results:
        keypoints = result.keypoints.xyn[0]
        if keypoints is not None and len(keypoints) > 0:
            left_foot_y = keypoints[15][1]
            right_foot_y = keypoints[16][1]
            if (0.98 > left_foot_y > 0.25) or (0.98 > right_foot_y > 0.25):
                filtered_results.append(result)
    return filtered_results

def divide_frame_and_detect_people(frame):
    height, width, _ = frame.shape
    slice_width = width // n
    people_dict = {}
    results = model(frame)
    filtered_results = filter_people(results)
    for i in range(n):
        start_x = i * slice_width
        end_x = start_x + slice_width
        for result in filtered_results:
            keypoints = result.keypoints.xyn[0]
            if keypoints is not None and len(keypoints) > 0:
                center_x = (keypoints[0][0] + keypoints[15][0]) / 2
                if start_x <= center_x < end_x:
                    if i % 2 == 0:
                        if not people_dict or keypoints[0][1] > list(people_dict.values())[0].keypoints.xyn[0][0][1]:
                            people_dict[i + 1] = result
                    else:
                        if not people_dict or keypoints[0][1] < list(people_dict.values())[0].keypoints.xyn[0][0][1]:
                            people_dict[i + 1] = result
                    break
    return people_dict



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    people_info = divide_frame_and_detect_people(frame)
    print(people_info)

    # 在这两行代码中间加入的代码
    for index, result in people_info.items():
        if result.boxes.xyxy.shape[0] > 0:
            x_min, y_min, x_max, y_max = result.boxes.xyxy[0].tolist()
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            # 在人上方添加序号
            cv2.putText(frame, str(index), (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()