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
    for single_frame_results in results:
        for person_result in single_frame_results:
            keypoints = person_result.keypoints.xyn[0]
            if keypoints is not None and len(keypoints) > 0:
                left_foot_y = keypoints[15][1]
                right_foot_y = keypoints[16][1]
                if (0.95 > left_foot_y > 0.25) or (0.95 > right_foot_y > 0.25):
                    filtered_results.append(person_result)
    return filtered_results

def divide_frame_and_detect_people(frame):
    height, width, _ = frame.shape
    # 每个小矩形区域占总宽度的比例
    slice_percentage = 1 / n
    people_dict = {}
    results = model(frame)
    filtered_results = filter_people(results)
    for i in range(n, 0, -1):
        start_percentage = (n - i) * slice_percentage
        end_percentage = start_percentage + slice_percentage
        for result in filtered_results:
            box = result.boxes
            keypoints = result.keypoints.xyn[0]
            if box is not None:
                x1, y1, x2, y2 = box.xyxyn[0]
                person_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                center_x = person_center[0]
                # 判断人的中心点是否在当前小矩形区域内
                if start_percentage <= center_x < end_percentage:
                    # 如果小矩形区域序号为偶数
                    if i % 2 == 0:
                        # 如果字典为空或者当前人的脚部 y 坐标大于字典中已有的人的脚部 y 坐标
                        if i not in people_dict or keypoints[15][1] > people_dict[i].keypoints.xyn[0][15][1]:
                            people_dict[i] = result
                    else:
                        # 如果字典为空或者当前人的脚部 y 坐标小于字典中已有的人的脚部 y 坐标
                        if i not in people_dict or keypoints[15][1] < people_dict[i].keypoints.xyn[0][15][1]:
                            people_dict[i] = result
    return people_dict

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 划分视频帧并检测人所在的小矩形区域，得到包含小矩形区域序号和对应检测结果的字典
    people_info = divide_frame_and_detect_people(frame)
    print([key for key in people_info])

    # 在这两行代码中间加入的代码
    for index, result in people_info.items():
        if result.boxes.xyxy.shape[0] > 0:
            x_min, y_min, x_max, y_max = result.boxes.xyxy[0].tolist()
            # 在人周围绘制绿色矩形框
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            # 在人上方添加序号，从右到左标注
            cv2.putText(frame, str(index), (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()