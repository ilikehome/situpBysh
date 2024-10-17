import cv2
import numpy as np
from ultralytics import YOLO


def process_video(mp4_path_zn, n):
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(mp4_path_zn)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = frame_width // 2
    center_y = frame_height // 2
    sub_rect_width = frame_width * 0.9
    sub_rect_height = frame_height * 0.7
    sub_rect_start_x = center_x - sub_rect_width // 2
    sub_rect_start_y = center_y - sub_rect_height // 2
    sub_rectangles = [(int(sub_rect_start_x + i * sub_rect_width // 5), int(sub_rect_start_y),
                       int(sub_rect_start_x + (i + 1) * sub_rect_width // 5), int(sub_rect_start_y + sub_rect_height))
                      for i in range(5)]
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = model(frame)[0]
        people = []
        for detection in detections:
            box = detection.boxes
            if box is not None:
                x1, y1, x2, y2 = box.xyxy[0]
                person_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                if sub_rect_start_x <= person_center[0] < sub_rect_start_x + sub_rect_width and sub_rect_start_y <= \
                        person_center[1] < sub_rect_start_y + sub_rect_height:
                    people.append(detection)
        if len(people) == n:
            for person in people:
                results.append(person)
        else:
            sub_results = [None] * 5
            for i, sub_rect in enumerate(sub_rectangles):
                sub_people = [person for person in people if
                              sub_rect[0] <= person.boxes.xyxy[0][0] < sub_rect[2] and sub_rect[1] <=
                              person.boxes.xyxy[0][1] < sub_rect[3]]
                if len(sub_people) == 1:
                    sub_results[i] = sub_people[0]
                elif len(sub_people) > 1:
                    if i in [1, 3, 5]:
                        sub_results[i] = min(sub_people, key=lambda x: (x.boxes.xyxy[0][1] + x.boxes.xyxy[0][3]) / 2)
                    elif i in [2, 4]:
                        sub_results[i] = max(sub_people, key=lambda x: (x.boxes.xyxy[0][1] + x.boxes.xyxy[0][3]) / 2)
            results = [result for result in sub_results if result is not None]

        # 绘制 sub_rectangles 对应的矩形框
        for rect in sub_rectangles:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

        # 显示当前帧
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return results


mp4path = "3.mp4"
n = 3  # 期望的人数
result = process_video(mp4path, n)
print(result)