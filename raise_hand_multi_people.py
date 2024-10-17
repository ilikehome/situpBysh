import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

def filter_result(frame):
    detections = model(frame)[0]
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    center_x = frame_width // 2
    center_y = frame_height // 2
    sub_rect_width = frame_width * 0.9
    sub_rect_height = frame_height * 0.7
    sub_rect_start_x = center_x - sub_rect_width // 2
    sub_rect_start_y = center_y - sub_rect_height // 2
    sub_rectangles = [(sub_rect_start_x + i * sub_rect_width // 5, sub_rect_start_y,
                       sub_rect_start_x + (i + 1) * sub_rect_width // 5, sub_rect_start_y + sub_rect_height) for i in
                      range(5)]
    people = []
    for detection in detections:
        box = detection.boxes
        if box is not None:
            x1, y1, x2, y2 = box.xyxy[0]
            person_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if sub_rect_start_x <= person_center[0] < sub_rect_start_x + sub_rect_width and sub_rect_start_y <= person_center[1] < sub_rect_start_y + sub_rect_height:
                people.append((detection, person_center))

    sub_results = [None] * 5
    for i, sub_rect in enumerate(sub_rectangles):
        sub_people = [(person,person_center) for person,person_center in people if sub_rect[0] <= person_center[0] < sub_rect[2] and sub_rect[1] <= person_center[1] < sub_rect[3]]
        if len(sub_people) == 1:
            sub_results[i] = sub_people[0][0]
        elif len(sub_people) > 1:
            if i in [1, 3, 5]:
                sub_results[i] = min(sub_people, key=lambda x: x[1][1])[0]
            elif i in [2, 4]:
                sub_results[i] = max(sub_people, key=lambda x: x[1][1])[0]

    # 绘制 sub_rectangles 对应的矩形框
    for rect in sub_rectangles:
        cv2.rectangle(frame, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 2)

    return sub_results

mp4path = "3.mp4"

cap = cv2.VideoCapture(mp4path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_results = filter_result(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()