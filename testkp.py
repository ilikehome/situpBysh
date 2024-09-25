import cv2
import numpy as np
from ultralytics import YOLO

# 加载 YOLOv8n-pose 模型
model = YOLO("yolov8n-pose.pt")

# 打开视频设备（这里以摄像头为例，你也可以指定视频文件路径）
cap = cv2.VideoCapture(r"C:\Users\ilike\PycharmProjects\ai-sports-algorithm-test\situp\3.mp4")

def get_center_xy():
    # 获取图片的高度和宽度
    height, width, _ = frame.shape
    # 计算图片的中心坐标
    center_x = width // 2
    center_y = height // 2
    return center_x, center_y

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 使用模型进行预测
    results = model.predict(frame)

    # 绘制关键点
    for result in results:
        xyxy = result.boxes.xyxy
        for person_points in result.keypoints.data:
            for i in range(len(person_points)):
                # 在中心位置绘制一个点（以红色为例）
                cv2.circle(frame, get_center_xy(), 5, (0, 0, 255), -1)

                x, y = int(person_points[i][0]), int(person_points[i][1])
                # 根据关键点索引分配颜色
                if i == 0:#鼻子
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                elif i in [1, 2]:#眼睛
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                elif i in [3, 4]:#耳朵
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                elif i in [5, 6]:#肩膀
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                elif i in [7, 8]:#手肘
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                elif i in [9, 10]:#手腕
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                elif i in [11, 12]:#臀
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                elif i in [13, 14]:#膝盖
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                elif i in [15, 16]:#脚腕
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                else:
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

    # 显示结果
    cv2.imshow("Pose Detection", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()