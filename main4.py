import cv2
import numpy as np
from ultralytics import YOLO

# 加载 YOLOv8n-pose 模型
model = YOLO("yolov8n-pose.pt")

# 打开视频设备（这里以摄像头为例，你也可以指定视频文件路径）
cap = cv2.VideoCapture(r"C:\Users\ilike\PycharmProjects\ai-sports-algorithm-test\situp\3.mp4")

# 定义不同部位的颜色映射
colors = [(0, 0, 255),   # 红色代表头部（假设鼻子关键点代表头部）
          (255, 255, 0), # 浅蓝色代表上肢（假设肩膀、手肘、手腕关键点代表上肢）
          (0, 255, 255), # 黄色代表胸部（假设胸部中心相关关键点，实际需根据模型输出确定合适索引）
          (255, 0, 255), # 紫色代表腹部（类似胸部，需合理确定腹部代表关键点索引）
          (128, 0, 128), # 紫色代表髋部（臀部区域）
          (0, 255, 0)]   # 绿色代表下肢（假设髋关节、膝盖、脚踝关键点代表下肢）

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 使用模型进行预测
    results = model.predict(frame)

    # 绘制关键点
    for result in results:
        for person_points in result.keypoints.data:
            for i in range(len(person_points)):
                x, y = int(person_points[i][0]), int(person_points[i][1])
                # 根据关键点索引分配颜色
                if i == 0:
                    cv2.circle(frame, (x, y), 5, colors[0], -1)
                elif 5 <= i <= 10:
                    cv2.circle(frame, (x, y), 5, colors[1], -1)
                elif i in [2, 3]:
                    cv2.circle(frame, (x, y), 5, colors[2], -1)
                elif i in [8, 9]:
                    cv2.circle(frame, (x, y), 5, colors[3], -1)
                elif i in [11, 12]:
                    cv2.circle(frame, (x, y), 5, colors[4], -1)
                elif 13 <= i <= 16:
                    cv2.circle(frame, (x, y), 5, colors[5], -1)

    # 显示结果
    cv2.imshow("Pose Detection", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()