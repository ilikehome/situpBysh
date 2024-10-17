from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO("yolov8n-pose.pt")

image_path = "testMat.jpg"

# 读取磁盘文件为二进制数组
with open(image_path, 'rb') as f:
    image_data = f.read()




start_time = time.time()
for _ in range(20):
    # 将二进制数组转换为 numpy 数组
    np_arr = np.frombuffer(image_data, np.uint8)
    # 解码为图像
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"循环 20 次所需时间为：{elapsed_time} 秒")




# 将二进制数组转换为 numpy 数组
np_arr = np.frombuffer(image_data, np.uint8)
# 解码为图像
img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
results = model(img)

has_person = False
for result in results:
    for cls_id in result.boxes.cls:
        if cls_id == 0:  # Assuming person class ID is 0
            has_person = True
            break

if has_person:
    print("Image contains a person.")
else:
    print("Image does not contain a person.")