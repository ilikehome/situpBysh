import io

from ultralytics import YOLO
import cv2
import numpy as np
import time
from PIL import Image


model = YOLO("yolov8n-pose.pt")

image_path = "testMat.jpg"

# 读取磁盘文件为二进制数组
with open(image_path, 'rb') as f:
    image_data = f.read()



start_time = time.time()
for _ in range(20):
    img = Image.open(io.BytesIO(image_data))
    np_arr = np.array(img)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"循环 20 次所需时间为：{elapsed_time} 秒")



img = Image.open(io.BytesIO(image_data))
np_arr = np.array(img)
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