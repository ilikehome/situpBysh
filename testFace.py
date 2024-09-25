import time

import cv2
from mtcnn import MTCNN
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
import numpy as np

# 加载人脸检测模型
detector = MTCNN()

# 加载人脸特征提取模型
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 打开视频文件
video_capture = cv2.VideoCapture(r"D:\2.mp4")
while True:
    # 读取视频帧
    ret, frame = video_capture.read()
    if not ret:
        break

    # 检测人脸
    boxes = detector.detect_faces(frame)
    for boxit in boxes:
        box = boxit['box']
        face_np = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        # 显示识别的人脸
        cv2.imshow('Detected Face', face_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        face_pil_image = Image.fromarray(face_np)

        # 将图像转换为合适的格式并标准化
        face_image = face_pil_image.resize((160, 160))
        face_tensor = fixed_image_standardization(torch.tensor(np.array(face_image)).permute(2, 0, 1).float())

        # 生成人脸特征向量
        start_time = time.time()
        face_vector = resnet(face_tensor.unsqueeze(0))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"特征提取耗时: {elapsed_time} 秒")
        print(face_vector)

# 释放视频捕获对象
video_capture.release()