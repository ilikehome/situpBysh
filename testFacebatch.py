import cv2
from mtcnn import MTCNN
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
import numpy as np
import time

# 加载人脸检测模型
detector = MTCNN()

# 加载人脸特征提取模型
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 打开视频文件
video_capture = cv2.VideoCapture(r"D:\2.mp4")

def extract_faces_and_features(frame):
    faces = []
    # 检测人脸
    boxes = detector.detect_faces(frame)
    for box in boxes:
        box_data = box['box']
        face_np = frame[box_data[1]:box_data[1]+box_data[3], box_data[0]:box_data[0]+box_data[2]]
        faces.append(face_np)
    face_tensors = []
    for face_np in faces:
        face_pil_image = Image.fromarray(face_np)
        face_image = face_pil_image.resize((160, 160))
        face_tensor = fixed_image_standardization(torch.tensor(np.array(face_image)).permute(2, 0, 1).float())
        face_tensors.append(face_tensor)
    start_time = time.time()
    face_vectors = resnet(torch.stack(face_tensors))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"特征提取耗时: {elapsed_time} 秒")
    return face_vectors

while True:
    # 读取视频帧
    ret, frame = video_capture.read()
    if not ret:
        break

    face_vectors = extract_faces_and_features(frame)
    for face_vector in face_vectors:
        print(face_vector)

# 释放视频捕获对象
video_capture.release()