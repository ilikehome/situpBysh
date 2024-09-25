import time

import cv2
from mtcnn import MTCNN
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
import numpy as np

# 判断是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 加载人脸检测模型
detector = MTCNN()

# 加载人脸特征提取模型
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face_features_from_image(image_path):
    img = cv2.imread(image_path)
    # 检测人脸
    boxes = detector.detect_faces(img)
    if boxes is not None and len(boxes) > 0:
        # 假设只取第一个检测到的人脸
        box = boxes[0]['box']
        face_np = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

        face_pil_image = Image.fromarray(face_np)

        # 将图像转换为合适的格式并标准化
        face_image = face_pil_image.resize((320, 320))
        face_tensor = fixed_image_standardization(torch.tensor(np.array(face_image)).permute(2, 0, 1).float()).to(device)

        # 生成人脸特征向量
        face_vector = resnet(face_tensor.unsqueeze(0))


        return face_vector
    else:
        return None

image_path = "12.jpg"

start_time = time.time()
face_vector = extract_face_features_from_image(image_path)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"extract_face_features_from_image 函数运行时间: {elapsed_time} 秒")

if face_vector is not None:
    print(face_vector)
else:
    print("No face detected in the image.")