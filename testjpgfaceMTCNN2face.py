import cv2
from mtcnn import MTCNN
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
import numpy as np

# 加载人脸检测模型
detector = MTCNN()

def is_same_person(vector1, vector2, threshold=0.6):
    distance = torch.dist(vector1, vector2).item()
    print(distance)
    return distance < threshold

# 加载人脸特征提取模型
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face_features_from_image(image_path):
    img = cv2.imread(image_path)
    # 检测人脸
    boxes = detector.detect_faces(img)
    if boxes is not None and len(boxes) > 0:
        # 假设只取第一个检测到的人脸
        box = boxes[0]['box']
        width = box[2]
        height = box[3]
        side_length = max(width, height)
        center_x = box[0] + width // 2
        center_y = box[1] + height // 2
        start_x = center_x - side_length // 2
        start_y = center_y - side_length // 2
        face_np = img[start_y:start_y + side_length, start_x:start_x + side_length]

        # 显示识别的人脸
        cv2.imshow('Detected Face', face_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        face_pil_image = Image.fromarray(face_np)

        # 将图像转换为合适的格式并标准化
        #face_image = face_pil_image.resize((320, 320))
        face_tensor = fixed_image_standardization(torch.tensor(np.array(face_pil_image)).permute(2, 0, 1).float())

        # 生成人脸特征向量
        face_vector = resnet(face_tensor.unsqueeze(0))
        return face_vector
    else:
        return None

image_path1 = r"C:\Users\ilike\PycharmProjects\situpBysh\1.jpg"
face_vector1 = extract_face_features_from_image(image_path1)
image_path3 = r"C:\Users\ilike\PycharmProjects\situpBysh\3.jpg"
face_vector3 = extract_face_features_from_image(image_path3)
image_path10 = r"C:\Users\ilike\PycharmProjects\situpBysh\10.jpg"
face_vector10 = extract_face_features_from_image(image_path10)
print(is_same_person(face_vector1, face_vector3))
print(is_same_person(face_vector1, face_vector10))