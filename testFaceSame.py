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



def is_same_person(vector1, vector2, threshold=0.6):
    """
    判断两个脸部向量是否表示同一个人。

    参数：
    vector1 (torch.Tensor): 第一个脸部特征向量。
    vector2 (torch.Tensor): 第二个脸部特征向量。
    threshold (float): 距离阈值，小于该阈值则认为是同一个人。

    返回：
    bool: 是否为同一个人。
    """
    distance = torch.dist(vector1, vector2).item()
    print(distance)
    return distance < threshold

def extract_face_features_from_image(image_path):
    img = cv2.imread(image_path)
    # 检测人脸
    boxes = detector.detect_faces(img)
    if boxes is not None and len(boxes) > 0:
        # 假设只取第一个检测到的人脸
        box = boxes[0]['box']
        face_np = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

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

image_path = r"C:\Users\ilike\PycharmProjects\situpBysh\xiangrui.jpg"
xiangrui_face_vector = extract_face_features_from_image(image_path)

image_path = r"C:\Users\ilike\PycharmProjects\situpBysh\xiangrui.jpg"
xiangrui_face_vector2 = extract_face_features_from_image(image_path)

is_same_person(xiangrui_face_vector, xiangrui_face_vector2)