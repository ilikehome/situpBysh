import cv2
import dlib
import numpy as np

# 加载人脸检测器
detector = dlib.get_frontal_face_detector()
# 加载特征提取器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def extract_face_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = predictor(gray, face)
    shape = np.array([(part.x, part.y) for part in landmarks.parts()])
    face_descriptor = facerec.compute_face_descriptor(img, landmarks)
    return np.array(face_descriptor)

image_path = r"C:\Users\ilike\PycharmProjects\situpBysh\1.jpg"
feature_vector = extract_face_features(image_path)
if feature_vector is not None:
    print(feature_vector)
else:
    print("No face detected in the image.")