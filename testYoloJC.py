import cv2
from ultralytics import YOLO

model = YOLO('modelJC.pt')

img = cv2.imread(r"C:\Users\ilike\PycharmProjects\situpBysh\shen.jpg")

results = model(img)

for result in results:
    for box in result.boxes:
        if box.cls == 0:  # Assuming class index for person is 0
            x1, y1, x2, y2 = box.xyxy[0]
            face_img = img[int(y1):int(y2), int(x1):int(x2)]
            cv2.imshow('Detected Face', face_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
    break