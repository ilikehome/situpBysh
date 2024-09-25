import cv2


url = "rtsp://admin:tongda2024@10.21.53.64/Streaming/Channels/101"
cap = cv2.VideoCapture(url)
print(cap.isOpened())
ret, frame = cap.read()

while ret:
    ret, frame = cap.read()
    print(frame.shape)


cap.release()