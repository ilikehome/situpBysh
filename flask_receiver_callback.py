import base64
import cv2
import numpy as np
import os
import json
import time
import cv2
import requests

# rope skipping init
#url = 'http://10.31.61.174:5003/algorithm/situp/init'
url = 'http://10.31.61.174:5003/algorithm/situp/init'
response = requests.post(url, data={'locationIndex': '1,2'})
processed_frame_base64 = response.json()
if response.status_code == 200:
    code, msg = processed_frame_base64['code'], processed_frame_base64['msg']
    print(f"situp initialization: {msg}")
else:
    print(f"situp initialization: {processed_frame_base64['msg']}")


cap = cv2.VideoCapture(r"C:\Users\ilike\PycharmProjects\ai-sports-algorithm-test\situp\3.mp4")
t1 = time.time()
while cap.isOpened():
    
    success, frame = cap.read()
    if not success:
        print("fail to read frame from video")
        break
    
    assert len(frame.shape) == 3
    assert frame.shape[1] > frame.shape[0] > frame.shape[2]
    #frame = cv2.resize(frame, (128 * frame.shape[1] // frame.shape[0], 128))
    
    # 编码图像
    _, buffer = cv2.imencode('.jpg', frame)
    # 将字节数组直接发送到服务器
    url = 'http://10.31.61.174:5003/algorithm/situp/count'
    response = requests.post(url, files={'frame': buffer.tobytes()})

    processed_frame_base64 = response.json()
    
    if response.status_code == 200:
        # processed_frame_base64 = response.content
        
        # print(processed_frame_base64, type(processed_frame_base64))
        # print(json.loads(processed_frame_base64))
        # processed_frame_bytes = base64.b64decode(processed_frame_base64)
        # np_arr = np.frombuffer(processed_frame_base64, np.uint8)
        # processed_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        code, info = processed_frame_base64['code'], processed_frame_base64['data']
        print(info)
        
        # 显示处理后的图像
        # print(type(processed_frame_base64), processed_frame)
        if code != '0':
            print(processed_frame_base64['msg'])
        else:                 
            t2 = time.time()
            fps = 1 / (t2 - t1)
    else:
        print(processed_frame_base64['msg'])
                       
    t1 = time.time()



# rope skipping finish
url = 'http://10.31.61.174:5003/algorithm/situp/finish'
response = requests.post(url)
processed_frame_base64 = response.json()
if response.status_code == 200:
    code = processed_frame_base64['code']
    print(code)
    
        
