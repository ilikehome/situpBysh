import base64
import cv2
import numpy as np
import os
import json
import time
import cv2
import requests


url = 'https://tongdasports.com/algorithm/faceImport/22322'
data = [
    {
        "faceUrlList": ["https://img1.baidu.com/it/u=2300672649,2195936276&fm=253&fmt=auto&app=120&f=JPEG?w=608&h=379"],
        "name": "jiahui",
        "userId": 2,
        "gender": 1,
        "grade": "三年级"
    }
]
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers, timeout=30)
if response.status_code == 200:
    print(f"situp initialization")
else:
    print(f"situp initialization falied")

