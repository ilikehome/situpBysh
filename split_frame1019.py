import logging

import cv2
from ultralytics import YOLO

# 设置 Ultralytics 的日志级别为 WARNING，避免不必要的日志输出
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# 加载 YOLOv8n-pose 模型
model = YOLO('yolov8n-pose.pt')

# 指定要处理的视频文件路径
video_path = 'rasie_hand_fail_1.mp4'
# 创建视频捕获对象
cap = cv2.VideoCapture(video_path)

# 假设将视频帧分为 5 个部分，可根据实际情况调整
n = 5

# 定义一个函数用于过滤出符合特定条件的人
def filter_people(results):
    """
    这个函数用于过滤出脚部位置满足特定条件的检测结果。

    参数：
    results (list): YOLO 模型的检测结果列表，每个结果包含关键点信息。

    返回：
    filtered_results (list): 过滤后的检测结果列表。
    """
    filtered_results = []
    for result in results:
        keypoints = result.keypoints.xyn[0]
        # 确保关键点存在且数量大于 0
        if keypoints is not None and len(keypoints) > 0:
            # 获取左右脚的 y 坐标
            left_foot_y = keypoints[15][1]
            right_foot_y = keypoints[16][1]
            # 判断左右脚至少有一只脚的 y 坐标在指定范围内
            if (0.98 > left_foot_y > 0.25) or (0.98 > right_foot_y > 0.25):
                filtered_results.append(result)
    return filtered_results

# 定义一个函数用于划分视频帧并检测人所在的小矩形区域
def divide_frame_and_detect_people(frame):
    """
    此函数将视频帧划分为 n 个小矩形区域，并检测每个区域中的人。

    参数：
    frame (numpy.ndarray): 视频帧图像。

    返回：
    people_dict (dict): 包含小矩形区域序号和对应检测结果的字典。
    """
    height, width, _ = frame.shape
    # 每个小矩形区域占总宽度的比例
    slice_percentage = 1 / n
    people_dict = {}
    results = model(frame)
    filtered_results = filter_people(results)
    for i in range(n - 1, -1, -1):
        # 计算起始和结束位置的百分比
        start_percentage = i * slice_percentage
        end_percentage = start_percentage + slice_percentage
        for result in filtered_results:
            keypoints = result.keypoints.xyn[0]
            if keypoints is not None and len(keypoints) > 0:
                center_x = (keypoints[0][0] + keypoints[15][0]) / 2
                # 判断人的中心点是否在当前小矩形区域内
                if start_percentage <= center_x < end_percentage:
                    # 如果小矩形区域序号为偶数
                    if i % 2 == 0:
                        # 如果字典为空或者当前人的头部 y 坐标大于字典中已有的人的头部 y 坐标
                        if not people_dict or keypoints[0][1] > list(people_dict.values())[0].keypoints.xyn[0][0][1]:
                            people_dict[n - i] = result
                    # 如果小矩形区域序号为奇数
                    else:
                        # 如果字典为空或者当前人的头部 y 坐标小于字典中已有的人的头部 y 坐标
                        if not people_dict or keypoints[0][1] < list(people_dict.values())[0].keypoints.xyn[0][0][1]:
                            people_dict[n - i] = result
                    break
    return people_dict

# 开始处理视频
while cap.isOpened():
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 划分视频帧并检测人所在的小矩形区域，得到包含小矩形区域序号和对应检测结果的字典
    people_info = divide_frame_and_detect_people(frame)
    print(people_info)

    # 在这两行代码中间加入的代码
    for index, result in people_info.items():
        if result.boxes.xyxy.shape[0] > 0:
            x_min, y_min, x_max, y_max = result.boxes.xyxy[0].tolist()
            # 在人周围绘制绿色矩形框
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            # 在人上方添加序号，从右到左标注
            cv2.putText(frame, str(index), (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示处理后的视频帧
    cv2.imshow('Frame', frame)
    # 如果按下'q'键，则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象
cap.release()
# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()