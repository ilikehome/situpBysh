import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 加载 YOLOv8n-pose 模型
model = YOLO("yolov8n-pose.pt")

# 打开视频设备（这里以摄像头为例，你也可以指定视频文件路径）
cap = cv2.VideoCapture(r"C:\Users\ilike\PycharmProjects\situpBysh\jumping.mp4")

def get_person_center_xy(one_person_point):
    valid_points = []
    for keypoint_idx in range(one_person_point.shape[0]):#one_person_point.shape[0]是17，因为固定是17个关键点
        x, y, confidence = one_person_point[keypoint_idx]#confidence是可信度
        if x >= 1 and y >= 1 and confidence >= 0.3:
            valid_points.append((x, y))
    if valid_points:
        valid_points = torch.tensor(valid_points)
        mean_x = valid_points[:, 0].mean()
        mean_y = valid_points[:, 1].mean()
        return int(mean_x.item()), int(mean_y.item())


# 计算距离函数
def distance(point1, point2):
    return np.linalg.norm(point1 - point2)



def get_person12345_inx(points):
    return


g_mem_person_angle_dict = [[], [], []]
while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 使用模型进行预测
    results = model.predict(frame)

    # 绘制关键点
    for result in results:
        person_center_point_arr = []
        for points_for_1person in result.keypoints.data:#5个人、每个人的身体关键点信息都在data中。 points_for_1person是一个人的信息
            person_center_xy = get_person_center_xy(points_for_1person)
            if person_center_xy is not None:
                person_center_point_arr.append(person_center_xy)

        person12345_inx = np.argsort([point[0] for point in person_center_point_arr]) # 获取12345点位人员信息在yolo result.keypoints.data中的索引.这个代码要自己改下！！！！！这儿只是为了走通！！！！！！
        if person12345_inx[0] is not None:
            cv2.putText(frame, '1', person_center_point_arr[person12345_inx[0]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            p1_left_hip = result.keypoints.data[person12345_inx[0]][11] #result.keypoints.data[person12345_inx[0]]是第一个人的17个身体关键点。[11]是左屁股
            p1_right_hip = result.keypoints.data[person12345_inx[0]][12]#[12]是左屁股
            cv2.circle(frame, (int(p1_left_hip[0].item()), int(p1_left_hip[1].item())), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(p1_right_hip[0].item()), int(p1_right_hip[1].item())), 5, (0, 0, 255), -1)
        if person12345_inx[1] is not None:
            cv2.putText(frame, '2', person_center_point_arr[person12345_inx[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            p2_left_hip = result.keypoints.data[person12345_inx[1]][11]#result.keypoints.data[person12345_inx[1]]是第一个人的17个身体关键点。[11]是左屁股
            p2_right_hip = result.keypoints.data[person12345_inx[1]][12]
            cv2.circle(frame, (int(p2_left_hip[0].item()), int(p2_left_hip[1].item())), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(p2_right_hip[0].item()), int(p2_right_hip[1].item())), 5, (0, 0, 255), -1)
        if person12345_inx[2] is not None:
            cv2.putText(frame, '3', person_center_point_arr[person12345_inx[2]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            p3_left_hip = result.keypoints.data[person12345_inx[2]][11]#result.keypoints.data[person12345_inx[1]]是第一个人的17个身体关键点。[11]是左屁股
            p3_right_hip = result.keypoints.data[person12345_inx[2]][12]
            cv2.circle(frame, (int(p3_left_hip[0].item()), int(p3_left_hip[1].item())), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(p3_right_hip[0].item()), int(p3_right_hip[1].item())), 5, (0, 0, 255), -1)
        if person12345_inx[3] is not None:
            cv2.putText(frame, '4', person_center_point_arr[person12345_inx[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            p4_left_hip = result.keypoints.data[person12345_inx[3]][11]#result.keypoints.data[person12345_inx[1]]是第一个人的17个身体关键点。[11]是左屁股
            p4_right_hip = result.keypoints.data[person12345_inx[3]][12]
            cv2.circle(frame, (int(p4_left_hip[0].item()), int(p4_left_hip[1].item())), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(p4_right_hip[0].item()), int(p4_right_hip[1].item())), 5, (0, 0, 255), -1)
        if person12345_inx[4] is not None:
            cv2.putText(frame, '5', person_center_point_arr[person12345_inx[4]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            p5_left_hip = result.keypoints.data[person12345_inx[4]][11]#result.keypoints.data[person12345_inx[1]]是第一个人的17个身体关键点。[11]是左屁股
            p5_right_hip = result.keypoints.data[person12345_inx[4]][12]
            cv2.circle(frame, (int(p5_left_hip[0].item()), int(p5_left_hip[1].item())), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(p5_right_hip[0].item()), int(p5_right_hip[1].item())), 5, (0, 0, 255), -1)
        if len(person12345_inx) == 6:
            cv2.putText(frame, '6', person_center_point_arr[person12345_inx[5]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            p6_left_hip = result.keypoints.data[person12345_inx[5]][11]#result.keypoints.data[person12345_inx[1]]是第一个人的17个身体关键点。[11]是左屁股
            p6_right_hip = result.keypoints.data[person12345_inx[5]][12]
            cv2.circle(frame, (int(p6_left_hip[0].item()), int(p6_left_hip[1].item())), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(p6_right_hip[0].item()), int(p6_right_hip[1].item())), 5, (0, 0, 255), -1)

    # 显示结果
    cv2.imshow("Pose Detection", frame)


    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
