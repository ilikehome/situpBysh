import cv2
import numpy as np
import torch
from ultralytics import YOLO
from memory_profiler import profile

    # 获取图片的高度和宽度
    height, width, _ = frame.shape
    # 计算图片的中心坐标
    pic_center_x = width // 2
    pic_center_y = height // 2
    return pic_center_x, pic_center_y

def get_person_center_xy(one_person_point):
    valid_points = []
    for keypoint_idx in range(one_person_point.shape[0]):
        x, y, confidence = one_person_point[keypoint_idx]
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


# 获取123点位在data中的索引：取与中心点最近的左下点1，上点2，右下点3
def get_person123_inx(person_center_point_arr_np, center_xy):
    # 初始化最近点为无穷远
    above_closest_2 = np.array([0, 0])
    left_below_closest_1 = np.array([0, 0])
    right_below_closest_3 = np.array([0, 0])
    above_min_distance = np.inf
    left_below_min_distance = np.inf
    right_below_min_distance = np.inf
    left_below_closest_1_inx, above_closest_2_inx, right_below_closest_3_inx = None, None, None

    # 遍历所有点找到满足条件的最近点
    for inx, point in enumerate(person_center_point_arr_np):
        if point[1] < center_xy[1]:
            d = distance(point, center_xy)
            if d < above_min_distance:
                above_min_distance = d
                above_closest_2 = point
                above_closest_2_inx = inx
        elif point[0] < center_xy[0] and point[1] > center_xy[1]:
            d = distance(point, center_xy)
            if d < left_below_min_distance:
                left_below_min_distance = d
                left_below_closest_1 = point
                left_below_closest_1_inx = inx
        elif point[0] > center_xy[0] and point[1] > center_xy[1]:
            d = distance(point, center_xy)
            if d < right_below_min_distance:
                right_below_min_distance = d
                right_below_closest_3 = point
                right_below_closest_3_inx = inx

    return left_below_closest_1_inx, above_closest_2_inx, right_below_closest_3_inx

def calc_angle(side_a, vertex, side_b):
    # 计算向量 BA 和 BC
    ba = np.array(side_a) - np.array(vertex)
    bc = np.array(side_b) - np.array(vertex)

    # 计算向量的模
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # 计算向量的点积
    dot_product = np.dot(ba, bc)

    # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude_ba * magnitude_bc)

    # 使用反余弦函数得到夹角的弧度值
    theta_rad = np.arccos(cos_theta)

    # 将弧度转换为度数
    theta_deg = np.rad2deg(theta_rad)

    print("theta_deg = " + str(theta_deg))
    return theta_deg

def calc_body_bend_angle(body_points):
    upper_body_indices = [0, 1, 2, 3, 4, 5, 6] #从鼻子到肩膀，上半身最终使用平均值
    hip_indices = [11, 12]#臀部
    knee_indices = [13, 14]#膝盖
    upper_body_sum_x = 0
    upper_body_sum_y = 0
    upper_body_count = 0#从鼻子到肩膀，上半身使用平均值
    hip_sum_x = 0
    hip_sum_y = 0
    hip_count = 0#臀部有两个点，也使用平均值
    knee_sum_x = 0
    knee_sum_y = 0
    knee_count = 0#膝盖有两个点，也使用平均值
    for i, (x, y, conf) in enumerate(body_points):
        if conf > 0.5 and x >1 and y>1:#置信度低的不要
            if i in upper_body_indices:
                upper_body_sum_x += x
                upper_body_sum_y += y
                upper_body_count += 1
            elif i in hip_indices:
                hip_sum_x += x
                hip_sum_y += y
                hip_count += 1
            elif i in knee_indices:
                knee_sum_x += x
                knee_sum_y += y
                knee_count += 1
    upper_body_avg_x = upper_body_sum_x / upper_body_count if upper_body_count > 0 else 0
    upper_body_avg_y = upper_body_sum_y / upper_body_count if upper_body_count > 0 else 0
    hip_avg_x = hip_sum_x / hip_count if hip_count > 0 else 0
    hip_avg_y = hip_sum_y / hip_count if hip_count > 0 else 0
    knee_avg_x = knee_sum_x / knee_count if knee_count > 0 else 0
    knee_avg_y = knee_sum_y / knee_count if knee_count > 0 else 0
    return calc_angle((upper_body_avg_x, upper_body_avg_y), (hip_avg_x, hip_avg_y), (knee_avg_x, knee_avg_y))

def count_numbers(position_inx, lst):
    sit_up_angle = 30#第一排要卷到35度
    if position_inx == 1:
        sit_up_angle = 48#第二排要卷到45度

    count = 0
    in_range = False
    for num in lst:
        if num > 90:#大于90度认为是躺下了
            if not in_range:
                in_range = True
        elif num < sit_up_angle and in_range:#小于35度认为是仰卧起坐ok了
            count += 1
            in_range = False
    return count

@profile
def main_function():
    # 加载 YOLOv8n-pose 模型
    model = YOLO("yolov8n-pose.pt")

    # 打开视频设备（这里以摄像头为例，你也可以指定视频文件路径）
    cap = cv2.VideoCapture(r"C:\Users\ilike\PycharmProjects\ai-sports-algorithm-test\situp\3.mp4")

    g_mem_person_angle_dict = [[], [], []]
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break

        # 使用模型进行预测
        results = model.predict(frame)
        # 在中心位置绘制一个点
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # 绘制关键点
        for result in results:
            person_center_point_arr = []
            for points_for_1person in result.keypoints.data:
                person_center_xy = get_person_center_xy(points_for_1person)
                if person_center_xy is not None:
                    person_center_point_arr.append(person_center_xy)

            p1, p2, p3 = get_person123_inx(np.array(person_center_point_arr), (center_x, center_y))
            if p1 is not None:
                cv2.putText(frame, '1', person_center_point_arr[p1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                p1_angle = calc_body_bend_angle(result.keypoints.data[p1])
                g_mem_person_angle_dict[0].append(p1_angle)
            if p2 is not None:
                cv2.putText(frame, '2', person_center_point_arr[p2], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                p2_angle = calc_body_bend_angle(result.keypoints.data[p2])
                g_mem_person_angle_dict[1].append(p2_angle)
            if p3 is not None:
                cv2.putText(frame, '3', person_center_point_arr[p3], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                p3_angle = calc_body_bend_angle(result.keypoints.data[p3])
                g_mem_person_angle_dict[2].append(p3_angle)

        # 显示结果
        cv2.imshow("Pose Detection", frame)
        for i,angle_lst in enumerate(g_mem_person_angle_dict):
            if len(angle_lst) > 0:
                print(str(i) + " is situping, count= " + str(count_numbers(i, angle_lst)))

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_function()
