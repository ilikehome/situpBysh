import torch
import numpy as np
from ultralytics import YOLO
import cv2
import traceback
from flask import Flask, request, jsonify

app = Flask(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("pre-trained_yolo/yolov8n-pose.pt").to(device)
max_loc_index = 5

#从左到右，位置索引2,3,4
g_mem_person_angle_dict = {}
g_location_index = []

def get_pic_center_xy(frame):
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
    # 将张量移动到 CPU 上再转换为 NumPy 数组
    side_a = np.array([side_a[0].cpu().item(), side_a[1].cpu().item()])
    vertex = np.array([vertex[0].cpu().item(), vertex[1].cpu().item()])
    side_b = np.array([side_b[0].cpu().item(), side_b[1].cpu().item()])
    # 计算向量 BA 和 BC
    ba = side_a - vertex
    bc = side_b - vertex

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
    return theta_deg


def calc_body_bend_angle(body_points):
    upper_body_indices = [0, 1, 2, 3, 4, 5, 6]  # 从鼻子到肩膀，上半身最终使用平均值
    hip_indices = [11, 12]  # 臀部
    knee_indices = [13, 14]  # 膝盖
    upper_body_sum_x = 0
    upper_body_sum_y = 0
    upper_body_count = 0  # 从鼻子到肩膀，上半身使用平均值
    hip_sum_x = 0
    hip_sum_y = 0
    hip_count = 0  # 臀部有两个点，也使用平均值
    knee_sum_x = 0
    knee_sum_y = 0
    knee_count = 0  # 膝盖有两个点，也使用平均值
    for i, (x, y, conf) in enumerate(body_points):
        if conf > 0.5 and x > 1 and y > 1:  # 置信度低的不要
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

    if upper_body_count == 0 or hip_count == 0 or knee_count == 0:
        return None
    upper_body_avg_x = upper_body_sum_x / upper_body_count
    upper_body_avg_y = upper_body_sum_y / upper_body_count
    hip_avg_x = hip_sum_x / hip_count
    hip_avg_y = hip_sum_y / hip_count
    knee_avg_x = knee_sum_x / knee_count
    knee_avg_y = knee_sum_y / knee_count
    return calc_angle((upper_body_avg_x, upper_body_avg_y), (hip_avg_x, hip_avg_y), (knee_avg_x, knee_avg_y))


def count_numbers(position_inx, lst):
    sit_up_angle = 30  # 第一排要卷到30度
    if position_inx == 1:
        sit_up_angle = 48  # 第二排要卷到48度

    count = 0
    in_range = False
    for num in lst:
        if num > 90:  # 大于90度认为是躺下了
            if not in_range:
                in_range = True
        elif num < sit_up_angle and in_range:  # 小于35度认为是仰卧起坐ok了
            count += 1
            in_range = False
    return count


@app.route('/algorithm/situp/init', methods=['POST'])
# @profile
def situp_init():
    global g_mem_person_angle_dict
    global g_location_index
    g_mem_person_angle_dict = {}
    torch.cuda.empty_cache()

    try:
        location_index = str(request.form.get('locationIndex'))
        location_index = location_index.split(',')
        g_location_index = [int(item) for item in location_index]
        if not all(index in [2, 3, 4] for index in g_location_index):
            raise ValueError("locationIndex should be within 2, 3, or 4.")

        for inx in g_location_index:
            g_mem_person_angle_dict[inx] = []

        return jsonify({
            'code': 0,
            'msg': 'success'
        }), 200

    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e) + ', ' + traceback.format_exc()}), 500


@app.route("/algorithm/situp/count", methods=['POST'])
def situp():
    # 接收OpenCV编码后的字节数组
    file = request.files['frame']

    file_bytes = file.read()
    # pdb.set_trace()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    # pdb.set_trace()
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 使用模型进行预测
    results = model.predict(frame)
    center_x, center_y = get_pic_center_xy(frame)

    # 绘制关键点
    for result in results:
        person_center_point_arr = []
        for points_for_1person in result.keypoints.data:
            person_center_xy = get_person_center_xy(points_for_1person)
            if person_center_xy is not None:
                person_center_point_arr.append(person_center_xy)

        p2, p3, p4 = get_person123_inx(np.array(person_center_point_arr), (center_x, center_y))
        if (p2 is not None) and (2 in g_mem_person_angle_dict):
            p2_angle = calc_body_bend_angle(result.keypoints.data[p2])
            if p2_angle is not None:
                g_mem_person_angle_dict[2].append(p2_angle)
        if (p3 is not None) and (3 in g_mem_person_angle_dict):
            p3_angle = calc_body_bend_angle(result.keypoints.data[p3])
            if p3_angle is not None:
                g_mem_person_angle_dict[3].append(p3_angle)
        if (p4 is not None) and (4 in g_mem_person_angle_dict):
            p4_angle = calc_body_bend_angle(result.keypoints.data[p4])
            if p4_angle is not None:
                g_mem_person_angle_dict[4].append(p4_angle)

    data = {}
    for i, angle_lst in g_mem_person_angle_dict.items():
        if len(angle_lst) > 0:
            data[i] = count_numbers(i, angle_lst)
        else:
            data[i] = 0

    results = {'data': data, 'msg': 'success', 'code': '0'}
    return jsonify(results), 200


@app.route('/algorithm/situp/finish', methods=['POST'])
# @profile
def situp_finish():
    global g_mem_person_angle_dict
    global g_location_index
    try:
        data = {}
        for i, angle_lst in g_mem_person_angle_dict.items():
            if len(angle_lst) > 0:
                data[i] = count_numbers(i, angle_lst)
            else:
                data[i] = 0

        results = {'data': data, 'msg': 'success', 'code': '0'}

        g_mem_person_angle_dict = {}
        g_location_index = []

        return jsonify(results), 200
    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e) + ', ' + traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003, debug=False)
