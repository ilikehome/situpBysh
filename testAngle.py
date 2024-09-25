import numpy as np
import math

def calc_angle(side_a, vertex, side_b):
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

# 示例点的坐标
a = np.array([0, 1])
b = np.array([0, 0])
c = np.array([3, 0])

angle = calc_angle(a, b, c)
print(f"夹角的度数为：{angle} 度")