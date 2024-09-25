import numpy as np
from scipy.interpolate import splrep, splev

# 生成一些示例数据点
x = np.linspace(0, 10, 20)
y = np.sin(x)

# 使用 splrep 进行 B 样条拟合
tck = splrep(x, y, k=3)

# 可以使用 splev 在新的点上评估拟合曲线
x_new = np.linspace(0, 10, 100)
y_new = splev(x_new, tck)

import matplotlib.pyplot as plt

plt.plot(x, y, 'o', label='Original data')
plt.plot(x_new, y_new, label='Fitted curve')
plt.legend()
plt.show()