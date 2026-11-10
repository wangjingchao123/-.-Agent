# -*- coding: utf-8 -*-
"""验证fastdtw安装 + 未破坏mediapipe链"""
import numpy as np

# 1. numpy版本未被升级
print("numpy:", np.__version__)
assert np.__version__.startswith("1.26"), "numpy被升级了！"

# 2. mediapipe仍可导入
import mediapipe
print("mediapipe:", mediapipe.__version__)

# 3. fastdtw基本功能：返回(distance, path)元组
from fastdtw import fastdtw
x = np.array([[1, 0], [1, 1], [2, 1], [3, 1]], dtype=float)
y = np.array([[1, 0], [2, 1], [3, 1]], dtype=float)
d, path = fastdtw(x, y)
print("fastdtw distance:", round(d, 4))
print("path len:", len(path), "首:", path[0], "尾:", path[-1])
assert d >= 0
assert isinstance(path, list) and isinstance(path[0], tuple)

# 4. 自定义距离函数
d2, _ = fastdtw(x, y, dist=lambda a, b: np.linalg.norm(a - b))
print("custom dist:", round(d2, 4))

# 5. 一维序列
a = np.array([1, 2, 3, 4, 5], dtype=float)
b = np.array([1, 2, 3, 4], dtype=float)
d3, p3 = fastdtw(a, b)
print("1D dist:", round(d3, 4), "len:", len(p3))

print("\nALL OK")