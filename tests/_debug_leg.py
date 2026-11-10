# -*- coding: utf-8 -*-
"""调试：腿部偏差异常巨大问题定位"""
import sys
sys.path.insert(0, ".")
import numpy as np
from taiji_joycode_submit import (
    load_template, normalize_sequence, smooth_leg_sequence,
    dtw_with_path, LEG_IDS, UPPER_IDS,
)

tpl = load_template("起势")
print("tpl dtype:", tpl.dtype, "shape:", tpl.shape)
print("tpl腿部(点23-32)范围:", tpl[:, 69:99].min(), "~", tpl[:, 69:99].max())
print("tpl上身(点0-22)范围:", tpl[:, :69].min(), "~", tpl[:, :69].max())

rng = np.random.default_rng(42)
noisy = normalize_sequence(tpl + rng.normal(0, 0.01, tpl.shape))
print("\nnoisy dtype:", noisy.dtype)
print("noisy腿部范围:", noisy[:, 69:99].min(), "~", noisy[:, 69:99].max())

sm = smooth_leg_sequence(noisy)
print("\nsmooth后dtype:", sm.dtype)
print("smooth后腿部范围:", sm[:, 69:99].min(), "~", sm[:, 69:99].max())
print("smooth后上身范围:", sm[:, :69].min(), "~", sm[:, :69].max())

# 直接对比第0帧
print("\n第0帧腿部差:", np.linalg.norm(sm[0, 69:99] - tpl[0, 69:99]))
print("第60帧腿部差:", np.linalg.norm(sm[60, 69:99] - tpl[60, 69:99]))

# DTW后看path和pcl
dist, path, pc, pcu, pcl, kpc = dtw_with_path(sm, tpl)
print(f"\nDTW: dist={dist:.4f}  path长度={len(path)}")
print(f"pcu均值={np.mean(pcu):.4f}  pcl均值={np.mean(pcl):.4f}")
print(f"path前5: {path[:5]}")
print(f"pcl前5: {pcl[:5]}")
# 检查path是否对角
diag = sum(1 for a, b in path if a == b)
print(f"对角匹配数: {diag}/{len(path)}")