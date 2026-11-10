# -*- coding: utf-8 -*-
"""标定 MATCH_DIST_THRESHOLD：dtw_with_path 归一化dist在以下场景的分布
- 自比对（应≈0）
- 同招式+噪声（合理差异上限）
- 交叉模板（不同招式）
- 真实视频 vs 各模板（含匹配/非匹配）
"""
import sys
sys.path.insert(0, ".")
from taiji_joycode_submit import (
    preload_templates, template_library, load_template,
    dtw_with_path, smooth_leg_sequence, extract_seq_from_video,
    DANGER_DIST, ACTION_LIST,
)
import numpy as np
from pathlib import Path

print(f"当前 DANGER_DIST(score=0线) = {DANGER_DIST}\n")

preload_templates()
names = list(template_library.keys())

print("=== 1. 自比对 ===")
d_self = dtw_with_path(template_library["起势"], template_library["起势"])[0]
print(f"  起势自比对 dist={d_self:.4f}")

print("\n=== 2. 同招式+噪声（归一化坐标值域约-2.7~2.6）===")
rng = np.random.default_rng(42)
for sigma in (0.05, 0.10, 0.20, 0.30):
    ds = []
    for n in names:
        t = template_library[n]
        noisy = t + rng.normal(0, sigma, t.shape).astype(np.float32)
        noisy = smooth_leg_sequence(noisy)
        ds.append(dtw_with_path(noisy, t)[0])
    print(f"  σ={sigma:.2f}: min={min(ds):.4f} mean={np.mean(ds):.4f} max={max(ds):.4f}")

print("\n=== 3. 交叉模板（不同招式两两，抽样12对）===")
cross = []
rng2 = np.random.default_rng(1)
for _ in range(12):
    i, j = rng2.choice(len(names), 2, replace=False)
    d = dtw_with_path(template_library[names[i]], template_library[names[j]])[0]
    cross.append((d, names[i], names[j]))
cross.sort()
for d, a, b in cross:
    print(f"  {d:.4f}  {a} vs {b}")
print(f"  交叉 min={cross[0][0]:.4f} max={cross[-1][0]:.4f}")

print("\n=== 4. 真实视频 vs 全部24模板 ===")
vids = ["uploads/7db8e9fc793da435ac23a44e8db55342.mp4",
        "uploads/1469ceee5dc981d157a72c8c992c3ab5.mp4"]
for vid in vids:
    if not Path(vid).exists():
        print(f"  {vid} 不存在，跳过")
        continue
    result = extract_seq_from_video(vid)
    if result is None:
        print(f"  {vid} 提取失败，跳过")
        continue
    nu, _, _ = result
    nu = smooth_leg_sequence(nu)
    dists = []
    for n in names:
        d = dtw_with_path(nu, template_library[n])[0]
        dists.append((d, n))
    dists.sort()
    print(f"\n  {Path(vid).name}:")
    print(f"    最匹配3个：")
    for d, n in dists[:3]:
        print(f"      {d:.4f}  {n}")
    print(f"    最不匹配3个：")
    for d, n in dists[-3:]:
        print(f"      {d:.4f}  {n}")

print("\n=== 5. 建议 ===")
print("MATCH_DIST_THRESHOLD 应 > 同招式合理差异上限，且 < 交叉最小值")
print("（真实视频若为非匹配招式，dist应落在交叉区间）")