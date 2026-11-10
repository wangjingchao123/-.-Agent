# -*- coding: utf-8 -*-
"""标定 RECOG_THRESH：quick_compare 在归一化序列上的值分布
- 自比对（应=0）
- 同招式 + 高斯噪声（模拟不同人做同一招式的合理差异）
- 交叉：不同招式两两（误识别风险下限）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from taiji_joycode_submit import (
    preload_templates, template_library, quick_compare, RECOG_THRESH,
)


def main():
    preload_templates()
    names = list(template_library.keys())
    print(f"模板数：{len(names)}\n")

    # 1. 自比对
    self_d = [quick_compare(template_library[n], template_library[n]) for n in names]
    print(f"[自比对] max={max(self_d):.6f}（应=0）")

    # 2. 同招式 + 高斯噪声（归一化坐标值域约-2.7~2.6）
    rng = np.random.default_rng(42)
    for sigma in (0.05, 0.10, 0.20, 0.30, 0.50):
        ds = []
        for n in names:
            t = template_library[n]
            noisy = t + rng.normal(0, sigma, t.shape).astype(np.float32)
            ds.append(quick_compare(noisy, t))
        print(f"[同招式+噪声σ={sigma:.2f}] min={min(ds):.4f} mean={np.mean(ds):.4f} max={max(ds):.4f}")

    # 2b. 帧序打乱（模拟完全不同动作的极端对照）
    shuf_d = []
    for n in names:
        t = template_library[n]
        idx = rng.permutation(t.shape[0])
        shuf_d.append(quick_compare(t[idx], t))
    print(f"[帧序打乱] min={min(shuf_d):.4f} mean={np.mean(shuf_d):.4f} max={max(shuf_d):.4f}")

    # 3. 交叉：不同招式两两
    cross = []
    cross_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = quick_compare(template_library[names[i]], template_library[names[j]])
            cross.append(d)
            cross_pairs.append((d, names[i], names[j]))
    cross = np.array(cross)
    print(f"\n[交叉] 共{len(cross)}对 min={cross.min():.4f} mean={cross.mean():.4f} max={cross.max():.4f}")
    print(f"[交叉] p5={np.percentile(cross,5):.4f} p25={np.percentile(cross,25):.4f} p50={np.percentile(cross,50):.4f}")
    cross_pairs.sort()
    print("[交叉] 最相似5对：")
    for d, a, b in cross_pairs[:5]:
        print(f"    {d:.4f}  {a}  vs  {b}")

    print(f"\n当前 RECOG_THRESH = {RECOG_THRESH}")
    print("建议：阈值应 > 同招式合理差异上限，且 < 交叉最小值（否则误识别）")


if __name__ == "__main__":
    main()