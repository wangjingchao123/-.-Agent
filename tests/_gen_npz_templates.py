# -*- coding: utf-8 -*-
"""从 saved_templates_final.json 生成 templates/*.npz 模板文件"""
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

TARGET_SEQ_LEN = 120
NUM_KEYPOINTS = 33

SRC = "saved_templates_final.json"
OUT_DIR = Path("templates")
OUT_DIR.mkdir(exist_ok=True)

with open(SRC, "r", encoding="utf-8") as f:
    raw = json.load(f)

print("=== 模板名列表 ===")
for name in raw.keys():
    print(f"  {name}")

print("\n=== 生成 .npz ===")
for name, item in raw.items():
    frames = item.get("frames") if isinstance(item, dict) else item
    if not isinstance(frames, list) or len(frames) < 20:
        print(f"[跳过] {name}：帧数不足")
        continue
    mats = []
    ok = True
    for fr in frames:
        if not isinstance(fr, list) or len(fr) != NUM_KEYPOINTS:
            ok = False
            break
        arr = np.zeros((NUM_KEYPOINTS, 3))
        for pt in fr:
            if isinstance(pt, dict) and "idx" in pt:
                arr[pt["idx"]] = [pt["x"], pt["y"], pt["z"]]
        mats.append(arr.flatten())
    if not ok:
        print(f"[跳过] {name}：关键点格式异常")
        continue
    seq = np.array(mats, dtype=np.float32)
    # 归一化到统一长度
    n, feat = seq.shape
    if n != TARGET_SEQ_LEN:
        old_x = np.linspace(0, 1, n)
        new_x = np.linspace(0, 1, TARGET_SEQ_LEN)
        out = np.empty((TARGET_SEQ_LEN, feat))
        for col in range(feat):
            f = interp1d(old_x, seq[:, col], kind="linear", fill_value="extrapolate")
            out[:, col] = f(new_x)
        seq = out.astype(np.float32)
    np.savez(OUT_DIR / f"{name}.npz", seq=seq)
    print(f"  {name}.npz  shape={seq.shape}")

print(f"\n完成，共生成 {len(list(OUT_DIR.glob('*.npz')))} 个 .npz 文件")