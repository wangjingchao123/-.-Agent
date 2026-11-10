# -*- coding: utf-8 -*-
"""自动识别对照：模板自识别（加噪声）+ 真实视频最小距离诊断"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import app

app.load_data()
names = list(app.seq_library.keys())

print("== A. 模板自识别（加噪声 sigma=0.02）==")
hit = 0
for name in names[:8]:
    tpl = app.seq_library[name]
    rng = np.random.default_rng(42)
    noisy = tpl + rng.normal(0, 0.02, tpl.shape).astype(np.float32)
    found, dist = app.auto_select_template(noisy)
    ok = (found == name)
    hit += ok
    print(f"  {name} -> {found} (dist={dist:.4f}) {'OK' if ok else 'MISS'}")
print(f"  自识别命中: {hit}/8")

print("\n== B. 真实视频自动识别最小距离 ==")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vids = sorted(glob.glob(os.path.join(root, "uploads", "*.mp4")))
for v in vids[:4]:
    result = app.extract_seq_from_video(v)
    if result is None:
        print(f"  {os.path.basename(v)}: 提取失败")
        continue
    seq = result[0]
    # 列出前3小距离
    dists = []
    for name in names:
        d = app.quick_compare(seq, app.seq_library[name])
        dists.append((d, name))
    dists.sort()
    best_name, best_d = app.auto_select_template(seq)
    tag = "识别成功" if best_name else "超阈值拒绝"
    print(f"  {os.path.basename(v)}: 最优3={dists[:3]} -> {tag}")