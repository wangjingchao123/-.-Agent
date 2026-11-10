# -*- coding: utf-8 -*-
"""Tab1 定式识别升级版验证（谷值检测+加权+投票+双重保护）"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.spatial.distance import cosine
from scipy.signal import find_peaks
import app

ok = 0
fail = 0
def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {name}")
    else:
        fail += 1
        print(f"  [FAIL] {name} {extra}")

print("== 1. 数据加载（向量加权）==")
app.load_data()
check("24个定式", len(app.keyframe_templates) == 24)
t0 = app.keyframe_templates[0]
check("vector 99维", len(t0["vector"]) == 99)

print("== 2. 加权一致性：模板vs自身余弦≈1 ==")
v0 = np.array(t0["vector"], dtype=np.float32)
sim_self = 1.0 - cosine(v0, v0)
check("自身相似度>0.999", sim_self > 0.999, f"sim={sim_self:.4f}")

print("== 3. 加权区分度：交叉相似度明显低 ==")
v1 = np.array(app.keyframe_templates[1]["vector"], dtype=np.float32)
sim_cross = 1.0 - cosine(v0, v1)
check(f"交叉[{t0['template_name']}]vs[{app.keyframe_templates[1]['template_name']}] 与自身有区分(gap>0.05)",
      sim_self - sim_cross > 0.05, f"self={sim_self:.4f} cross={sim_cross:.4f}")

print("== 4. 空输入防护 ==")
img, c, s, n = app.run_recognize(None, "无特殊情况")
check("空视频", img is None and s == 0.0 and n == "未知")

print("== 5. 谷值检测（合成静止-动-静止序列）==")
base = v0
rng = np.random.default_rng(0)
vecs_list = []
for phase in range(5):
    noise = 0.01 if phase % 2 == 0 else 0.3
    for _ in range(20):
        vecs_list.append(base + rng.normal(0, noise, base.shape).astype(np.float32))
vecs = np.array(vecs_list)
diffs = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
valleys, _ = find_peaks(-diffs, distance=app.MIN_VALLEY_DIST)
check("检出≥2个谷值（定式停顿点）", len(valleys) >= 2, f"valleys={len(valleys)}")

print("== 6. 真实视频 run_recognize（新逻辑）==")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vids = sorted(glob.glob(os.path.join(root, "uploads", "*.mp4")))
if vids:
    for v in vids[:3]:
        img7, c7, s7, n7 = app.run_recognize(v, "膝关节炎")
        status = "识别成功" if img7 is not None else "拒绝"
        print(f"  {os.path.basename(v)}: {status} 招式={n7} 得分={s7}")
        print(f"    评语: {c7[:100]}")
    check("run_recognize 全链路不崩溃", True)
else:
    print("  (无上传视频)")

print("== 7. Tab2 未受影响：自比对仍100 ==")
name0 = t0["template_name"]
tpl = app.seq_library[name0]
sm = app.smooth_leg_sequence(tpl.copy())
dist, *_ = app.dtw_with_path(sm, tpl)
check("Tab2自比对dist<0.01", dist < 0.01, f"dist={dist:.4f}")

print(f"\n结果: {ok} PASS / {fail} FAIL")
sys.exit(1 if fail else 0)