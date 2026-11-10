# -*- coding: utf-8 -*-
"""app.py 竞赛定稿全链路验证"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
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

print("== 1. 数据加载 ==")
app.load_data()
check("24个定式", len(app.keyframe_templates) == 24, f"实际{len(app.keyframe_templates)}")
check("24个序列模板", len(app.seq_library) == 24, f"实际{len(app.seq_library)}")

print("== 2. 定式向量/序列维度 ==")
t0 = app.keyframe_templates[0]
check("vector 99维", len(t0["vector"]) == 99, f"实际{len(t0['vector'])}")
seq0 = app.seq_library[t0["template_name"]]
check("seq (120,99)", seq0.shape == (120, 99), f"实际{seq0.shape}")
check("health_note 非空", bool(t0.get("health_note")))

print("== 3. 空输入防护 ==")
img, c, s, n = app.run_recognize(None, "正常")
check("run_recognize 空视频", img is None and s == 0.0)
s2, l2, r2, g2 = app.analyze("自动识别", None)
check("analyze 空视频", s2 == 0.0 and l2 == "未上传")

print("== 4. 自比对100分 ==")
name0 = t0["template_name"]
tpl = app.seq_library[name0]
sm = app.smooth_leg_sequence(tpl.copy())
dist, path, pc, pcu, pcl, kpc = app.dtw_with_path(sm, tpl)
score = app.score_from_distance(dist)
check(f"自比对[{name0}] dist<0.01", dist < 0.01, f"dist={dist:.4f}")
check("自比对得分>99", score > 99.0, f"score={score:.1f}")

print("== 5. 匹配门槛拦截交叉 ==")
names = list(app.seq_library.keys())
a, b = app.seq_library[names[0]], app.seq_library[names[1]]
sa = app.smooth_leg_sequence(a.copy())
d_cross, *_ = app.dtw_with_path(sa, b)
check(f"交叉[{names[0]}]vs[{names[1]}] dist>2.0",
      d_cross > app.MATCH_DIST_THRESHOLD, f"dist={d_cross:.4f}")

print("== 6. 真实视频 analyze（自动识别）==")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vids = sorted(glob.glob(os.path.join(root, "uploads", "*.mp4")))
if vids:
    v = vids[0]
    s3, l3, r3, g3 = app.analyze("自动识别", v)
    print(f"  视频: {os.path.basename(v)}")
    print(f"  得分={s3} 等级={l3} 抽样图={len(g3)}张")
    check("analyze 返回正常", isinstance(s3, (int, float)) and isinstance(r3, str) and isinstance(g3, list))
    check("报告非空", len(r3) > 20)
else:
    print("  (无上传视频，跳过)")

print("== 7. 真实视频 run_recognize ==")
if vids:
    v = vids[0]
    img7, c7, s7, n7 = app.run_recognize(v, "膝关节炎")
    print(f"  识别招式={n7} 得分={s7}")
    check("run_recognize 有输出", img7 is not None or "未" in c7 or "请" in c7)

print(f"\n结果: {ok} PASS / {fail} FAIL")
sys.exit(1 if fail else 0)