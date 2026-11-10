# -*- coding: utf-8 -*-
"""Tab1 全套招式特征校验验证（精修版：步型判定+混淆区分）"""
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

class FakeLM:
    """模拟mediapipe NormalizedLandmark对象（属性访问）"""
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def make_lm(overrides):
    """构造33点mediapipe对象骨架，默认站直"""
    lm = [FakeLM() for _ in range(33)]
    lm[0].y = 0.30   # nose
    lm[7].x, lm[7].y = 0.42, 0.32   # 左耳
    lm[8].x, lm[8].y = 0.58, 0.32   # 右耳
    lm[11].x, lm[11].y = 0.40, 0.40  # 左肩
    lm[12].x, lm[12].y = 0.60, 0.40  # 右肩
    lm[15].x, lm[15].y = 0.40, 0.40  # 左腕=肩高
    lm[16].x, lm[16].y = 0.60, 0.40  # 右腕=肩高
    lm[23].x, lm[23].y = 0.45, 0.55  # 左髋
    lm[24].x, lm[24].y = 0.55, 0.55  # 右髋
    lm[25].x, lm[25].y = 0.46, 0.70  # 左膝
    lm[26].x, lm[26].y = 0.54, 0.70  # 右膝
    lm[27].x, lm[27].y = 0.47, 0.85  # 左踝
    lm[28].x, lm[28].y = 0.53, 0.85  # 右踝
    for idx, vals in overrides.items():
        for k, v in vals.items():
            setattr(lm[idx], k, v)
    return lm

print("== 1. 起势：标准过 / 手举过高不过 ==")
check("起势标准通过", app.feature_verify("起势", make_lm({})) is True)
check("起势手举过头拒绝", app.feature_verify("起势", make_lm({15: {"y": 0.15}, 16: {"y": 0.15}})) is False)

print("== 2. 单鞭（一）右向 / 单鞭（二）左向 ==")
lm_d1 = make_lm({15: {"x": 0.55, "y": 0.70}, 16: {"x": 0.90, "y": 0.30}})
check("单鞭（一）右向通过", app.feature_verify("单鞭（一）", lm_d1) is True)
lm_d2 = make_lm({15: {"x": 0.10, "y": 0.70}, 16: {"x": 0.45, "y": 0.30}})
check("单鞭（二）左向通过", app.feature_verify("单鞭（二）", lm_d2) is True)
check("单鞭（一）左向拒绝(方向反)", app.feature_verify("单鞭（一）", lm_d2) is False)

print("== 3. 步型判定：弓步/虚步/仆步/提腿 ==")
lm_gongbu = make_lm({27: {"x": 0.25, "y": 0.85}, 28: {"x": 0.75, "y": 0.85},
                     25: {"x": 0.30, "y": 0.65}, 26: {"x": 0.65, "y": 0.70}})
check("弓步判定", app.is_gongbu(lm_gongbu) is True)
check("站直非弓步", app.is_gongbu(make_lm({})) is False)
check("站直是虚步(膝近伸直)", app.is_xubu(make_lm({})) is True)
lm_pubu = make_lm({23: {"y": 0.68}, 24: {"y": 0.68}})
check("深蹲是仆步", app.is_pubu(lm_pubu) is True)
check("站直非仆步", app.is_pubu(make_lm({})) is False)
lm_ti = make_lm({27: {"y": 0.55}})
check("抬脚是提腿", app.is_ti_tui(lm_ti) is True)
check("双脚着地非提腿", app.is_ti_tui(make_lm({})) is False)

print("== 4. 蹬脚：右蹬抬右脚 / 转身左蹬抬左脚 ==")
lm_rd = make_lm({28: {"y": 0.55}, 15: {"x": 0.30}, 16: {"x": 0.70}})
check("右蹬脚抬右脚通过", app.feature_verify("右蹬脚", lm_rd) is True)
lm_ld = make_lm({27: {"y": 0.55}, 15: {"x": 0.30}, 16: {"x": 0.70}})
check("转身左蹬脚抬左脚通过", app.feature_verify("转身左蹬脚", lm_ld) is True)
check("右蹬脚抬左脚拒绝(腿反)", app.feature_verify("右蹬脚", lm_ld) is False)

print("== 5. 高探马：虚步+眼前伸过 / 无前伸拒 ==")
lm_gtm = make_lm({15: {"y": 0.35, "z": -0.10}})
check("高探马眼前伸通过", app.feature_verify("高探马", lm_gtm) is True)
check("高探马无前伸拒绝", app.feature_verify("高探马", make_lm({})) is False)

print("== 6. 混淆区分：野马分鬃手侧展 / 搂膝拗步手近膝前推 ==")
lm_ym = make_lm({27: {"x": 0.25, "y": 0.85}, 28: {"x": 0.75, "y": 0.85},
                 25: {"x": 0.30, "y": 0.65}, 26: {"x": 0.65, "y": 0.70},
                 15: {"x": 0.15, "y": 0.25}, 16: {"x": 0.90, "y": 0.50}})
check("野马分鬃侧展通过", app.feature_verify("左右野马分鬃", lm_ym) is True)
lm_lx = make_lm({27: {"x": 0.25, "y": 0.85}, 28: {"x": 0.75, "y": 0.85},
                 25: {"x": 0.30, "y": 0.65}, 26: {"x": 0.65, "y": 0.70},
                 15: {"x": 0.32, "y": 0.65, "z": -0.10}, 16: {"x": 0.60, "y": 0.45}})
check("搂膝拗步近膝前推通过", app.feature_verify("左右搂膝拗步", lm_lx) is True)

print("== 7. 收势：身体直立手低过 ==")
lm_ss = make_lm({15: {"y": 0.65}, 16: {"y": 0.65}, 23: {"y": 0.50}, 24: {"y": 0.50}})
check("收势直立手低通过", app.feature_verify("收势", lm_ss) is True)
check("收势手高拒绝", app.feature_verify("收势", make_lm({15: {"y": 0.30}, 16: {"y": 0.30}})) is False)

print("== 8. 24式feature_verify全部不崩溃 ==")
app.load_data()
all_pass = True
for t in app.keyframe_templates:
    try:
        app.feature_verify(t["template_name"], make_lm({}))
    except Exception as e:
        all_pass = False
        print(f"    {t['template_name']} 异常: {e}")
check("24式不崩溃", all_pass)

print("== 9. 真实视频 run_recognize（精修版）==")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vids = sorted(glob.glob(os.path.join(root, "uploads", "*.mp4")))
for v in vids[:3]:
    img7, c7, s7, n7 = app.run_recognize(v, "无特殊情况")
    status = "识别成功" if img7 is not None else "拒绝"
    print(f"  {os.path.basename(v)}: {status} 招式={n7} 得分={s7}")
    if img7 is None:
        print(f"    原因: {c7[:60]}")
check("run_recognize 全链路不崩溃", True)

print("== 10. Tab2 未受影响 ==")
name0 = app.keyframe_templates[0]["template_name"]
tpl = app.seq_library[name0]
sm = app.smooth_leg_sequence(tpl.copy())
dist, *_ = app.dtw_with_path(sm, tpl)
check("Tab2自比对dist<0.01", dist < 0.01, f"dist={dist:.4f}")

print(f"\n结果: {ok} PASS / {fail} FAIL")
sys.exit(1 if fail else 0)