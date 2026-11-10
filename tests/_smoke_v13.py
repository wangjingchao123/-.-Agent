# -*- coding: utf-8 -*-
"""v13冒烟测试：24式手工标准库实时纠错"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import taiji_coach as tc

PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {name}")
    else:
        FAIL += 1; print(f"  [FAIL] {name} {detail}")

print("=== 1. calc_angle 角度计算 ===")
# 直角：p1=(1,0) p2=(0,0) p3=(0,1)
ang = tc.calc_angle((1, 0), (0, 0), (0, 1))
check("直角=90°", abs(ang - 90) < 0.1, f"got {ang}")
# 平角：p1=(1,0) p2=(0,0) p3=(-1,0)
ang = tc.calc_angle((1, 0), (0, 0), (-1, 0))
check("平角=180°", abs(ang - 180) < 0.1, f"got {ang}")
# 零向量兜底
ang = tc.calc_angle((0, 0), (0, 0), (1, 1))
check("零向量返回0", ang == 0, f"got {ang}")
# 数组输入
ang = tc.calc_angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1]))
check("数组输入OK", abs(ang - 90) < 0.1, f"got {ang}")

print("=== 2. standard_lib_24 完整性 ===")
check("共24式", len(tc.standard_lib_24) == 24, f"got {len(tc.standard_lib_24)}")
all_ok = True
for name, std in tc.standard_lib_24.items():
    if not all(k in std for k in ("spine_angle_max", "knee_angle_min", "shoulder_diff_max", "elbow_angle_range")):
        all_ok = False; print(f"    缺参数: {name}")
    if not isinstance(std["elbow_angle_range"], list) or len(std["elbow_angle_range"]) != 2:
        all_ok = False; print(f"    elbow_range格式错: {name}")
    if std["elbow_angle_range"][0] >= std["elbow_angle_range"][1]:
        all_ok = False; print(f"    elbow_range区间反: {name}")
check("每项4参数+区间格式", all_ok)
check("action_list=24", len(tc.action_list) == 24)

print("=== 3. get_body_metrics 身体指标 ===")
# 正常站姿（像素坐标，YOLO坐标系y向下）
# 鼻(320,80) 肩L(290,150)R(350,150) 肘L(280,200)R(360,200) 腕L(275,250)R(365,250)
# 髋L(300,280)R(340,280) 膝L(300,380)R(340,380) 踝L(300,460)R(340,460)
kpts = np.zeros((17, 2), dtype=np.float32)
for i, (x, y) in {0:(320,80), 5:(290,150), 6:(350,150), 7:(280,200), 8:(360,200),
                   9:(275,250), 10:(365,250), 11:(300,280), 12:(340,280),
                   13:(300,380), 14:(340,380), 15:(300,460), 16:(340,460)}.items():
    kpts[i] = (x, y)
met = tc.get_body_metrics(kpts)
check("脊柱角≈0（站直）", met["spine_angle"] < 5, f"got {met['spine_angle']}")
# 膝角：髋(300,280)膝(300,380)踝(300,460) → 180°
check("膝角≈180（站直）", met["knee_min_angle"] > 170, f"got {met['knee_min_angle']}")
check("肩差=0（平肩）", met["shoulder_diff"] < 1, f"got {met['shoulder_diff']}")
# 肘角：肩(290,150)肘(280,200)腕(275,250) → 接近180°
check("肘角>150（手臂下垂）", met["elbow_avg_angle"] > 150, f"got {met['elbow_avg_angle']}")
# 弯腰：肩中心前倾（x偏移，模拟上身前倾）
kpts_bend = kpts.copy()
kpts_bend[5] = (250, 200); kpts_bend[6] = (290, 200)  # 肩前倾向左
met_bend = tc.get_body_metrics(kpts_bend)
check("弯腰脊柱角增大", met_bend["spine_angle"] > met["spine_angle"], f"{met['spine_angle']}->{met_bend['spine_angle']}")

print("=== 4. check_standard 标准检查 ===")
std = tc.standard_lib_24["01 起势"]  # spine<10 knee>160 shoulder<12 elbow 70~110
# 全达标
met_ok = {"spine_angle": 2, "knee_min_angle": 175, "shoulder_diff": 3, "elbow_avg_angle": 90}
r = tc.check_standard(met_ok, std)
check("全达标", "符合标准" in r, f"got {r}")
# 脊柱超限
met_bad = {"spine_angle": 15, "knee_min_angle": 175, "shoulder_diff": 3, "elbow_avg_angle": 90}
r = tc.check_standard(met_bad, std)
check("脊柱超限警告", "躯干倾斜" in r, f"got {r}")
# 膝角不足
met_bad = {"spine_angle": 2, "knee_min_angle": 140, "shoulder_diff": 3, "elbow_avg_angle": 90}
r = tc.check_standard(met_bad, std)
check("膝角不足警告", "膝盖弯曲" in r, f"got {r}")
# 肩差超限
met_bad = {"spine_angle": 2, "knee_min_angle": 175, "shoulder_diff": 20, "elbow_avg_angle": 90}
r = tc.check_standard(met_bad, std)
check("肩差超限警告", "双肩" in r, f"got {r}")
# 肘角超限
met_bad = {"spine_angle": 2, "knee_min_angle": 175, "shoulder_diff": 3, "elbow_avg_angle": 160}
r = tc.check_standard(met_bad, std)
check("肘角超限警告", "手肘" in r, f"got {r}")
# 多项同时超限
met_bad = {"spine_angle": 20, "knee_min_angle": 100, "shoulder_diff": 30, "elbow_avg_angle": 160}
r = tc.check_standard(met_bad, std)
check("多项同时警告", r.count("⚠️") == 4, f"got {r.count('⚠️')} warns")

print("=== 5. switch_action 动作切换 ===")
tc.current_action_manual = None
r = tc.switch_action("01 起势")
check("选择动作", "已加载" in r and tc.current_action_manual == "01 起势", f"got {r}")
r = tc.switch_action("请选择24式动作")
check("取消选择", tc.current_action_manual is None, f"got {r}")
r = tc.switch_action(None)
check("None取消", tc.current_action_manual is None, f"got {r}")
r = tc.switch_action("24 收势")
check("选收势", tc.current_action_manual == "24 收势", f"got {r}")

print("=== 6. _en_action 拼音映射 ===")
check("起势→01 QiShi", tc._en_action("01 起势") == "01 QiShi", f"got {tc._en_action('01 起势')}")
check("云手→10 YunShou", tc._en_action("10 云手") == "10 YunShou", f"got {tc._en_action('10 云手')}")
check("收势→24 ShouShi", tc._en_action("24 收势") == "24 ShouShi", f"got {tc._en_action('24 收势')}")
check("请选择→Select", tc._en_action("请选择24式动作") == "Select action")
check("None→Select", tc._en_action(None) == "Select action")

print("=== 7. get_frame / get_tip ===")
img = tc.get_frame()
check("初始帧480x640", img is not None and img.shape == (480, 640, 3))
tc.global_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
check("有帧返回帧", int(tc.get_frame()[0, 0, 0]) == 100)
tc.global_tip = "测试"
check("get_tip", tc.get_tip() == "测试")

print("=== 8. extract_kpts conf=None兜底 ===")
class MockKpts:
    def __init__(self, data, conf):
        self.data = data; self.conf = conf
class MockResult:
    def __init__(self, data, conf):
        self.keypoints = MockKpts(data, conf)
import torch
# 正常关键点
kpt_data = torch.rand(1, 17, 3)
kpt_conf = torch.ones(1, 17)
res = MockResult(kpt_data, kpt_conf)
k = tc.extract_kpts_from_frame(res)
check("正常提取返回(17,2)", k is not None and k.shape == (17, 2), f"got {k.shape if k is not None else None}")
# conf=None兜底
res_none = MockResult(kpt_data, None)
k = tc.extract_kpts_from_frame(res_none)
check("conf=None不崩溃", k is not None and k.shape == (17, 2))
# data为空tensor（len=0，模拟无人）
res_empty = MockResult(torch.empty(0, 17, 3), None)
k = tc.extract_kpts_from_frame(res_empty)
check("data空返回None", k is None)

print(f"\n{'='*40}")
print(f"v13冒烟测试：{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)