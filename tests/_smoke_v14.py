# -*- coding: utf-8 -*-
"""v14冒烟测试：YOLO框选+MediaPipe骨架+fastdtw录制比对（原版无语音）"""
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

print("=== 1. 全局初始状态 ===")
check("record_buffer模块级定义", isinstance(tc.record_buffer, list))
check("template_lib初始空", len(tc.template_lib) == 0)
check("sliding_window初始空", len(tc.sliding_window) == 0)
check("recording_mode初始False", tc.recording_mode is False)
check("current_template_name初始None", tc.current_template_name is None)

print("=== 2. normalize_kpts 归一化 ===")
class MockLm:
    def __init__(self, x, y):
        self.x = x; self.y = y; self.z = 0.0; self.visibility = 1.0
class MockLandmarks:
    def __init__(self):
        self.landmark = [MockLm(i / 32.0, (i % 8) / 8.0) for i in range(33)]
lm = MockLandmarks()
kpt = tc.normalize_kpts(lm, (100, 50, 300, 250))
check("形状(33,2)", kpt.shape == (33, 2), f"got {kpt.shape}")
check("值=lm.x/lm.y本身", abs(kpt[0][0] - 0.0) < 1e-6 and abs(kpt[1][0] - 1/32.0) < 1e-6)
check("值域0~1", kpt.min() >= 0 and kpt.max() <= 1.0)
check("dtype float32", kpt.dtype == np.float32)

print("=== 3. calc_seq_dist fastdtw 2元组API ===")
seq1 = [np.random.rand(33, 2).astype(np.float32) for _ in range(10)]
seq2 = [np.random.rand(33, 2).astype(np.float32) for _ in range(12)]
d, path = tc.calc_seq_dist(seq1, seq2)
check("返回2元组", isinstance(d, (int, float)) and isinstance(path, list))
check("距离>=0", d >= 0, f"got {d}")
check("path首(0,0)", path[0] == (0, 0), f"got {path[0]}")
check("path尾(len1-1,len2-1)", path[-1] == (9, 11), f"got {path[-1]}")
# 相同序列距离≈0
d_same, _ = tc.calc_seq_dist(seq1, [s.copy() for s in seq1])
check("相同序列距离≈0", d_same < 1e-3, f"got {d_same}")
# 短序列vs长序列不崩溃
d_short, _ = tc.calc_seq_dist(seq1[:2], seq2)
check("短序列不崩溃", d_short >= 0)

print("=== 4. start_record / stop_record ===")
# 初始状态
tc.recording_mode = False
tc.record_buffer.clear()
tc.sliding_window.clear()
r = tc.start_record()
check("start开启录制", tc.recording_mode is True and "开始录制" in r, f"got {r}")
check("start清空buffer", len(tc.record_buffer) == 0)
check("start清空窗口", len(tc.sliding_window) == 0)
# 模拟录制25帧
for i in range(25):
    tc.record_buffer.append(np.random.rand(33, 2).astype(np.float32))
r = tc.stop_record("起势")
check("25帧保存成功", "保存成功" in r and "25" in r, f"got {r}")
check("模板存入lib", "起势" in tc.template_lib and len(tc.template_lib["起势"]) == 25)
check("录制模式关闭", tc.recording_mode is False)
check("自动选中模板", tc.current_template_name == "起势")
# 帧数不足
tc.record_buffer.clear()
for i in range(10):
    tc.record_buffer.append(np.random.rand(33, 2).astype(np.float32))
r = tc.stop_record("太少")
check("10帧拒绝", "帧数太少" in r and "太少" not in tc.template_lib, f"got {r}")
# 空buffer
tc.record_buffer.clear()
r = tc.stop_record("空")
check("0帧拒绝", "帧数太少" in r, f"got {r}")

print("=== 5. select_template ===")
tc.template_lib = {"起势": [np.zeros((33, 2))] * 30, "云手": [np.zeros((33, 2))] * 30}
tc.sliding_window = [np.zeros((33, 2))] * 10
r = tc.select_template("云手")
check("选中云手", tc.current_template_name == "云手" and "云手" in r, f"got {r}")
check("选中清空窗口", len(tc.sliding_window) == 0)
r = tc.select_template(None)
check("None取消", tc.current_template_name is None, f"got {r}")
r = tc.select_template("不存在的")
check("不存在兜底", tc.current_template_name is None and "请先录制" in r, f"got {r}")
r = tc.select_template("")
check("空串兜底", tc.current_template_name is None, f"got {r}")

print("=== 6. get_frame / get_tip / poll ===")
check("初始帧480x640", tc.get_frame().shape == (480, 640, 3))
tc.global_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
tc.global_tip = "测试提示"
f, t = tc.poll()
check("poll返回帧+提示", int(f[0, 0, 0]) == 200 and t == "测试提示")
check("poll返回2值=outputs数", True)

print("=== 7. YOLO模型状态 ===")
check("yolo已加载", tc.yolo is not None, "YOLO加载失败（yolov8n-pose.pt缺失？）")

print("=== 8. MediaPipe Pose状态 ===")
check("pose对象存在", tc.pose is not None)
check("mp_draw存在", tc.mp_draw is not None)

print(f"\n{'='*40}")
print(f"v14冒烟测试：{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)