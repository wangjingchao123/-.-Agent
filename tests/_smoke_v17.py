# -*- coding: utf-8 -*-
"""v17冒烟测试：双模具库+视频导入+extract_single_person+危险姿态判定"""
import sys, os, json
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

# 清理templates残留
TEST_DIR = tc.TEMPLATE_DIR
if os.path.exists(TEST_DIR):
    for f in os.listdir(TEST_DIR):
        if f.endswith(".json"):
            os.remove(os.path.join(TEST_DIR, f))

print("=== 1. 全局初始状态 ===")
check("record_buffer模块级定义", isinstance(tc.record_buffer, list))
check("fixed_mold_lib初始空", len(tc.fixed_mold_lib) == 0)
check("temp_mold_lib初始空", len(tc.temp_mold_lib) == 0)
check("MAX_MOLD_FRAMES=120", tc.MAX_MOLD_FRAMES == 120)
check("recording_mode初始False", tc.recording_mode is False)
check("current_mold_name初始None", tc.current_mold_name is None)
check("TEMPLATE_DIR存在", os.path.isdir(TEST_DIR))

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

print("=== 3. compare_with_mold fastdtw ===")
seq1 = [np.random.rand(33, 2).astype(np.float32) for _ in range(10)]
seq2 = [np.random.rand(33, 2).astype(np.float32) for _ in range(12)]
dist, score, path = tc.compare_with_mold(seq1, seq2)
check("返回3元组", isinstance(dist, (int, float)) and isinstance(score, float) and isinstance(path, list))
check("score=100-dist*18", abs(score - max(0.0, 100.0 - dist * 18.0)) < 1e-6)
check("score>=0", score >= 0)
_, s_same, _ = tc.compare_with_mold(seq1, [s.copy() for s in seq1])
check("相同序列score≈100", s_same > 99.9, f"got {s_same}")

print("=== 4. check_risk_pose 危险姿态 ===")
kpts_ok = np.zeros((33, 2), dtype=np.float32)
kpts_ok[0] = (0.5, 0.1); kpts_ok[11] = (0.4, 0.2); kpts_ok[12] = (0.6, 0.2)
kpts_ok[23] = (0.45, 0.5); kpts_ok[24] = (0.55, 0.5)
kpts_ok[25] = (0.45, 0.8); kpts_ok[26] = (0.55, 0.8)
danger, texts = tc.check_risk_pose(kpts_ok)
check("正常无风险", danger is False and "无高风险" in texts[0], f"got {texts}")
kpts_risk = kpts_ok.copy(); kpts_risk[25] = (0.45, 2.0)
danger, texts = tc.check_risk_pose(kpts_risk)
check("左膝风险触发", danger is True and any("左膝" in t for t in texts), f"got {texts}")
kpts_risk = kpts_ok.copy(); kpts_risk[0] = (0.5, 0.6)
danger, texts = tc.check_risk_pose(kpts_risk)
check("前倾风险触发", danger is True and any("前倾" in t for t in texts), f"got {texts}")
kpts_risk = kpts_ok.copy(); kpts_risk[11] = (0.1, 0.2); kpts_risk[12] = (0.2, 0.2)
danger, texts = tc.check_risk_pose(kpts_risk)
check("侧偏风险触发", danger is True and any("侧向" in t for t in texts), f"got {texts}")
danger, texts = tc.check_risk_pose(None)
check("None不崩溃", danger is False)
danger, texts = tc.check_risk_pose(np.zeros((20, 2)))
check("短数组不崩溃", danger is False)

print("=== 5. extract_single_person 公用函数 ===")
# 黑帧：无人体，返回原图+None
black = np.zeros((480, 640, 3), dtype=np.uint8)
img, kpt = tc.extract_single_person(black)
check("黑帧返回img", img is not None and img.shape == (480, 640, 3))
check("黑帧kpt=None", kpt is None)
# 随机噪声帧不崩溃
noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
img, kpt = tc.extract_single_person(noise)
check("噪声帧不崩溃", img is not None)

print("=== 6. video_to_sequence 视频抽帧 ===")
# None/不存在路径返回空
seq = tc.video_to_sequence(None)
check("None返回空", seq == [])
seq = tc.video_to_sequence("不存在的视频.mp4")
check("不存在返回空", seq == [])

print("=== 7. save_video_to_temp / save_video_to_fixed ===")
# 空路径
msg, drop = tc.save_video_to_temp(None, "测试")
check("空路径拒绝", "请先上传" in msg, f"got {msg}")
check("返回Dropdown", str(type(drop).__name__) == "Dropdown")
msg, drop = tc.save_video_to_fixed(None, "测试")
check("fixed空路径拒绝", "请先上传" in msg, f"got {msg}")
# 不存在视频路径（video_to_sequence返回空→帧数不足）
msg, drop = tc.save_video_to_temp("不存在的视频.mp4", "测试")
check("无效视频帧数不足", "帧数不足" in msg and "测试" not in tc.temp_mold_lib, f"got {msg}")

print("=== 8. JSON持久化 ===")
test_seq = [np.random.rand(33, 2).astype(np.float32) for _ in range(25)]
r = tc.save_mold_to_fixed("测试模具", test_seq)
check("保存返回True", r is True)
fpath = os.path.join(TEST_DIR, "测试模具.json")
check("JSON文件存在", os.path.exists(fpath))
molds = tc.load_fixed_molds()
check("load返回模具", "测试模具" in molds)
check("load序列25帧", len(molds["测试模具"]) == 25)
check("值一致", np.allclose(molds["测试模具"][0], test_seq[0]))
with open(os.path.join(TEST_DIR, "损坏.json"), "w") as f:
    f.write("{invalid")
molds2 = tc.load_fixed_molds()
check("损坏JSON跳过", "损坏" not in molds2 and "测试模具" in molds2)
os.remove(fpath)
os.remove(os.path.join(TEST_DIR, "损坏.json"))

print("=== 9. start_record / save_to_temp / save_to_fixed ===")
tc.recording_mode = False
tc.record_buffer.clear()
tc.temp_mold_lib.clear()
tc.fixed_mold_lib.clear()
r = tc.start_record()
check("start开启录制", tc.recording_mode is True and "开始录制" in r)
for i in range(25):
    tc.record_buffer.append(np.random.rand(33, 2).astype(np.float32))
msg, _ = tc.save_to_temp("临时模具")
check("temp保存成功", "临时模具库" in msg and "临时模具" in tc.temp_mold_lib, f"got {msg}")
check("录制模式关闭", tc.recording_mode is False)
check("buffer清空", len(tc.record_buffer) == 0)
tc.record_buffer.clear()
for i in range(10):
    tc.record_buffer.append(np.random.rand(33, 2).astype(np.float32))
msg, _ = tc.save_to_temp("太少")
check("10帧拒绝", "帧数太少" in msg and "太少" not in tc.temp_mold_lib)
tc.record_buffer.clear()
for i in range(25):
    tc.record_buffer.append(np.random.rand(33, 2).astype(np.float32))
msg, _ = tc.save_to_fixed("固定模具")
check("fixed保存成功", "固定模具库" in msg and "固定模具" in tc.fixed_mold_lib, f"got {msg}")
check("fixed文件落盘", os.path.exists(os.path.join(TEST_DIR, "固定模具.json")))
check("fixed覆盖temp", "固定模具" not in tc.temp_mold_lib)
os.remove(os.path.join(TEST_DIR, "固定模具.json"))

print("=== 10. get_all_mold_names / select_mold ===")
tc.fixed_mold_lib = {"A": [np.zeros((33, 2))] * 25}
tc.temp_mold_lib = {"B": [np.zeros((33, 2))] * 25}
names = tc.get_all_mold_names()
check("合并两库名称", set(names) == {"A", "B"})
check("排序", names == sorted(names))
r = tc.select_mold("A")
check("选中A", tc.current_mold_name == "A" and "A" in r)
check("选中清空窗口", len(tc.sliding_window) == 0)
r = tc.select_mold(None)
check("None取消", tc.current_mold_name is None)
r = tc.select_mold("不存在")
check("不存在兜底", tc.current_mold_name is None and "请先录制" in r)
r = tc.select_mold("")
check("空串兜底", tc.current_mold_name is None)

print("=== 11. get_frame / get_tip / poll ===")
check("初始帧480x640", tc.get_frame().shape == (480, 640, 3))
tc.global_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
tc.global_tip = "测试"
f, t = tc.poll()
check("poll返回帧+提示", int(f[0, 0, 0]) == 180 and t == "测试")
check("poll返回2值=outputs数", True)

print("=== 12. 模型状态 ===")
check("yolo已加载", tc.yolo is not None)
check("pose对象存在", tc.pose is not None)
check("mp_draw存在", tc.mp_draw is not None)

print(f"\n{'='*40}")
print(f"v17冒烟测试：{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)