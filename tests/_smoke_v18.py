# -*- coding: utf-8 -*-
"""v18冒烟测试：双模具库+视频导入+extract_single_person公用函数+get_video_filepath"""
import sys, os, json, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taiji_coach as t
import gradio as gr

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

# ===== 1. 全局初始状态 =====
print("=== 1. 全局初始状态 ===")
check("run_flag初始False", t.run_flag is False)
check("recording_mode初始False", t.recording_mode is False)
check("fixed_mold_lib是dict", isinstance(t.fixed_mold_lib, dict))
check("temp_mold_lib是dict", isinstance(t.temp_mold_lib, dict))
check("sliding_window初始空", t.sliding_window == [])
check("current_mold_name初始None", t.current_mold_name is None)
check("MAX_MOLD_FRAMES=120", t.MAX_MOLD_FRAMES == 120)
check("WINDOW_LEN=25", t.WINDOW_LEN == 25)
check("TEMPLATE_DIR存在", os.path.exists(t.TEMPLATE_DIR))

# ===== 2. normalize_kpts 归一化 =====
print("=== 2. normalize_kpts 归一化 ===")
class FakeLM:
    def __init__(self, x, y):
        self.x = x; self.y = y; self.z = 0.0; self.visibility = 0.9
class FakeLandmarks:
    def __init__(self):
        self.landmark = [FakeLM(i / 32.0, (i % 7) / 7.0) for i in range(33)]
lm = FakeLandmarks()
kpt = t.normalize_kpts(lm, (10, 20, 100, 200))
check("形状(33,2)", kpt.shape == (33, 2))
check("值=lm.x/lm.y本身", np.allclose(kpt[5], [5/32.0, 5%7/7.0]))
check("值域0~1", kpt.min() >= 0.0 and kpt.max() <= 1.0)
check("dtype float32", kpt.dtype == np.float32)

# ===== 3. compare_with_mold fastdtw =====
print("=== 3. compare_with_mold fastdtw ===")
seqA = [np.random.rand(33, 2).astype(np.float32) for _ in range(20)]
seqB = [np.random.rand(33, 2).astype(np.float32) for _ in range(20)]
dist, score, path = t.compare_with_mold(seqA, seqB)
check("返回3元组", isinstance(dist, float) and isinstance(score, float) and isinstance(path, list))
check("score=100-dist*18", abs(score - max(0.0, 100.0 - dist * 18.0)) < 1e-6)
check("score>=0", score >= 0.0)
d2, s2, _ = t.compare_with_mold(seqA, seqA)
check("相同序列score≈100", s2 > 99.0)
d3, s3, p3 = t.compare_with_mold([], seqA)
check("空序列兜底", d3 == 0.0 and s3 == 0.0 and p3 == [])

# ===== 4. check_risk_pose 危险姿态 =====
print("=== 4. check_risk_pose 危险姿态 ===")
normal = np.zeros((33, 2), dtype=np.float32)
normal[0] = [0.5, 0.1]; normal[11] = [0.4, 0.3]; normal[12] = [0.6, 0.3]
normal[23] = [0.45, 0.6]; normal[24] = [0.55, 0.6]
normal[25] = [0.45, 0.8]; normal[26] = [0.55, 0.8]
r, msgs = t.check_risk_pose(normal)
check("正常无风险", r is False)

knee_risk = normal.copy()
knee_risk[25] = [0.45, 2.0]
r, msgs = t.check_risk_pose(knee_risk)
check("左膝风险触发", r is True and any("左膝" in m for m in msgs))

lean_risk = normal.copy()
lean_risk[0] = [0.5, 0.8]
r, msgs = t.check_risk_pose(lean_risk)
check("前倾风险触发", r is True and any("前倾" in m for m in msgs))

side_risk = normal.copy()
side_risk[11] = [0.1, 0.3]; side_risk[12] = [0.2, 0.3]
r, msgs = t.check_risk_pose(side_risk)
check("侧偏风险触发", r is True and any("侧向" in m for m in msgs))

r, msgs = t.check_risk_pose(None)
check("None不崩溃", r is False)

r, msgs = t.check_risk_pose(np.zeros((17, 2), dtype=np.float32))
check("短数组(<27)不崩溃", r is False)

r, msgs = t.check_risk_pose(np.zeros((27, 2), dtype=np.float32))
check("27长度可判断", r is not None)

# ===== 5. extract_single_person 公用函数 =====
print("=== 5. extract_single_person 公用函数 ===")
black = np.zeros((480, 640, 3), dtype=np.uint8)
img, kpt2 = t.extract_single_person(black)
check("黑帧返回img", img is not None and img.shape == (480, 640, 3))
check("黑帧kpt=None", kpt2 is None)

noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
img2, kpt3 = t.extract_single_person(noise)
check("噪声帧不崩溃", img2 is not None)

# ===== 6. get_video_filepath =====
print("=== 6. get_video_filepath ===")
check("None返回None", t.get_video_filepath(None) is None)
check("list取[0]", t.get_video_filepath(["/a/b.mp4"]) == "/a/b.mp4")
check("tuple取[0]", t.get_video_filepath(("/a/b.mp4",)) == "/a/b.mp4")
check("str原样返回", t.get_video_filepath("abc.mp4") == "abc.mp4")

# ===== 7. video_to_sequence 视频抽帧 =====
print("=== 7. video_to_sequence 视频抽帧 ===")
seq_v = t.video_to_sequence(None)
check("None返回空", seq_v == [])
seq_v2 = t.video_to_sequence("Z:/not_exist.mp4")
check("不存在返回空", seq_v2 == [])

# ===== 8. save_video_to_temp / save_video_to_fixed =====
print("=== 8. save_video_to_temp / save_video_to_fixed ===")
msg, dd = t.save_video_to_temp(None, "testV")
check("空路径拒绝", "没有选择" in msg)
check("返回Dropdown", isinstance(dd, gr.Dropdown))

msg2, dd2 = t.save_video_to_fixed(None, "testV")
check("fixed空路径拒绝", "没有选择" in msg2)
check("fixed返回Dropdown", isinstance(dd2, gr.Dropdown))

msg3, dd3 = t.save_video_to_temp("Z:/not_exist.mp4", "testV")
check("不存在路径拒绝", "没有选择" in msg3 or "出错" in msg3)

# ===== 9. JSON持久化 =====
print("=== 9. JSON持久化 ===")
test_seq = [np.random.rand(33, 2).astype(np.float32) for _ in range(25)]
ok = t.save_mold_to_fixed("smoke_test_mold", test_seq)
check("保存返回True", ok is True)
json_path = os.path.join(t.TEMPLATE_DIR, "smoke_test_mold.json")
check("JSON文件存在", os.path.exists(json_path))

loaded = t.load_fixed_molds()
check("load返回模具", "smoke_test_mold" in loaded)
check("load序列25帧", len(loaded["smoke_test_mold"]) == 25)
check("值一致", np.allclose(loaded["smoke_test_mold"][0], test_seq[0]))

# 损坏JSON跳过
bad_path = os.path.join(t.TEMPLATE_DIR, "bad_mold.json")
with open(bad_path, "w") as f:
    f.write("{not valid json!!!")
loaded2 = t.load_fixed_molds()
check("损坏JSON跳过", "bad_mold" not in loaded2)
os.remove(bad_path)
os.remove(json_path)

# save_mold_to_fixed 失败返回False（用非法名称）
ok2 = t.save_mold_to_fixed("bad/name?", test_seq)
check("非法名称返回False", ok2 is False)

# ===== 10. start_record / save_to_temp / save_to_fixed =====
print("=== 10. start_record / save_to_temp / save_to_fixed ===")
t.run_flag = True
msg = t.start_record()
check("start开启录制", "开始录制" in msg and t.recording_mode is True)

t.sliding_window = [np.random.rand(33, 2).astype(np.float32) for _ in range(25)]
msg, dd = t.save_to_temp("rec_mold")
check("temp保存成功", "存入临时库" in msg)
check("录制模式关闭", t.recording_mode is False)
check("buffer清空", t.sliding_window == [])
check("temp返回Dropdown", isinstance(dd, gr.Dropdown))
check("temp库含rec_mold", "rec_mold" in t.temp_mold_lib)

t.recording_mode = True
t.sliding_window = [np.random.rand(33, 2).astype(np.float32) for _ in range(10)]
msg, _ = t.save_to_temp("short_mold")
check("10帧拒绝", "帧数太少" in msg)

t.recording_mode = True
t.sliding_window = [np.random.rand(33, 2).astype(np.float32) for _ in range(25)]
msg, _ = t.save_to_fixed("fixed_mold")
check("fixed保存成功", "存入固定库" in msg)
check("fixed文件落盘", os.path.exists(os.path.join(t.TEMPLATE_DIR, "fixed_mold.json")))
check("fixed覆盖temp", "fixed_mold" not in t.temp_mold_lib)
os.remove(os.path.join(t.TEMPLATE_DIR, "fixed_mold.json"))
t.fixed_mold_lib.pop("fixed_mold", None)
t.temp_mold_lib.pop("rec_mold", None)
t.run_flag = False

# ===== 11. get_all_mold_names / select_mold =====
print("=== 11. get_all_mold_names / select_mold ===")
t.temp_mold_lib["A_mold"] = test_seq
t.fixed_mold_lib["B_mold"] = test_seq
names = t.get_all_mold_names()
check("合并两库名称", "A_mold" in names and "B_mold" in names)
check("排序", names == sorted(names))

msg = t.select_mold("A_mold")
check("选中A", "选中" in msg and t.current_mold_name == "A_mold")
check("选中清空窗口", t.sliding_window == [])

msg = t.select_mold(None)
check("None取消", "取消" in msg and t.current_mold_name is None)

msg = t.select_mold("not_exist")
check("不存在兜底", "不存在" in msg)

msg = t.select_mold("")
check("空串兜底", "取消" in msg or "不存在" in msg)
t.temp_mold_lib.pop("A_mold", None)
t.fixed_mold_lib.pop("B_mold", None)

# ===== 12. poll =====
print("=== 12. poll ===")
result = t.poll()
check("poll返回2值=outputs数", len(result) == 2)
check("poll帧480x640", result[0].shape == (480, 640, 3))
check("poll提示是str", isinstance(result[1], str))

# ===== 13. 模型状态 =====
print("=== 13. 模型状态 ===")
check("yolo已加载", t.yolo is not None)
check("pose对象存在", t.pose is not None)
check("mp_draw存在", t.mp_draw is not None)

print(f"\n{'='*40}")
print(f"v18冒烟测试：{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL > 0 else 0)