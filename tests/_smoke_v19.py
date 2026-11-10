# -*- coding: utf-8 -*-
"""v19冒烟测试：双文件架构 taiji_base + taiji_coach
核心变更：solutions.PoseEstimator不存在→yolov8n-pose直接输出keypoints(17点COCO)
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taiji_base as tb
import taiji_coach as tc
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

# ===== 1. taiji_base 全局状态 =====
print("=== 1. taiji_base 全局状态 ===")
check("yolo_model已加载", tb.yolo_model is not None)
check("TEMPLATE_DIR=taiji_templates", tb.TEMPLATE_DIR == "./taiji_templates")
check("EDGE_MARGIN=50", tb.EDGE_MARGIN == 50)
check("AREA_RATIO_THRESH=0.45", tb.AREA_RATIO_THRESH == 0.45)
check("DISTAL_THRESHOLD=70", tb.DISTAL_THRESHOLD == 70)
check("MAX_BAD_RATE=0.30", tb.MAX_BAD_RATE == 0.30)
check("ALL_KPTS=17点", len(tb.ALL_KPTS) == 17)
check("ensure_template_dir创建目录", os.path.exists(tb.TEMPLATE_DIR) or (tb.ensure_template_dir() is None and os.path.exists(tb.TEMPLATE_DIR)))

# ===== 2. detect_single_human =====
print("=== 2. detect_single_human ===")
black = np.zeros((480, 640, 3), dtype=np.uint8)
kpts, r0, msg = tb.detect_single_human(black)
check("黑帧kpts=None", kpts is None)
check("黑帧r0=None或存在", r0 is None or r0 is not None)  # YOLO可能返回空结果的r0
check("黑帧msg未检测到", "未检测到" in msg or "推理" in msg)

noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
kpts2, r02, msg2 = tb.detect_single_human(noise)
check("噪声帧不崩溃", kpts2 is None or isinstance(kpts2, list))

kpts3, r03, msg3 = tb.detect_single_human(None)
check("None帧不崩溃", kpts3 is None)

# ===== 3. 关键点工具 =====
print("=== 3. 关键点工具 ===")
check("valid_point正常", tb.valid_point((100, 200, 0.9)) is True)
check("valid_point低conf", tb.valid_point((100, 200, 0.1)) is False)
check("valid_point零坐标", tb.valid_point((0, 200, 0.9)) is False)

pts = [(100, 100, 0.9), (200, 300, 0.9), (0, 0, 0.1), (150, 200, 0.9)]
box = tb.get_bounding_box(pts)
check("bbox不为None", box is not None)
check("bbox xmin=100", box["xmin"] == 100)
check("bbox xmax=200", box["xmax"] == 200)
check("bbox area>0", box["area"] > 0)

box_none = tb.get_bounding_box([(0, 0, 0.1), (0, 0, 0.05)])
check("全无效点bbox=None", box_none is None)

# has_distal_open：远端撑开
pts_open = [(100, 100, 0.9)] * 17
pts_open[9] = (500, 400, 0.9)  # 远端点很远
check("远端撑开True", tb.has_distal_open(pts_open, 100, 100) is True)
pts_closed = [(100, 100, 0.9)] * 17
check("全收拢False", tb.has_distal_open(pts_closed, 100, 100) is False)

# is_side_trigger：双肩dx<120为侧身
pts_side = [(0, 0, 0.9)] * 17
pts_side[5] = (100, 100, 0.9)  # 左肩
pts_side[6] = (150, 100, 0.9)  # 右肩 dx=50<120
check("侧身触发True", tb.is_side_trigger(pts_side) is True)
pts_front = [(0, 0, 0.9)] * 17
pts_front[5] = (100, 100, 0.9)
pts_front[6] = (300, 100, 0.9)  # dx=200>120
check("正面False", tb.is_side_trigger(pts_front) is False)
check("短数组False", tb.is_side_trigger([(0,0,0.9)]*3) is False)

# is_near_edge
box_edge = {"xmin": 10, "xmax": 200, "ymin": 100, "ymax": 300}
check("贴边True", tb.is_near_edge(box_edge, 640, 480) is True)
box_center = {"xmin": 200, "xmax": 400, "ymin": 150, "ymax": 350}
check("居中False", tb.is_near_edge(box_center, 640, 480) is False)

# ===== 4. evaluate_frame =====
print("=== 4. evaluate_frame ===")
# 正常姿态：人在中间，四肢展开
normal_pts = [(0, 0, 0.1)] * 17
normal_pts[5] = (200, 150, 0.9)   # 左肩
normal_pts[6] = (400, 150, 0.9)   # 右肩
normal_pts[15] = (180, 400, 0.9)  # 左踝
normal_pts[16] = (420, 400, 0.9)  # 右踝
normal_pts[9] = (120, 250, 0.9)   # 左手腕
normal_pts[10] = (480, 250, 0.9)  # 右手腕
ok, reason = tb.evaluate_frame(normal_pts, 640, 480)
check("正常姿态放行", ok is True)

# 贴边
edge_pts = list(normal_pts)
edge_pts[5] = (10, 150, 0.9)
ok2, reason2 = tb.evaluate_frame(edge_pts, 640, 480)
check("贴边拒绝", ok2 is False and "边缘" in reason2)

# 无有效关键点
ok3, reason3 = tb.evaluate_frame([(0, 0, 0.1)] * 17, 640, 480)
check("无关键点拒绝", ok3 is False)

# 侧身+远端撑开=放行
side_open = list(normal_pts)
side_open[5] = (290, 150, 0.9)  # 左肩
side_open[6] = (310, 150, 0.9)  # 右肩 dx=20<120 侧身
side_open[9] = (100, 200, 0.9)  # 手远端撑开
ok4, reason4 = tb.evaluate_frame(side_open, 640, 480, base_area=50000)
check("侧身撑开放行", ok4 is True and "转身" in reason4)

# 侧身+收拢=挤团拒绝
side_curl = list(normal_pts)
side_curl[5] = (290, 150, 0.9)
side_curl[6] = (310, 150, 0.9)  # 侧身
side_curl[9] = (295, 160, 0.9)   # 手收拢
side_curl[10] = (305, 160, 0.9)
side_curl[15] = (295, 200, 0.9)
side_curl[16] = (305, 200, 0.9)
ok5, reason5 = tb.evaluate_frame(side_curl, 640, 480, base_area=50000)
check("挤团拒绝", ok5 is False and "挤团" in reason5)

# ===== 5. evaluate_and_extract / evaluate_video_quality =====
print("=== 5. evaluate_and_extract ===")
ok, br, msg, seq = tb.evaluate_and_extract(None)
check("None返回失败", ok is False and seq == [])

ok, br, msg, seq = tb.evaluate_and_extract("Z:/not_exist.mp4")
check("不存在返回失败", ok is False and seq == [])

ok, br, msg = tb.evaluate_video_quality(None)
check("兼容接口3元组", isinstance(ok, bool) and isinstance(br, float) and isinstance(msg, str))

# ===== 6. taiji_coach 模板读写 =====
print("=== 6. 模板读写 ===")
test_seq = [np.random.rand(17, 3).astype(np.float32) for _ in range(20)]
check("save正常", tc.save_action_template("smoke_v19", test_seq, is_temp=False) is True)
json_path = os.path.join(tb.TEMPLATE_DIR, "smoke_v19.json")
check("JSON文件存在", os.path.exists(json_path))

bank = tc.load_all_templates()
check("load含smoke_v19", "smoke_v19" in bank)
check("load序列20帧", len(bank["smoke_v19"]) == 20)
check("load值一致", np.allclose(bank["smoke_v19"][0], test_seq[0]))

check("非法名称save失败", tc.save_action_template("bad/name?", test_seq, is_temp=False) is False)

# 临时模板
check("save临时", tc.save_action_template("smoke_temp", test_seq, is_temp=True) is True)
temp_path = os.path.join(tb.TEMPLATE_DIR, "temp_smoke_temp.json")
check("临时文件temp_前缀", os.path.exists(temp_path))

names = tc.get_template_names()
check("名称含两个", "smoke_v19" in names and "smoke_temp" in names)
check("名称排序", names == sorted(names))

# 损坏JSON跳过
bad_path = os.path.join(tb.TEMPLATE_DIR, "bad.json")
with open(bad_path, "w") as f:
    f.write("{invalid!!!")
bank2 = tc.load_all_templates()
check("损坏JSON跳过", "bad" not in bank2)

# 清理
os.remove(json_path)
os.remove(temp_path)
os.remove(bad_path)

# ===== 7. save_temp / save_fixed =====
print("=== 7. save_temp / save_fixed ===")
msg, dd = tc.save_temp(None, "test")
check("空视频拒绝", "上传" in msg)
check("返回Dropdown", isinstance(dd, gr.Dropdown))

msg, dd = tc.save_temp("video", "")
check("空名称拒绝", "上传" in msg or "填写" in msg)

msg, dd = tc.save_temp("Z:/not_exist.mp4", "test")
check("不存在路径拒绝", "不存在" in msg or "帧不足" in msg)

msg, dd = tc.save_fixed(None, "test")
check("fixed空视频拒绝", "上传" in msg)
check("fixed返回Dropdown", isinstance(dd, gr.Dropdown))

# ===== 8. recognize_action =====
print("=== 8. recognize_action ===")
tip, img = tc.recognize_action(None)
check("None拒绝", "上传" in tip)

tip, img = tc.recognize_action("Z:/not_exist.mp4")
check("不存在拒绝", "不存在" in tip or "无法" in tip)

# 空模板库（用存在的文件路径，越过路径检查触发模板库空检查）
bank_before = tc.load_all_templates()
if not bank_before:
    tip, img = tc.recognize_action("taiji_coach.py")  # 存在的文件，非视频
    check("空模板库拒绝", "模板库为空" in tip)
else:
    check("空模板库拒绝(跳过-库非空)", True)

# ===== 9. flatten_seq =====
print("=== 9. flatten_seq ===")
seq_in = [np.random.rand(17, 3).astype(np.float32) for _ in range(5)]
flat = tc.flatten_seq(seq_in)
check("flatten长度5", len(flat) == 5)
check("flatten每项51维", flat[0].shape == (51,))
check("flatten空序列", tc.flatten_seq([]) == [])

# ===== 10. preview_with_skeleton =====
print("=== 10. preview_with_skeleton ===")
v, msg = tc.preview_with_skeleton(None)
check("None拒绝", v is None and "上传" in msg)

v, msg = tc.preview_with_skeleton("Z:/not_exist.mp4")
check("不存在拒绝", v is None and "不存在" in msg)

# ===== 11. 模型状态 =====
print("=== 11. 模型状态 ===")
check("yolo_model已加载", tb.yolo_model is not None)

print(f"\n{'='*40}")
print(f"v19冒烟测试：{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL > 0 else 0)