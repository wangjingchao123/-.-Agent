# -*- coding: utf-8 -*-
"""v12冒烟测试：DTW动态模板比对+实时纠错"""
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

print("=== 1. normalize_kpts 骨骼归一化 ===")
# 正常站姿：髋(300,240)(340,240) 肩(300,150)(340,150) 鼻(320,80)
kpts = np.zeros((17, 2), dtype=np.float32)
for i, (x, y) in {0:(320,80), 5:(300,150), 6:(340,150), 11:(300,240), 12:(340,240)}.items():
    kpts[i] = (x, y)
nk = tc.normalize_kpts(kpts)
check("归一化后髋中点为原点", abs(nk[11][0]+nk[12][0]) < 1e-5 and abs(nk[11][1]+nk[12][1]) < 1e-5,
      f"lhip={nk[11]} rhip={nk[12]}")
# 髋肩距离归一化为1：肩中心(320,150) 髋中心(320,240) 距离90
check("肩髋距离归一化为1", abs(np.linalg.norm((nk[5]+nk[6])/2 - (nk[11]+nk[12])/2) - 1.0) < 1e-4,
      f"got {np.linalg.norm((nk[5]+nk[6])/2 - (nk[11]+nk[12])/2)}")
# 平移不变性：整体平移后归一化结果相同
kpts2 = kpts + np.array([100, 50])
nk2 = tc.normalize_kpts(kpts2)
check("平移不变", np.allclose(nk, nk2, atol=1e-4))
# 缩放不变性：整体放大2倍后归一化结果相同
kpts3 = kpts * 2.0
nk3 = tc.normalize_kpts(kpts3)
check("缩放不变", np.allclose(nk, nk3, atol=1e-4))
# scale=0兜底（肩髋重合）
kpts0 = np.zeros((17, 2), dtype=np.float32)
nk0 = tc.normalize_kpts(kpts0)
check("scale=0不崩溃", nk0 is not None and nk0.shape == (17, 2))

print("=== 2. calc_seq_dist DTW距离（1.5.3返回DTW对象） ===")
seqA = [np.random.rand(17, 2).tolist() for _ in range(6)]
seqB = [np.random.rand(17, 2).tolist() for _ in range(10)]
d, align = tc.calc_seq_dist(seqA, seqB)
check("返回(distance, DTW对象)", isinstance(d, float) and hasattr(align, "index2"), f"d={d} type={type(align)}")
check("distance>0", d > 0, f"d={d}")
check("index2有效", len(align.index2) > 0, f"len={len(align.index2)}")
# 相同序列距离接近0
same = [np.random.rand(17, 2).tolist() for _ in range(6)]
d_same, _ = tc.calc_seq_dist(same, [list(x) for x in same])
check("相同序列距离≈0", d_same < 0.01, f"d={d_same}")
# 不同序列距离大于相同序列
d_diff, _ = tc.calc_seq_dist(same, [np.random.rand(17, 2).tolist() for _ in range(6)])
check("不同序列距离更大", d_diff > d_same, f"diff={d_diff} same={d_same}")

print("=== 3. record_template 参数校验 ===")
tc.yolo_model = None  # 模拟模型不可用
r = tc.record_template("fake.mp4", "起势")
check("模型不可用降级", "不可用" in r, f"got {r}")
tc.yolo_model = object()  # 恢复非None但不实际推理
r = tc.record_template(None, "起势")
check("无视频降级", "请上传" in r, f"got {r}")
r = tc.record_template("fake.mp4", "  ")
check("空动作名降级", "请上传" in r or "填写" in r, f"got {r}")
r = tc.record_template("不存在的文件.mp4", "起势")
check("打不开视频降级", "无法打开" in r, f"got {r}")
tc.yolo_model = None  # 恢复

print("=== 4. set_manual_action 手动锁定 ===")
tc.template_lib = {"起势": [[0.0]*34]*10, "云手": [[0.0]*34]*10}
r = tc.set_manual_action("自动识别")
check("切回自动", "自动" in r and tc.current_action_manual is None, f"got {r}")
r = tc.set_manual_action("起势")
check("锁定动作", "锁定" in r and tc.current_action_manual == "起势", f"got {r}")
r = tc.set_manual_action(None)
check("None回自动", tc.current_action_manual is None, f"got {r}")

print("=== 5. voice_cmd_bridge 语音指令桥接 ===")
tc.template_lib = {"起势": [[0.0]*34]*10, "左右野马分鬃": [[0.0]*34]*10}
tc.current_action_manual = None
r = tc.voice_cmd_bridge("起势")
check("语音匹配起势", "锁定" in r and tc.current_action_manual == "起势", f"got {r}")
r = tc.voice_cmd_bridge("野马分鬃")
check("语音映射野马分鬃", "锁定" in r and tc.current_action_manual == "左右野马分鬃", f"got {r}")
r = tc.voice_cmd_bridge("不存在的动作")
check("不匹配提示", "不匹配" in r, f"got {r}")
r = tc.voice_cmd_bridge(None)
check("None不崩溃", isinstance(r, str), f"got {r}")
# 库中无该动作
tc.template_lib = {"云手": [[0.0]*34]*10}
r = tc.voice_cmd_bridge("起势")
check("库中无该动作", "不匹配" in r, f"got {r}")

print("=== 6. get_frame / get_tip 降级 ===")
tc.global_frame = None
img = tc.get_frame()
check("无帧返回占位黑图", img is not None and img.shape == (480, 640, 3))
tc.global_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
img = tc.get_frame()
check("有帧返回帧", int(img[0, 0, 0]) == 200)
tc.global_tip = "测试提示"
check("get_tip返回文本", tc.get_tip() == "测试提示")

print("=== 7. 模板库线程安全 ===")
import threading
errors = []
def writer():
    try:
        for i in range(50):
            with tc._tpl_lock:
                tc.template_lib[f"动作{i}"] = [[0.0]*34]*10
    except Exception as e:
        errors.append(str(e))
def reader():
    try:
        for i in range(50):
            with tc._tpl_lock:
                _ = dict(tc.template_lib)
    except Exception as e:
        errors.append(str(e))
t1 = threading.Thread(target=writer); t2 = threading.Thread(target=reader)
t1.start(); t2.start(); t1.join(); t2.join()
check("并发读写无异常", len(errors) == 0, f"errors={errors[:2]}")

print("=== 8. _EN_MAP putText中文映射 ===")
check("起势有映射", tc._EN_MAP.get("起势") == "QiShi")
check("未识别有映射", tc._EN_MAP.get("未识别") == "Unknown")
check("未知动作兜底", tc._EN_MAP.get("新动作", "Action") == "Action")

print("=== 9. ERR_THRESH 合理性 ===")
check("ERR_THRESH在归一化范围", 0.1 < tc.ERR_THRESH < 3.0, f"got {tc.ERR_THRESH}")
check("SIM_THRESHOLD>0", tc.SIM_THRESHOLD > 0)

print(f"\n{'='*40}")
print(f"v12冒烟测试：{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)