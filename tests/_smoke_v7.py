"""v7修复版冒烟测试：验证analyze所有分支返回3值且不崩溃"""
import sys
sys.path.insert(0, r"d:\豆包\老年健康")
import numpy as np
import cv2
from taiji_coach import analyze

passed = 0
failed = 0

def check(name, result, expect_len=3):
    global passed, failed
    if isinstance(result, tuple) and len(result) == expect_len:
        print(f"  PASS | {name}")
        passed += 1
    else:
        print(f"  FAIL | {name} -> 返回: {type(result)} len={len(result) if isinstance(result, tuple) else 'N/A'}")
        failed += 1

print("=== v7冒烟测试 ===")

# 1. 空图
print("[1] 空图兜底")
r = analyze(None)
check("None输入", r)
print(f"       报告: {r[1][:50]}")

# 2. 小图（<100px）
print("[2] 小图兜底")
small = np.zeros((50, 50, 3), dtype=np.uint8)
r = analyze(small)
check("50x50小图", r)
print(f"       报告: {r[1][:50]}")

# 3. 灰度图（ndim=2）
print("[3] 灰度图兜底")
gray = np.zeros((200, 200), dtype=np.uint8)
r = analyze(gray)
check("灰度图", r)
print(f"       报告: {r[1][:50]}")

# 4. RGBA图（4通道，验证不崩溃）
print("[4] RGBA图兜底")
rgba = np.zeros((200, 200, 4), dtype=np.uint8)
r = analyze(rgba)
check("RGBA图", r)
print(f"       报告: {r[1][:60]}")

# 5. 纯色无人体图
print("[5] 无人体图")
blank = np.full((300, 300, 3), 128, dtype=np.uint8)
r = analyze(blank)
check("纯色图", r)
print(f"       报告: {r[1][:60]}")

# 6. 真实人体图片
print("[6] 真实图片检测")
img = cv2.imdecode(np.fromfile(r"d:\豆包\老年健康\save_img\pose_20261028_004031.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
if img is not None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = analyze(img_rgb)
    check("真实图片", r)
    print(f"       报告: {r[1][:80]}")
    print(f"       音频: {'有' if r[2] else 'None（网络受限降级正常）'}")
else:
    print("  SKIP | 测试图片不存在")

# 7. 只读数组测试（模拟Gradio传入只读数组）
print("[7] 只读数组")
ro = np.zeros((300, 300, 3), dtype=np.uint8)
ro.flags.writeable = False
r = analyze(ro)
check("只读数组", r)

print(f"\n=== 结果: {passed} PASS / {failed} FAIL ===")
sys.exit(1 if failed else 0)