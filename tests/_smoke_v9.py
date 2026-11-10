"""v9冒烟测试：验证calc_angle修复+各函数兜底"""
import sys
sys.path.insert(0, r"d:\豆包\老年健康")
import numpy as np
from taiji_coach import calc_angle, lm2array, analyze_single_image, load_template_video, compare_user_video

passed = 0; failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        print(f"  PASS | {name}"); passed += 1
    else:
        print(f"  FAIL | {name}"); failed += 1

print("=== v9冒烟测试 ===")

# 1. calc_angle接收数组（原版接收landmark对象会AttributeError）
print("[1] calc_angle数组修复")
try:
    a = calc_angle([0,0], [1,0], [1,1])  # 90度直角
    check(f"直角={a:.1f}", abs(a - 90) < 1)
    b = calc_angle([0,0], [1,0], [2,0])  # 180度平角
    check(f"平角={b:.1f}", abs(b - 180) < 1)
except AttributeError:
    check("calc_angle数组调用", False)

# 2. analyze_single_image None兜底
print("[2] analyze_single_image None兜底")
r = analyze_single_image(None)
check("None返回2值", isinstance(r, tuple) and len(r) == 2)

# 3. analyze_single_image 真实图片
print("[3] analyze_single_image 真实图片")
import cv2
img = cv2.imdecode(np.fromfile(r"d:\豆包\老年健康\save_img\upload_pose_20261028_184734.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r = analyze_single_image(rgb)
check("真实图片返回2值", isinstance(r, tuple) and len(r) == 2)
print(f"       报告: {r[1][:60]}")

# 4. load_template_video None兜底
print("[4] load_template_video None兜底")
r = load_template_video(None)
check("None返回字符串", isinstance(r, str))

# 5. compare_user_video None兜底
print("[5] compare_user_video None兜底")
r = compare_user_video(None)
check("None返回2值", isinstance(r, tuple) and len(r) == 2)
print(f"       报告: {r[1]}")

# 6. compare_user_video 无模板兜底
print("[6] compare_user_video 无模板兜底")
import taiji_coach
taiji_coach.template_keypoints_seq = []
r = compare_user_video("fake.mp4")
check("无模板返回2值", isinstance(r, tuple) and len(r) == 2)
print(f"       报告: {r[1]}")

# 7. lm2array格式验证
print("[7] lm2array格式")
class FakeLM:
    def __init__(self, x, y): self.x = x; self.y = y
fake = [FakeLM(i*0.1, i*0.2) for i in range(33)]
arr = lm2array(fake)
check(f"shape={arr.shape}", arr.shape == (33, 2))
# 用lm2array结果调calc_angle不崩
ang = calc_angle(arr[11], arr[13], arr[15])
check(f"lm2array→calc_angle不崩={ang:.1f}", True)

print(f"\n=== 结果: {passed} PASS / {failed} FAIL ===")
sys.exit(1 if failed else 0)