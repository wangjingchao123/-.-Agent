"""v10冒烟测试：相似度匹配+防抖计数+模板加载+各函数兜底"""
import sys
sys.path.insert(0, r"d:\豆包\老年健康")
import numpy as np
import cv2
import taiji_coach as tc

passed = 0; failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        print(f"  PASS | {name}"); passed += 1
    else:
        print(f"  FAIL | {name}"); failed += 1

print("=== v10冒烟测试 ===")

# 1. calc_angle防除零
print("[1] calc_angle防除零")
a = tc.calc_angle([0,0], [0,0], [1,1])  # 重合点norm=0，不崩
check(f"重合点不崩={a:.1f}", True)
b = tc.calc_angle([0,0], [1,0], [1,1])
check(f"直角={b:.1f}", abs(b - 90) < 1)

# 2. calc_similarity
print("[2] calc_similarity")
lm1 = np.array([[0.5, 0.5]] * 33)
lm2 = np.array([[0.5, 0.5]] * 33)
s_same = tc.calc_similarity(lm1, lm2)
check(f"相同姿态={s_same}", s_same == 100.0)
lm3 = np.array([[0.8, 0.8]] * 33)  # 偏移0.42
s_diff = tc.calc_similarity(lm1, lm3)
check(f"偏移姿态={s_diff}（应<100）", s_diff < 100)

# 3. find_best_pose（无分段）
print("[3] find_best_pose无分段")
tc.pose_segments = []
tc.template_keypoints_seq = []
name, score, std = tc.find_best_pose(lm1, 50)
check(f"无分段返回未识别", name == "未识别动作" and std is None)

# 4. find_best_pose（有分段+匹配）
print("[4] find_best_pose有分段")
tc.template_keypoints_seq = [lm1] * 10  # 10帧标准
tc.pose_segments = [{"name": "起势", "start": 0, "end": 9}]
name, score, std = tc.find_best_pose(lm1, 50)
check(f"匹配起势 score={score}", name == "起势" and score == 100.0 and std is not None)
# 阈值过高不匹配
name2, score2, std2 = tc.find_best_pose(lm3, 90)
check(f"阈值90不匹配", name2 == "未识别动作" and std2 is None)

# 5. _check_errors防抖：连续4帧才告警
print("[5] _check_errors防抖")
# 标准姿态：膝盖伸直（髋-膝-踝竖直，角度180°）
std = np.array([[0.5, 0.5]] * 33)
std[23] = [0.4, 0.4]; std[25] = [0.4, 0.6]; std[27] = [0.4, 0.8]  # 左腿竖直
# 错误姿态：膝盖弯曲（角度~127°，差值53°>25°超标）
user_bad = np.array([[0.5, 0.5]] * 33)
user_bad[23] = [0.4, 0.4]; user_bad[25] = [0.5, 0.6]; user_bad[27] = [0.4, 0.8]
std_ang = tc.calc_angle(std[23], std[25], std[27])
bad_ang = tc.calc_angle(user_bad[23], user_bad[25], user_bad[27])
print(f"       标准膝角={std_ang:.1f}° 错误膝角={bad_ang:.1f}° 差={abs(std_ang-bad_ang):.1f}°")
counter = {}
errs1 = tc._check_errors(user_bad, std, 22, 25, counter)
check(f"第1帧不告警={errs1}", len(errs1) == 0)
tc._check_errors(user_bad, std, 22, 25, counter)
tc._check_errors(user_bad, std, 22, 25, counter)
errs4 = tc._check_errors(user_bad, std, 22, 25, counter)
check(f"第4帧告警={errs4}（含25左膝）", 25 in errs4)
# 恢复正常→计数清零
errs_ok = tc._check_errors(std, std, 22, 25, counter)
check(f"恢复正常清零={errs_ok}", len(errs_ok) == 0 and counter.get(25, 0) == 0)

# 6. analyze_single_image None兜底
print("[6] analyze_single_image None兜底")
r = tc.analyze_single_image(None, 22, 25)
check("None返回2值", isinstance(r, tuple) and len(r) == 2)

# 7. analyze_single_image 真实图片
print("[7] analyze_single_image 真实图片")
img = cv2.imdecode(np.fromfile(r"d:\豆包\老年健康\save_img\upload_pose_20261028_184734.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r = tc.analyze_single_image(rgb, 22, 25)
check("真实图片返回2值", isinstance(r, tuple) and len(r) == 2)
print(f"       报告: {r[1][:50]}")

# 8. load_template_video None兜底
print("[8] load_template_video None兜底")
r = tc.load_template_video(None)
check("None返回字符串", isinstance(r, str))

# 9. compare_user_video None/无模板兜底
print("[9] compare_user_video兜底")
r = tc.compare_user_video(None, 50, 22, 25)
check("None返回2值", isinstance(r, tuple) and len(r) == 2)
tc.template_keypoints_seq = []; tc.pose_segments = []
r = tc.compare_user_video("fake.mp4", 50, 22, 25)
check("无模板返回2值", isinstance(r, tuple) and len(r) == 2)

# 10. load_saved_template None兜底
print("[10] load_saved_template None兜底")
r = tc.load_saved_template(None)
check("None返回字符串", isinstance(r, str))

# 11. add_pose_segment 非数字兜底
print("[11] add_pose_segment非数字兜底")
r = tc.add_pose_segment("测试", "abc", "xyz")
check("非数字返回错误提示", isinstance(r, str) and "数字" in r)

print(f"\n=== 结果: {passed} PASS / {failed} FAIL ===")
sys.exit(1 if failed else 0)