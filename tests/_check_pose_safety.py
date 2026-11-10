# -*- coding: utf-8 -*-
"""验证 taiji_pose_safety.py：定式识别+膝关节安全检测"""
import sys
sys.path.insert(0, ".")
from taiji_pose_safety import (
    load_templates, normalize_and_flat, calculate_angle,
    check_knee_risk, run_recognize,
    LEFT_LEG, RIGHT_LEG, SIM_THRESHOLD, ANGLE_WARN_THRESHOLD,
)
import numpy as np
from pathlib import Path

print("=== 1. 定式模板加载 ===")
ts = load_templates()
assert len(ts) == 24, f"应24式，实际{len(ts)}"
assert all(len(t["vector"]) == 99 for t in ts), "向量应99维"
assert all(t["health_note"] for t in ts), "安全提示不应为空"
print(f"  24式定式模板 OK（99维向量+专属安全提示）")
print(f"  LEFT_LEG={LEFT_LEG}  RIGHT_LEG={RIGHT_LEG}")
print(f"  SIM_THRESHOLD={SIM_THRESHOLD}  ANGLE_WARN_THRESHOLD={ANGLE_WARN_THRESHOLD}")

print("\n=== 2. 膝关节角度计算 ===")
# 2a 直腿（180°）：髋/膝/踝三点共线竖直
hip = np.array([0.5, 0.3])
knee = np.array([0.5, 0.5])
ankle = np.array([0.5, 0.7])
a180 = calculate_angle(hip, knee, ankle)
print(f"  直腿角度：{a180:.1f}°（应≈180）")
assert abs(a180 - 180) < 1.0
# 2b 直角（90°）：髋在膝正上，踝在膝正右
hip2 = np.array([0.5, 0.3])
knee2 = np.array([0.5, 0.5])
ankle2 = np.array([0.7, 0.5])
a90 = calculate_angle(hip2, knee2, ankle2)
print(f"  直角角度：{a90:.1f}°（应≈90）")
assert abs(a90 - 90) < 1.0
print("  角度计算正确 OK")

print("\n=== 3. 膝关节风险检测 ===")
# 3a 安全姿态：直腿180° > 95° 且膝盖不超踝
pts_safe = np.zeros((33, 2), dtype=np.float32)
pts_safe[23] = [0.5, 0.3]; pts_safe[25] = [0.5, 0.5]; pts_safe[27] = [0.5, 0.7]  # 左腿直
pts_safe[24] = [0.5, 0.3]; pts_safe[26] = [0.5, 0.5]; pts_safe[28] = [0.5, 0.7]  # 右腿直
la, lr, lo = check_knee_risk(pts_safe, *LEFT_LEG)
ra, rr, ro = check_knee_risk(pts_safe, *RIGHT_LEG)
print(f"  安全姿态：左膝{la:.0f}°风险={lr}  右膝{ra:.0f}°风险={rr}（应均无风险）")
assert not lr and not rr
# 3b 风险姿态：深蹲90° < 95°
pts_risk = np.zeros((33, 2), dtype=np.float32)
pts_risk[23] = [0.5, 0.4]; pts_risk[25] = [0.5, 0.55]; pts_risk[27] = [0.7, 0.55]  # 左腿90°
pts_risk[24] = [0.5, 0.4]; pts_risk[26] = [0.5, 0.55]; pts_risk[28] = [0.7, 0.55]  # 右腿90°
la2, lr2, lo2 = check_knee_risk(pts_risk, *LEFT_LEG)
ra2, rr2, ro2 = check_knee_risk(pts_risk, *RIGHT_LEG)
print(f"  深蹲姿态：左膝{la2:.0f}°风险={lr2}  右膝{ra2:.0f}°风险={rr2}（应均有风险）")
assert lr2 and rr2
# 3c 右腿索引不越界（原代码33越界崩溃点）
print("  左右腿检测均不崩溃 OK")

print("\n=== 4. 真实视频完整识别流程 ===")
vid = "uploads/7db8e9fc793da435ac23a44e8db55342.mp4"
if Path(vid).exists():
    img, comment, score, name = run_recognize(vid, "正常")
    print(f"  识别招式：{name}")
    print(f"  得分：{score:.1f}")
    print(f"  图片：{'有' if img is not None else '无'}")
    print(f"  评语：\n{comment}")
    assert name != "未知招式", "真实太极视频应能识别"
    assert img is not None
    assert score >= 0
    # 4b 病史提醒
    _, c2, _, _ = run_recognize(vid, "高血压")
    assert "高血压" in c2, "高血压标签应附加提醒"
    _, c3, _, _ = run_recognize(vid, "膝关节炎")
    assert "膝关节" in c3, "膝关节炎标签应附加提醒"
    print("  病史提醒附加 OK")
else:
    print("  测试视频不存在，跳过")

print("\n=== 5. 空视频防护 ===")
img0, c0, s0, n0 = run_recognize(None, "正常")
assert img0 is None and n0 == "未知招式"
print("  空输入返回提示 OK")

print("\n=== 全部验证通过 ===")