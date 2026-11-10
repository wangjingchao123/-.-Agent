# -*- coding: utf-8 -*-
"""烟雾测试：验证 taiji_ui.py 可导入、JSON可加载、核心函数逻辑正确"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("=== 1. 导入模块 ===")
import taiji_ui
print("OK: taiji_ui imported")

print("\n=== 2. 检查模板加载 ===")
print(f"OK: {len(taiji_ui.template_list)} templates, {len(taiji_ui.action_options)} options")
t0 = taiji_ui.template_list[0]
print(f"  first: {t0['template_name']}, threshold={t0['similarity_threshold']}, rules={len(t0['safety_checks'])}")
assert taiji_ui.template_map[t0['template_name']] is t0, "template_map mismatch"

print("\n=== 3. 测试 normalize_and_flat ===")
pts = np.zeros((33, 4), dtype=np.float32)
pts[taiji_ui.L_HIP] = [0.45, 0.60, 0, 0.9]
pts[taiji_ui.R_HIP] = [0.55, 0.60, 0, 0.9]
pts[taiji_ui.L_SHOULDER] = [0.46, 0.35, 0, 0.9]
pts[taiji_ui.R_SHOULDER] = [0.54, 0.35, 0, 0.9]
vec = taiji_ui.normalize_and_flat(pts[:, :3])
print(f"OK: vector shape={vec.shape}, range=[{vec.min():.3f}, {vec.max():.3f}]")
assert vec.shape == (99,), f"Expected (99,), got {vec.shape}"
hip_l = vec[taiji_ui.L_HIP * 3: taiji_ui.L_HIP * 3 + 3]
hip_r = vec[taiji_ui.R_HIP * 3: taiji_ui.R_HIP * 3 + 3]
hip_mid = (hip_l + hip_r) / 2.0
assert np.all(np.abs(hip_mid) < 0.01), f"hip midpoint not centered: {hip_mid}"
print("  hip midpoint centered correctly")

print("\n=== 4. 测试 calc_angle ===")
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 0.0, 0.0])
c = np.array([0.0, 1.0, 0.0])
angle = taiji_ui.calc_angle(a, b, c)
print(f"OK: right angle = {angle:.1f}°")
assert abs(angle - 90.0) < 1.0, f"Expected ~90°, got {angle}"

print("\n=== 5. 测试 run_safety_check ===")
# 正常站立姿态
pts3 = pts.copy()
pts3[taiji_ui.L_KNEE] = [0.45, 0.78, 0, 0.8]
pts3[taiji_ui.R_KNEE] = [0.55, 0.78, 0, 0.8]
pts3[taiji_ui.L_ANKLE] = [0.45, 0.92, 0, 0.7]
pts3[taiji_ui.R_ANKLE] = [0.55, 0.92, 0, 0.7]
pts3[taiji_ui.NOSE] = [0.50, 0.28, 0, 0.9]
pts3[taiji_ui.L_WRIST] = [0.40, 0.50, 0, 0.8]
pts3[taiji_ui.R_WRIST] = [0.60, 0.50, 0, 0.8]

rules = t0["safety_checks"]
warnings = taiji_ui.run_safety_check(pts3, rules)
print(f"OK: standing pose -> {warnings}")
assert any("姿态安全" in w for w in warnings), "Standing should be safe"

# 膝盖超脚尖应触发警告
pts4 = pts3.copy()
pts4[taiji_ui.L_KNEE] = [0.52, 0.78, 0, 0.8]
pts4[taiji_ui.L_ANKLE] = [0.44, 0.92, 0, 0.7]
warnings2 = taiji_ui.run_safety_check(pts4, rules)
toe_warnings = [w for w in warnings2 if "脚尖" in w]
print(f"OK: knee-over-toe -> {len(toe_warnings)} toe warnings")
assert len(toe_warnings) >= 1, f"Should trigger toe warning: {warnings2}"

print("\n=== 6. 测试 hand_height_diff 用腕关节 ===")
# 白鹤亮翅的规则含 hand_height_diff
bhlc = taiji_ui.template_map.get("白鹤亮翅")
if bhlc:
    hand_rules = [r for r in bhlc["safety_checks"] if r["type"] == "hand_height_diff"]
    if hand_rules:
        # 双腕同高应触发"高度差不足"警告
        pts5 = pts3.copy()
        pts5[taiji_ui.L_WRIST] = [0.40, 0.40, 0, 0.8]
        pts5[taiji_ui.R_WRIST] = [0.60, 0.40, 0, 0.8]
        w3 = taiji_ui.run_safety_check(pts5, hand_rules)
        hand_warns = [w for w in w3 if "高度差" in w]
        print(f"OK: same-height wrists -> {len(hand_warns)} hand_diff warnings")
        assert len(hand_warns) >= 1, f"Should trigger hand_diff: {w3}"
        # 双腕高度差大不应触发
        pts5[taiji_ui.R_WRIST] = [0.60, 0.20, 0, 0.8]
        w4 = taiji_ui.run_safety_check(pts5, hand_rules)
        hand_warns2 = [w for w in w4 if "高度差" in w]
        print(f"OK: different-height wrists -> {len(hand_warns2)} hand_diff warnings (expect 0)")
        assert len(hand_warns2) == 0, f"Should not trigger: {w4}"

print("\n=== ALL SMOKE TESTS PASSED ===")