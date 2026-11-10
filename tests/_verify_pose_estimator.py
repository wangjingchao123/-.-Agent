# -*- coding: utf-8 -*-
"""验证 ultralytics 8.4.191 solutions.PoseEstimator API"""
import numpy as np
import traceback

print("=== 1. import solutions ===")
try:
    from ultralytics import solutions
    print("  [OK] from ultralytics import solutions")
    print("  dir(solutions):", [x for x in dir(solutions) if not x.startswith('_')][:20])
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

print("\n=== 2. PoseEstimator 是否存在 ===")
try:
    cls = solutions.PoseEstimator
    print(f"  [OK] solutions.PoseEstimator = {cls}")
except AttributeError:
    print("  [FAIL] solutions 没有 PoseEstimator")
    # 尝试找替代
    for name in dir(solutions):
        if 'pose' in name.lower() or 'Pose' in name:
            print(f"  候选: {name}")

print("\n=== 3. PoseEstimator 构造 ===")
try:
    sol = solutions.PoseEstimator(show=False)
    print(f"  [OK] PoseEstimator(show=False) 构造成功: {type(sol)}")
    methods = [x for x in dir(sol) if not x.startswith('_')]
    print(f"  方法: {methods}")
except Exception as e:
    print(f"  [FAIL] 构造失败: {e}")
    traceback.print_exc()
    sol = None

print("\n=== 4. estimate 方法调用 ===")
if sol is not None:
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        out = sol.estimate(black)
        print(f"  [OK] estimate() 返回: {type(out)}")
        if hasattr(out, 'shape'):
            print(f"  shape: {out.shape}")
    except Exception as e:
        print(f"  [FAIL] estimate 失败: {e}")
        traceback.print_exc()
    try:
        res = sol.results
        print(f"  [OK] sol.results = {type(res)}, len={len(res) if res else 0}")
        if res:
            r0 = res[0]
            print(f"  results[0] type: {type(r0)}")
            if hasattr(r0, 'keypoints'):
                kd = r0.keypoints.data
                print(f"  keypoints.data type: {type(kd)}, shape: {kd.shape if hasattr(kd,'shape') else 'N/A'}")
    except Exception as e:
        print(f"  [FAIL] sol.results 失败: {e}")
        traceback.print_exc()

print("\n=== 5. yolov8n-pose.pt 直接输出关键点 ===")
try:
    from ultralytics import YOLO
    yolo = YOLO("yolov8n-pose.pt")
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    r = yolo(black, verbose=False, conf=0.5, classes=0)
    print(f"  [OK] results len={len(r)}")
    r0 = r[0]
    print(f"  boxes: {len(r0.boxes)}")
    if hasattr(r0, 'keypoints'):
        kd = r0.keypoints.data
        print(f"  keypoints.data shape: {kd.shape}")
        print(f"  keypoints.data[0] shape: {kd[0].shape if len(kd)>0 else 'N/A'}")
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

print("\n验证完成")