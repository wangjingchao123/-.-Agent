# -*- coding: utf-8 -*-
"""Tab1 诊断：模板两两相似度gap分布 + 被拒视频谷值帧vs全帧最佳相似度"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.spatial.distance import cosine
import app

app.load_data()
names = [t["template_name"] for t in app.keyframe_templates]
vecs = [np.array(t["vector"], dtype=np.float32) for t in app.keyframe_templates]

print("== A. 每式：自身(=1.0) vs 最相似其他式 → gap ==")
gaps = []
for i in range(len(names)):
    sims = [(1.0 - cosine(vecs[i], vecs[j]), names[j]) for j in range(len(names)) if j != i]
    sims.sort(reverse=True)
    top = sims[0]
    gap = 1.0 - top[0]
    gaps.append(gap)
    flag = " <-- gap<0.08歧义风险" if gap < app.SIM_GAP_MIN else ""
    print(f"  {names[i]}: 最相似={top[1]} sim={top[0]:.4f} gap={gap:.4f}{flag}")
gaps = np.array(gaps)
print(f"\n  gap分布: min={gaps.min():.4f} 中位={np.median(gaps):.4f} max={gaps.max():.4f}")
print(f"  gap<0.08的数量: {(gaps < app.SIM_GAP_MIN).sum()}/24")

print("\n== B. 被拒视频：谷值帧最佳 vs 全帧最佳相似度 ==")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vids = sorted(glob.glob(os.path.join(root, "uploads", "*.mp4")))
for v in vids:
    cap = app.cv2.VideoCapture(v)
    frames_data = []
    fi = 0
    while cap.isOpened() and fi < app.MAX_FRAMES_READ:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % app.FIXED_STEP == 0:
            rgb = app.cv2.cvtColor(frame, app.cv2.COLOR_BGR2RGB)
            res = app.pose_global.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                vec = app.normalize_and_flat_landmarks(lm)
                frames_data.append(vec)
        fi += 1
    cap.release()
    if len(frames_data) < 5:
        print(f"  {os.path.basename(v)}: 帧不足")
        continue
    arr = np.array(frames_data)
    # 全帧最佳
    all_best = -1.0
    all_name = ""
    for vec in arr:
        for t in app.keyframe_templates:
            s = 1.0 - cosine(vec, np.array(t["vector"], dtype=np.float32))
            if s > all_best:
                all_best, all_name = s, t["template_name"]
    # 谷值帧最佳
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    valleys, _ = app.find_peaks(-diffs, distance=app.MIN_VALLEY_DIST)
    if len(valleys) == 0:
        valleys = np.array([int(np.argmin(diffs))])
    val_best = -1.0
    val_name = ""
    for vi in valleys:
        for t in app.keyframe_templates:
            s = 1.0 - cosine(arr[vi], np.array(t["vector"], dtype=np.float32))
            if s > val_best:
                val_best, val_name = s, t["template_name"]
    tag = "过门槛" if val_best >= app.SIM_MIN_ABSOLUTE else "被拒"
    print(f"  {os.path.basename(v)}: 全帧最佳={all_best:.4f}({all_name}) "
          f"谷值最佳={val_best:.4f}({val_name}) 谷值数={len(valleys)} -> {tag}")