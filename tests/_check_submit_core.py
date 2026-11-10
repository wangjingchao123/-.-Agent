# -*- coding: utf-8 -*-
"""验证 taiji_joycode_submit.py 核心链路
覆盖：模板库预加载/分体加权DTW/腿部平滑/双门槛片段/红圈帧/自动识别/评委录入模板
"""
import sys
sys.path.insert(0, ".")
from taiji_joycode_submit import (
    preload_templates, template_library, load_template,
    dtw_with_path, score_from_distance, judge_level,
    locate_error_segments, normalize_sequence, smooth_leg_sequence,
    extract_error_frames, extract_seq_from_video,
    extract_sample_frames, sample_indices_by_percent,
    quick_compare, auto_select_template, add_new_template,
    ACTION_LIST, TEMPLATE_DIR, NUM_KEYPOINTS,
    LEG_IDS, UPPER_IDS, THRESH_UPPER, THRESH_LEG,
    MIN_FRAME_UPPER, MIN_FRAME_LEG, WEIGHT_UPPER, WEIGHT_LEG,
    RECOG_THRESH, TARGET_SEQ_LEN,
    MATCH_DIST_THRESHOLD, SAMPLE_RATIO, MAX_SAMPLE_IMAGES,
)
import numpy as np
from pathlib import Path

vfn = list(range(0, 240, 2))  # 模拟120个有效帧的原视频帧号

print("=== 1. 模板库预加载 ===")
preload_templates()
assert len(ACTION_LIST) == 24
missing = [n for n in ACTION_LIST if not (TEMPLATE_DIR / f"{n}.npz").exists()]
assert not missing, f"缺失模板文件：{missing}"
assert len(template_library) >= 24, f"预加载后模板库应≥24，实际{len(template_library)}"
tpl_qs = load_template("起势")
tpl_ss = load_template("收势")
assert tpl_qs is not None and tpl_qs.shape == (120, 99)
print(f"  template_library 预加载 {len(template_library)} 个招式 OK")
print(f"  LEG_IDS={LEG_IDS}  UPPER数={len(UPPER_IDS)}  权重: 上身{WEIGHT_UPPER}/腿部{WEIGHT_LEG}")
print(f"  RECOG_THRESH={RECOG_THRESH}  MATCH_DIST_THRESHOLD={MATCH_DIST_THRESHOLD}")
print(f"  SAMPLE_RATIO={SAMPLE_RATIO}  MAX_SAMPLE_IMAGES={MAX_SAMPLE_IMAGES}")

print("\n=== 2. 分体加权DTW：自比对 ===")
dist, path, pc, pcu, pcl, kpc = dtw_with_path(tpl_qs, tpl_qs)
score = score_from_distance(dist)
print(f"  dist={dist:.6f}  path={len(path)}  pc={len(pc)}  kpc={kpc.shape}  score={score:.1f}")
assert dist < 0.001 and score == 100.0
assert kpc.shape == (len(path), NUM_KEYPOINTS)
level, _ = judge_level(score)
assert level == "优秀"
print(f"  等级={level} OK")

print("\n=== 3. 噪声扰动（σ=0.01）===")
rng = np.random.default_rng(42)
noisy = normalize_sequence(tpl_qs + rng.normal(0, 0.01, tpl_qs.shape))
noisy = smooth_leg_sequence(noisy)
dist_n, path_n, pc_n, pcu_n, pcl_n, kpc_n = dtw_with_path(noisy, tpl_qs)
score_n = score_from_distance(dist_n)
print(f"  dist={dist_n:.4f}  上身偏差均值={np.mean(pcu_n):.4f}  腿部偏差均值={np.mean(pcl_n):.4f}")
print(f"  score={score_n:.1f}  上身门槛{THRESH_UPPER}  腿部门槛{THRESH_LEG}")
assert 60 <= score_n <= 100, f"噪声分数应在60-100，实际{score_n}"
segs_n = locate_error_segments(path_n, 120, vfn, pcu_n, pcl_n, kpc_n)
print(f"  噪声检出错误段：{len(segs_n)}（应很少或0）")

print("\n=== 4. 交叉比对（起势 vs 收势）===")
dist_x, path_x, pc_x, pcu_x, pcl_x, kpc_x = dtw_with_path(tpl_qs, tpl_ss)
score_x = score_from_distance(dist_x)
print(f"  dist={dist_x:.4f}  score={score_x:.1f}")
assert score_x < score_n, f"交叉分数应低于噪声分数"

print("\n=== 5. 双门槛片段检测 ===")
segs_x = locate_error_segments(path_x, 120, vfn, pcu_x, pcl_x, kpc_x)
print(f"  交叉检出 {len(segs_x)} 个错误段：")
for i, s in enumerate(segs_x, 1):
    kind = "下肢" if s["is_leg"] else "上身"
    print(f"    第{i}段：原帧{s['orig_start']}~{s['orig_end']}  {kind}  代表帧{s['orig_frame']}")
    print(f"      top_kp={s['top_kp']}  偏差={[f'{d:.2f}' for d in s['top_kp_dev']]}")
    assert s["orig_start"] <= s["orig_end"]
assert len(segs_x) <= 3
print("  OK")

print("\n=== 6. 腿部平滑效果 ===")
spike = tpl_qs.copy()
spike[60, 27 * 3:29 * 3] += 2.0  # 单帧脚踝大幅漂移（踝27/28在LEG_IDS内）
smoothed = smooth_leg_sequence(spike)
raw_dev = np.linalg.norm(spike[60, 27 * 3:29 * 3] - tpl_qs[60, 27 * 3:29 * 3])
sm_dev = np.linalg.norm(smoothed[60, 27 * 3:29 * 3] - tpl_qs[60, 27 * 3:29 * 3])
print(f"  平滑前踝部偏差={raw_dev:.4f}  平滑后={sm_dev:.4f}（应大幅削弱）")
assert sm_dev < raw_dev * 0.6, f"平滑应削弱单帧漂移，{sm_dev} vs {raw_dev}"
assert np.array_equal(smoothed[60, :23 * 3], spike[60, :23 * 3]), "上身不应被平滑"
assert np.array_equal(smoothed[60, 29 * 3:33 * 3], spike[60, 29 * 3:33 * 3]), "脚跟脚尖不应被平滑"
print("  腿部平滑削弱单帧漂移、上身与脚跟脚尖不受影响 OK")

print("\n=== 7. 瞬时漂移过滤（脚踝/膝盖单帧）===")
# 7a 脚踝单帧漂移
d_sp, p_sp, pc_sp, pcu_sp, pcl_sp, kpc_sp = dtw_with_path(smoothed, tpl_qs)
sc_sp = score_from_distance(d_sp)
segs_sp = locate_error_segments(p_sp, 120, vfn, pcu_sp, pcl_sp, kpc_sp)
print(f"  脚踝瞬时漂移：dist={d_sp:.4f}  score={sc_sp:.1f}  错误段={len(segs_sp)}")
assert len(segs_sp) == 0, f"脚踝瞬时漂移不应报错误段，实际{len(segs_sp)}个"
# 7b 膝盖单帧漂移
spike_k = tpl_qs.copy()
spike_k[60, 25 * 3:27 * 3] += 1.5
sm_k = smooth_leg_sequence(spike_k)
d_sk, p_sk, pc_sk, pcu_sk, pcl_sk, kpc_sk = dtw_with_path(sm_k, tpl_qs)
segs_sk = locate_error_segments(p_sk, 120, vfn, pcu_sk, pcl_sk, kpc_sk)
print(f"  膝盖瞬时漂移：dist={d_sk:.4f}  错误段={len(segs_sk)}")
assert len(segs_sk) == 0, f"膝盖瞬时漂移不应报错误段，实际{len(segs_sk)}个"
# 7c 脚跟脚尖单帧漂移（不在LEG_IDS，属UPPER，需连续5帧才报）
spike_f = tpl_qs.copy()
spike_f[60, 29 * 3:33 * 3] += 1.5
sm_f = smooth_leg_sequence(spike_f)
d_sf, p_sf, pc_sf, pcu_sf, pcl_sf, kpc_sf = dtw_with_path(sm_f, tpl_qs)
segs_sf = locate_error_segments(p_sf, 120, vfn, pcu_sf, pcl_sf, kpc_sf)
print(f"  脚跟脚尖瞬时漂移：dist={d_sf:.4f}  错误段={len(segs_sf)}")
assert len(segs_sf) == 0, f"脚跟脚尖单帧漂移不应报错误段，实际{len(segs_sf)}个"
print("  踝/膝/脚跟脚尖瞬时漂移均不报错误 OK")

print("\n=== 8. 持续偏离检出 ===")
# 8a 整腿持续偏离（帧40~80髋膝踝）
sust = tpl_qs.copy()
sust[40:80, 23 * 3:29 * 3] += 1.0
sm_su = smooth_leg_sequence(sust)
d_su, p_su, pc_su, pcu_su, pcl_su, kpc_su = dtw_with_path(sm_su, tpl_qs)
segs_su = locate_error_segments(p_su, 120, vfn, pcu_su, pcl_su, kpc_su)
print(f"  整腿持续偏离：dist={d_su:.4f}  错误段={len(segs_su)}")
has_leg = any(s["is_leg"] for s in segs_su)
for s in segs_su:
    kind = "下肢" if s["is_leg"] else "上身"
    print(f"    {kind}段：帧{s['orig_start']}~{s['orig_end']}  top_kp={s['top_kp']}")
assert len(segs_su) > 0 and has_leg, "持续腿部偏离应检出下肢片段"
# 8b 上身持续偏离（帧30~70手臂）
sust_u = tpl_qs.copy()
sust_u[30:70, 15 * 3:22 * 3] += 1.0
sm_su2 = smooth_leg_sequence(sust_u)
d_su2, p_su2, pc_su2, pcu_su2, pcl_su2, kpc_su2 = dtw_with_path(sm_su2, tpl_qs)
segs_su2 = locate_error_segments(p_su2, 120, vfn, pcu_su2, pcl_su2, kpc_su2)
print(f"  上身持续偏离：dist={d_su2:.4f}  错误段={len(segs_su2)}")
has_upper = any(not s["is_leg"] for s in segs_su2)
assert len(segs_su2) > 0 and has_upper, "上身持续偏离应检出上身片段"
print("  下肢/上身持续偏离均检出 OK")

print("\n=== 9. 真实视频：抽样样图提取（extract_seq_from_video 统一归一化）===")
vid = "uploads/7db8e9fc793da435ac23a44e8db55342.mp4"
if Path(vid).exists():
    result = extract_seq_from_video(vid)
    assert result is not None, "真实视频应能提取序列"
    nu_raw, all_lm, vfn_r = result
    vc = len(all_lm)
    print(f"  有效帧数：{vc}  序列shape：{nu_raw.shape}")
    nu = smooth_leg_sequence(nu_raw)
    d3, p3, pc3, pcu3, pcl3, kpc3 = dtw_with_path(nu, tpl_qs)
    print(f"  dist={d3:.4f}（匹配门槛{MATCH_DIST_THRESHOLD}）")
    segs3 = locate_error_segments(p3, vc, vfn_r, pcu3, pcl3, kpc3)
    sample_idx = sample_indices_by_percent(vc)
    imgs = extract_sample_frames(vid, sample_idx, vfn_r, all_lm, segs3)
    print(f"  抽样索引：{len(sample_idx)} 个  抽样图：{len(imgs)} 张  错误段：{len(segs3)}")
    for s in segs3:
        kind = "下肢" if s["is_leg"] else "上身"
        print(f"    {kind}段：帧{s['orig_start']}~{s['orig_end']}")
    assert len(imgs) == len(sample_idx), f"抽样图数应=抽样索引数，{len(imgs)} vs {len(sample_idx)}"
    assert 1 <= len(imgs) <= MAX_SAMPLE_IMAGES
    if imgs:
        print(f"  图片尺寸示例：{imgs[0].shape}")
    print("  抽样样图提取 OK")
else:
    print("  测试视频不存在，跳过")

print("\n=== 10. 自动识别 ===")
rng = np.random.default_rng(7)
# 10a 模板+噪声 → 识别为原招式
for name in ["起势", "云手", "左右搂膝拗步", "海底针"]:
    t = template_library[name]
    noisy = t + rng.normal(0, 0.15, t.shape).astype(np.float32)
    hit, d = auto_select_template(noisy)
    print(f"  {name}+噪声σ=0.15 → 识别：{hit}（距离{d:.4f}）")
    assert hit == name, f"应识别为{name}，实际{hit}"
    assert d < RECOG_THRESH
print("  同招式噪声识别正确 4/4 OK")
# 10b 随机乱动（非太极动作，姿态本身不同） → 拒绝
# 注：帧序打乱不适用——姿态集合不变，DTW弹性对齐会吸收节奏错乱，识别为原招式属合理行为
fake = rng.normal(0, 3.0, (120, 99)).astype(np.float32)
hit2, d2 = auto_select_template(fake)
print(f"  随机乱动 → 识别：{hit2}（距离{d2:.4f}，阈值{RECOG_THRESH}）")
assert hit2 is None, f"随机乱动应被拒绝，实际匹配{hit2}"
print("  非太极动作拒绝 OK")

print("\n=== 11. 评委录入新模板 ===")
vid2 = "uploads/7db8e9fc793da435ac23a44e8db55342.mp4"
if Path(vid2).exists():
    n_before = len(template_library)
    msg, _ = add_new_template(vid2, "测试新招式")
    print(f"  添加：{msg}")
    assert "添加成功" in msg
    assert len(template_library) == n_before + 1
    new_tpl = load_template("测试新招式")
    assert new_tpl is not None and new_tpl.shape == (TARGET_SEQ_LEN, 99)
    msg2, _ = add_new_template(vid2, "测试新招式")
    print(f"  重复添加：{msg2}")
    assert "已经存在" in msg2
    msg3, _ = add_new_template(vid2, "   ")
    assert "请输入" in msg3
    del template_library["测试新招式"]
    print(f"  添加/查重/空名校验 OK（已清理测试模板，库回 {len(template_library)} 个）")
else:
    print("  测试视频不存在，跳过")

print("\n=== 12. 整体匹配门槛 ===")
# 12a 同招式噪声（σ=0.20 大个体差异）应通过门槛
rng3 = np.random.default_rng(99)
t = template_library["起势"]
noisy_m = t + rng3.normal(0, 0.20, t.shape).astype(np.float32)
d_m = dtw_with_path(smooth_leg_sequence(noisy_m), t)[0]
print(f"  同招式+噪声σ=0.20：dist={d_m:.4f}（门槛{MATCH_DIST_THRESHOLD}，应通过）")
assert d_m < MATCH_DIST_THRESHOLD, f"同招式大差异应通过门槛，实际{d_m}"
# 12b 交叉模板（起势 vs 收势）应被门槛拦截
d_cross = dtw_with_path(tpl_qs, tpl_ss)[0]
print(f"  交叉（起势vs收势）：dist={d_cross:.4f}（门槛{MATCH_DIST_THRESHOLD}，应拦截）")
assert d_cross > MATCH_DIST_THRESHOLD, f"交叉应被门槛拦截，实际{d_cross}"
print("  匹配门槛：同招式通过/交叉拦截 OK")

print("\n=== 13. 均匀抽样函数 ===")
# 13a 常规帧数
idx_120 = sample_indices_by_percent(120)
print(f"  120帧 → 抽样{len(idx_120)}个：首={idx_120[0]} 尾={idx_120[-1]}")
assert idx_120[0] == 0 and idx_120[-1] == 119, "首尾必选"
assert len(idx_120) == max(1, int(np.ceil(120 * SAMPLE_RATIO)))
assert len(idx_120) <= MAX_SAMPLE_IMAGES
# 13b 大帧数封顶
idx_big = sample_indices_by_percent(1000)
print(f"  1000帧 → 抽样{len(idx_big)}个（封顶{MAX_SAMPLE_IMAGES}）")
assert len(idx_big) <= MAX_SAMPLE_IMAGES
# 13c 极小帧数
idx_small = sample_indices_by_percent(3)
print(f"  3帧 → 抽样{len(idx_small)}个")
assert len(idx_small) >= 1
print("  抽样：首尾必选/封顶/极小帧数 OK")

print("\n=== 全部验证通过 ===")