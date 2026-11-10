"""
噪声法 d_bad 计算（dtw-python 版）
对每个模板的归一化序列加 25 次高斯噪声（σ=0.028，肩宽=1.0 尺度），
DTW 计算加噪 vs 原始的累积距离，中位数 = d_bad。

适配说明（相对用户原代码的3处必改）：
1. dtw-python 1.5.3：`dist=lambda` → `dist_method="euclidean"`
   （dist 参数名已废弃，会报 TypeError）
2. dtw-python 返回 DTW 对象，不能解包 `dist,_,_,_ = dtw(...)`
   （会 ValueError）→ 用 `align.distance`
3. 新增归一化：模板存像素坐标（几百），σ=0.028 在像素尺度下无意义
   → 先归一化（髋心+肩宽，肩宽=1.0）再加噪声
4. 文件名：saved_templates.json → taiji_templates_final.json
"""
import json
import math
import numpy as np
from pathlib import Path
from dtw import dtw

IN_FILE = "taiji_templates_final.json"
SAMPLE_COUNT = 25
NOISE_STD = 0.028
VIS_VALID_THRESH = 0.40
FALLBACK_D_BAD = 12.0


# ---------- 归一化 ----------
def normalize_frame_33(frame):
    """33点帧归一化：髋部中心为原点，肩宽为单位长度
    返回 [(nx,ny),...] 33个；髋部不可用返回 None
    """
    if frame is None or len(frame) < 33:
        return None
    lhip, rhip = frame[23], frame[24]
    if lhip.get("visibility", 0) < VIS_VALID_THRESH or rhip.get("visibility", 0) < VIS_VALID_THRESH:
        return None
    cx = (lhip["x"] + rhip["x"]) / 2.0
    cy = (lhip["y"] + rhip["y"]) / 2.0
    lsh, rsh = frame[11], frame[12]
    if lsh.get("visibility", 0) >= VIS_VALID_THRESH and rsh.get("visibility", 0) >= VIS_VALID_THRESH:
        scale = math.hypot(lsh["x"] - rsh["x"], lsh["y"] - rsh["y"]) + 1e-8
    else:
        scale = math.hypot(lhip["x"] - rhip["x"], lhip["y"] - rhip["y"]) * 2 + 1e-8
    return [((p["x"] - cx) / scale, (p["y"] - cy) / scale) for p in frame]


def normalize_sequence(frames):
    """归一化帧序列，跳过失败帧"""
    out = []
    for f in frames:
        nf = normalize_frame_33(f)
        if nf is not None:
            out.append(nf)
    return out


# ---------- DTW ----------
def frame_to_vector(frame):
    """归一化帧 [(nx,ny),...] → np.array 66维"""
    vec = []
    for p in frame:
        vec.append(float(p[0]))
        vec.append(float(p[1]))
    return np.array(vec, dtype=np.float32)


def resample_sequence(seq, target_len):
    n = len(seq)
    if n == target_len:
        return seq
    out = []
    for i in range(target_len):
        ratio = i / max(target_len - 1, 1)
        src = int(np.clip(ratio * (n - 1), 0, n - 1))
        out.append(seq[src])
    return out


def compute_dtw_dist(seq_a, seq_b):
    """dtw-python 累积 DTW 距离"""
    if len(seq_a) < 5 or len(seq_b) < 5:
        return 1e9
    target_n = len(seq_b)
    a_sampled = resample_sequence(seq_a, target_n)
    mat_a = np.array([frame_to_vector(f) for f in a_sampled])
    mat_b = np.array([frame_to_vector(f) for f in seq_b])
    # dtw-python 1.5.3：dist_method 参数，返回 DTW 对象
    align = dtw(mat_a, mat_b, dist_method="euclidean")
    return float(align.distance)


# ---------- 噪声法 d_bad ----------
def estimate_shoulder_width(raw_frames):
    """估算模板的平均肩宽（像素），用于确定噪声尺度"""
    widths = []
    for frame in raw_frames[:20]:  # 取前20帧估算即可
        if frame is None or len(frame) < 33:
            continue
        lsh, rsh = frame[11], frame[12]
        if lsh.get("visibility", 0) >= VIS_VALID_THRESH and rsh.get("visibility", 0) >= VIS_VALID_THRESH:
            widths.append(math.hypot(lsh["x"] - rsh["x"], lsh["y"] - rsh["y"]))
    return float(np.median(widths)) if widths else 100.0


def calc_one_dbad(raw_frames, sample_count=SAMPLE_COUNT, noise_std=NOISE_STD):
    """噪声法 d_bad（模拟完整评分流程）
    在原始像素坐标加噪（σ=肩宽*noise_std，模拟检测噪声）→ 归一化 → DTW
    与 taiji_score.calc_dtw_score 的口径完全一致
    """
    base_norm = normalize_sequence(raw_frames)
    if len(base_norm) < 5:
        return FALLBACK_D_BAD

    shoulder_w = estimate_shoulder_width(raw_frames)
    sigma_pixel = shoulder_w * noise_std

    dists = []
    for _ in range(sample_count):
        # 在原始坐标加噪（模拟关键点检测噪声）
        noisy_raw = []
        for frame in raw_frames:
            if frame is None:
                continue
            nf = []
            for pt in frame:
                nf.append({
                    "idx": pt["idx"],
                    "x": pt["x"] + np.random.normal(0, sigma_pixel),
                    "y": pt["y"] + np.random.normal(0, sigma_pixel),
                    "z": pt["z"],
                    "visibility": pt["visibility"],
                })
            noisy_raw.append(nf)
        # 走评分流程的归一化
        noisy_norm = normalize_sequence(noisy_raw)
        if len(noisy_norm) < 5:
            continue
        d = compute_dtw_dist(noisy_norm, base_norm)
        dists.append(d)
    if not dists:
        return FALLBACK_D_BAD
    return float(np.median(dists))


def main():
    np.random.seed(42)
    in_file = Path(IN_FILE)
    if not in_file.exists():
        print(f"❌ 找不到模板文件：{IN_FILE}")
        return

    with in_file.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, dict):
        print("❌ 模板必须是字典格式")
        return

    # 备份旧 d_bad 用于对比
    old_dbad = {}
    for name, item in raw_data.items():
        if isinstance(item, dict) and "d_bad" in item:
            old_dbad[name] = item["d_bad"]

    out_data = {}
    total = len(raw_data)
    ok_count = 0

    for name, item in raw_data.items():
        print(f"\n处理：{name}")
        frames = item.get("frames") if isinstance(item, dict) else item
        if not isinstance(frames, list) or len(frames) < 15:
            print(f"   ⚠️ 跳过：帧数不足")
            out_data[name] = item
            continue

        # 归一化检查（确保模板可归一化）
        norm = normalize_sequence(frames)
        if len(norm) < 10:
            print(f"   ⚠️ 跳过：归一化后有效帧不足（{len(norm)}）")
            out_data[name] = item
            continue

        try:
            d_bad_val = calc_one_dbad(frames)
            out_data[name] = {
                "frames": frames,
                "d_bad": round(d_bad_val, 4),
            }
            old_val = old_dbad.get(name)
            old_str = f"｜旧(交叉比对法): {old_val}" if old_val is not None else ""
            print(f"   ✅ d_bad = {d_bad_val:.4f}{old_str}")
            ok_count += 1
        except Exception as e:
            print(f"   ⚠️ 计算失败：{e}，保留旧值")
            out_data[name] = item

    with in_file.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("===== 汇总 =====")
    print(f"读取总数：{total}")
    print(f"成功生成：{ok_count}")
    print(f"输出文件：{IN_FILE}（d_bad 已更新为噪声法）")


if __name__ == "__main__":
    main()