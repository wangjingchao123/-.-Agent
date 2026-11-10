import json
import math
import numpy as np
from pathlib import Path
from dtw import dtw


# ===================== 配置区 =====================
TEMPLATE_FILE = "taiji_templates_final.json"

RELAXED_NAMES = {
    "右下势独立",
    "左下势独立",
    "金鸡独立",
    "海底针",
    "闪通臂",
    "转身搬拦捶",
    "如封似闭",
    "十字手",
    "收势"
}

DEFAULT_BAD_RATIO = 0.40
DEFAULT_HEAVY_RATIO = 0.30
RELAXED_BAD_RATIO = 0.52
RELAXED_HEAVY_RATIO = 0.42
MOTION_THRESHOLD = 0.022
FALLBACK_D_BAD = 12.0
VIS_VALID_THRESH = 0.40


# ===================== 基础工具 =====================
def frame_to_vector(frame):
    """归一化帧 [(nx,ny),...] → np.array 66维"""
    vec = []
    for p in frame:
        vec.append(float(p[0]))
        vec.append(float(p[1]))
    return np.array(vec, dtype=np.float32)


def resample_sequence(seq, target_len: int):
    n_orig = len(seq)
    if n_orig == target_len:
        return seq
    new_seq = []
    for i in range(target_len):
        ratio = i / max(target_len - 1, 1)
        src_idx = int(np.clip(ratio * (n_orig - 1), 0, n_orig - 1))
        new_seq.append(seq[src_idx])
    return new_seq


def detect_motion(seq):
    """运动检测（归一化坐标下，肩宽=1.0，阈值0.022才有意义）"""
    if not isinstance(seq, list) or len(seq) < 3:
        return False, 0.0
    total_shift = 0.0
    prev_vec = frame_to_vector(seq[0])
    for i in range(1, len(seq)):
        curr_vec = frame_to_vector(seq[i])
        total_shift += np.linalg.norm(curr_vec - prev_vec)
        prev_vec = curr_vec
    avg_shift = total_shift / max(len(seq) - 1, 1)
    return avg_shift > MOTION_THRESHOLD, avg_shift


def filter_suspicious_frames(sequence, is_relaxed: bool):
    if is_relaxed:
        bad_thresh = RELAXED_BAD_RATIO
        heavy_thresh = RELAXED_HEAVY_RATIO
    else:
        bad_thresh = DEFAULT_BAD_RATIO
        heavy_thresh = DEFAULT_HEAVY_RATIO

    clean = []
    total_bad = 0
    if not isinstance(sequence, list):
        return [], True

    for frame in sequence:
        bad_count = 0
        if isinstance(frame, list):
            for pt in frame:
                if isinstance(pt, dict):
                    vis = float(pt.get("visibility", 1.0))
                    if vis < 0.35:
                        bad_count += 1
        ratio = bad_count / max(len(frame) if isinstance(frame, list) else 33, 1)
        if ratio < bad_thresh:
            clean.append(frame)
        else:
            total_bad += 1

    heavy = (total_bad / max(len(sequence), 1)) >= heavy_thresh
    return clean, heavy


# ===================== 归一化（髋部中心+肩宽，消除体型/距离差异） =====================
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


def normalize_clean_sequence(clean_seq):
    """归一化已过滤的帧序列，跳过归一化失败的帧"""
    out = []
    for frame in clean_seq:
        nf = normalize_frame_33(frame)
        if nf is not None:
            out.append(nf)
    return out


# ===================== DTW =====================
def compute_dtw_dist(seq_a, seq_b):
    """dtw-python 累积 DTW 距离（与 calc_dbad_noise.py 口径一致）"""
    if not isinstance(seq_a, list) or not isinstance(seq_b, list):
        return 1e9
    if len(seq_a) < 5 or len(seq_b) < 5:
        return 1e9
    target_n = len(seq_b)
    a_sampled = resample_sequence(seq_a, target_n)
    mat_a = np.array([frame_to_vector(f) for f in a_sampled])
    mat_b = np.array([frame_to_vector(f) for f in seq_b])
    # dtw-python 1.5.3：dist_method 参数（dist已废弃），返回 DTW 对象（不能解包4值）
    align = dtw(mat_a, mat_b, dist_method="euclidean")
    return float(align.distance)


# ===================== 自适应打分 =====================
def adaptive_score(avg_dist: float, d_bad: float):
    if d_bad <= 1e-8:
        return 0.0
    k = 2.2
    max_dist = k * d_bad

    if avg_dist >= max_dist:
        return 0.0

    score = 100.0 * (1.0 - avg_dist / max_dist)
    return float(np.clip(score, 0.0, 100.0))


def get_level_text(score: float):
    if score >= 85:
        return "优秀！动作匹配很好，请继续保持。"
    elif score >= 70:
        return "良好，存在少量姿态偏差。"
    else:
        return "偏差较大，请调整动作。"


# ===================== 主比对入口 =====================
def calc_dtw_score(user_seq, template_seq, action_name: str, d_bad: float):
    try:
        is_relaxed = action_name.strip() in RELAXED_NAMES

        # 1) 可疑帧过滤（两套阈值）
        user_clean, user_heavy_bad = filter_suspicious_frames(user_seq, is_relaxed)
        temp_clean, temp_heavy_bad = filter_suspicious_frames(template_seq, is_relaxed)

        if user_heavy_bad or temp_heavy_bad:
            if is_relaxed:
                msg = "遮挡较多（放宽检测），仍无法识别，请调整姿势与光线。"
            else:
                msg = "识别不稳，请调整光线与姿势。"
            return 0.0, msg, -1.0

        if len(user_clean) < 10 or len(temp_clean) < 10:
            return 0.0, "有效帧不足，无法比对。", -1.0

        # 2) 归一化（髋部中心+肩宽，d_bad 是此口径下算的，必须一致）
        user_norm = normalize_clean_sequence(user_clean)
        temp_norm = normalize_clean_sequence(temp_clean)

        if len(user_norm) < 5 or len(temp_norm) < 5:
            return 0.0, "归一化失败（髋部关键点缺失），请正对摄像头。", -1.0

        # 3) 运动检测（归一化坐标下，阈值0.022才有意义）
        has_motion, _ = detect_motion(user_norm)
        if not has_motion:
            return 0.0, "静止，未检测到动作，请开始演练。", -1.0

        # 4) DTW + 自适应打分
        dist = compute_dtw_dist(user_norm, temp_norm)
        score = adaptive_score(dist, d_bad)
        text = get_level_text(score)
        return round(score, 1), text, float(dist)
    except Exception as e:
        return 0.0, f"内部错误：{str(e)}", -1.0


# ===================== 加载模板库 =====================
def load_templates_only(json_path: str):
    file = Path(json_path)
    if not file.exists():
        raise FileNotFoundError(f"找不到文件：{json_path}")
    try:
        with file.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 损坏：{e}")
    if not isinstance(raw_data, dict):
        raise ValueError("模板文件必须是字典")

    invalid_items = []
    duplicate_items = []
    keep_names = set()
    cleaned_raw = {}

    for name, item in raw_data.items():
        norm_name = name.strip()
        if norm_name in ("倒卷肱", "左右倒卷肱"):
            key = "倒卷肱（标准）"
            if key in keep_names:
                duplicate_items.append(name)
                continue
            keep_names.add(key)
            cleaned_raw[key] = item
        else:
            if norm_name in keep_names:
                duplicate_items.append(name)
                continue
            keep_names.add(norm_name)
            cleaned_raw[norm_name] = item

    templates_out = {}
    for name, item in cleaned_raw.items():
        if not isinstance(item, dict):
            invalid_items.append((name, "不是模板对象"))
            continue
        frames = item.get("frames")
        d_bad_val = float(item.get("d_bad", FALLBACK_D_BAD))

        if not isinstance(frames, list):
            invalid_items.append((name, "缺少 frames"))
            continue
        if len(frames) < 15:
            invalid_items.append((name, f"帧数不足：{len(frames)}"))
            continue

        ok = True
        for fi, fr in enumerate(frames):
            if not isinstance(fr, list) or len(fr) != 33:
                invalid_items.append((name, f"帧 {fi}：不是 33 点"))
                ok = False
                break
        if not ok:
            continue

        templates_out[name] = {
            "frames": frames,
            "d_bad": d_bad_val
        }

    return {
        "readonly": True,
        "templates": templates_out,
        "invalid": invalid_items,
        "duplicates": duplicate_items
    }


# ===================== 样本读写 =====================
def save_test_sample(filepath: str, seq):
    with Path(filepath).open("w", encoding="utf-8") as f:
        json.dump(seq, f, ensure_ascii=False, indent=2)


def load_test_sample(filepath: str):
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"样本不存在：{filepath}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ===================== 实验入口 =====================
def run_experiment(json_template: str, sample_path: str | None, choose_name: str):
    print("=" * 82)
    print("【已校验｜自适应打分 + d_bad】")
    print("=" * 82)
    try:
        lib = load_templates_only(json_template)
    except Exception as e:
        print(f"\n❌ 加载模板失败：{e}")
        return

    templates = lib["templates"]
    if choose_name not in templates:
        print(f'\n❌ "{choose_name}"不存在。可用招式：')
        for k in sorted(templates):
            print(f"   - {k}")
        return

    tmpl = templates[choose_name]
    template_seq = tmpl["frames"]
    d_bad_val = tmpl["d_bad"]
    is_relaxed = choose_name in RELAXED_NAMES

    if sample_path is None:
        print("\n👉【基准自检：模板 vs 自身】")
        user_seq = template_seq
    else:
        print(f"\n👉【加载外部样本】\n文件：{sample_path}")
        try:
            user_seq = load_test_sample(sample_path)
        except Exception as e:
            print(f"❌ 加载样本失败：{e}")
            return

    score, msg, dist_val = calc_dtw_score(
        user_seq,
        template_seq,
        choose_name,
        d_bad_val
    )

    print(f"\n• 招式名称：{choose_name}")
    print(f"• 是否宽松（弯腰）：{'是' if is_relaxed else '否'}")
    print(f"• d_bad（专属标尺）：{d_bad_val:.3f}")
    print(f"• DTW 距离：{dist_val:.3f}")
    print(f"• 匹配分数：{score}")
    print(f"• 返回提示：{msg}")


if __name__ == "__main__":
    # ========== 切换测试项 ==========
    TEST_SAMPLE = None
    SELECT_TAIJI = "右下势独立"

    run_experiment(TEMPLATE_FILE, TEST_SAMPLE, SELECT_TAIJI)