# -*- coding: utf-8 -*-
"""
生成定式帧模板库 taiji_keyframes.json
- 输入：saved_templates_final.json（24式，每式147帧×33点）
- 每式提取"动作极点帧"（跳过起始静止段，找变化最小的定式帧）
- 髋部归一化 + 展平为99维向量
- 配套二十四式安全提示
- 输出：taiji_keyframes.json
"""
import json
import numpy as np
from scipy.interpolate import interp1d


def frames_to_raw_seq(frame_list):
    """帧列表(dict关键点) → (帧数, 99) 原始坐标数组"""
    arr = np.array(
        [[[p["x"], p["y"], p["z"]] for p in fr] for fr in frame_list],
        dtype=np.float32
    )
    return arr.reshape(len(frame_list), -1)


def normalize_hip_center_seq(seq):
    """逐帧髋部归一化：髋中心为原点、肩宽为尺度 (帧数,99) → (帧数,99)"""
    n = seq.shape[0]
    out = seq.reshape(n, 33, 3).copy()
    for f in range(n):
        center = (out[f, 23] + out[f, 24]) / 2.0
        shoulder = float(np.linalg.norm(out[f, 11] - out[f, 12]))
        if shoulder < 1e-6:
            shoulder = 1e-6
        out[f] = (out[f] - center) / shoulder
    return out.reshape(n, -1)


def resample_seq(seq, target_len=120):
    """线性插值重采样到 target_len 帧"""
    n, feat = seq.shape
    if n == target_len:
        return seq
    old_x = np.linspace(0, 1, n)
    new_x = np.linspace(0, 1, target_len)
    out = np.empty((target_len, feat))
    for col in range(feat):
        f = interp1d(old_x, seq[:, col], kind="linear", fill_value="extrapolate")
        out[:, col] = f(new_x)
    return out.astype(np.float32)


def normalize_and_flat(landmark_list):
    """髋部归一化：以髋中心为原点、肩宽为尺度，展平为99维"""
    pts = np.array(
        [[p["x"], p["y"], p["z"]] for p in landmark_list],
        dtype=np.float32
    )
    hip_l = pts[23]
    hip_r = pts[24]
    origin = (hip_l + hip_r) / 2.0
    shoulder_l = pts[11]
    shoulder_r = pts[12]
    scale = np.linalg.norm(shoulder_r - shoulder_l)
    if scale < 1e-8:
        scale = 1.0
    normalized = (pts - origin) / scale
    return normalized.flatten().tolist()


def pick_stable_keyframe(frame_list):
    """
    竞赛版：跳过前面静止起始帧，只在动作区内找变化最小（极点/定式帧）
    返回 (选中帧索引, 归一化99维向量)
    """
    total = len(frame_list)
    if total <= 1:
        return 0, normalize_and_flat(frame_list[0])

    # 帧数少直接取中间，避免掐头
    if total <= 20:
        mid = total // 2
        return mid, normalize_and_flat(frame_list[mid])

    # 安全起始：跳过前15帧，排除启动前静止
    skip_start = 15
    vecs = []
    for fr in frame_list:
        vecs.append(np.array(normalize_and_flat(fr)))

    deltas = []
    for i in range(1, total):
        deltas.append(np.linalg.norm(vecs[i] - vecs[i - 1]))

    # 在跳过起始段的范围内找最小变化
    search_start = skip_start
    if search_start >= len(deltas):
        search_start = 0

    sub_deltas = deltas[search_start:]
    local_min_idx = int(np.argmin(sub_deltas))
    chosen_idx = search_start + local_min_idx + 1

    if chosen_idx >= total:
        chosen_idx = total - 1

    return int(chosen_idx), vecs[chosen_idx].tolist()


# 完整二十四式配套安全提示（键名与模板招式名一一对应）
health_map = {
    "起势": "肩膀放松，不要耸肩，下蹲幅度轻柔，重心平稳。",
    "左右野马分鬃": "弓步不要压太低，如果膝盖不适就减小前屈幅度。",
    "白鹤亮翅": "保持上体中正，不要过度后仰，保护腰背。",
    "左右搂膝拗步": "重心平稳前移，膝盖尽量不要超过脚尖。",
    "手挥琵琶": "两臂轻展，不要僵紧，保持呼吸均匀舒缓。",
    "倒卷肱（标准）": "转身缓慢，不要猛扭腰，保护腰部和肩颈。",
    "左揽雀尾": "后坐充分，不要仓促前冲，稳住下肢重心。",
    "右揽雀尾": "注意重心转换，避免单腿长时间过度承重。",
    "单鞭（一）": "支撑腿微屈，不要锁死膝盖，保持身体稳定。",
    "云手": "重心左右平移要平缓，防止身体左右摇晃。",
    "单鞭（二）": "支撑腿微屈，不要锁死膝盖，保持身体稳定。",
    "高探马": "探身适度，不要猛地前倾，保护腰背。",
    "右蹬脚": "蹬脚不必追求高度，优先保证身体站稳。",
    "双峰贯耳": "出拳放松，不要憋气，保护肩肘。",
    "转身左蹬脚": "转身和蹬脚分开完成，不要急着一气猛做。",
    "左下势独立": "下蹲量力而行，不要强行压很低，起身放慢。",
    "右下势独立": "起身缓慢，避免猛地站起造成头晕。",
    "左右穿梭": "弓步幅度适中，控制膝盖压力，保护膝部。",
    "海底针": "俯身不要过猛，有高血压就减少大幅度俯身。",
    "闪通臂": "前推时保持下肢稳定，不要向前栽倒。",
    "转身搬拦捶": "转体缓慢，不要猛扭躯干，保护腰。",
    "如封似闭": "后坐充分，重心不要一下子向前冲出去。",
    "十字手": "收合轻柔，不要硬夹肩膀，放松肩背。",
    "收势": "慢慢落手，平稳站定，调匀呼吸，放松全身。",
}


def main():
    src_file = "saved_templates_final.json"
    out_file = "taiji_keyframes.json"

    with open(src_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    templates_out = []

    for move_name, entry in raw_data.items():
        frames = entry["frames"] if isinstance(entry, dict) else entry
        selected_index, selected_vector = pick_stable_keyframe(frames)
        tip = health_map.get(
            move_name,
            "保持动作舒缓，根据自身身体情况调整幅度。"
        )
        # 归一化120帧序列（供序列DTW评测使用，已髋部归一化，加载时无需再归一化）
        raw_seq = frames_to_raw_seq(frames)
        norm_seq = normalize_hip_center_seq(raw_seq)
        norm_seq = resample_seq(norm_seq, 120)
        templates_out.append({
            "template_name": move_name,
            "vector": selected_vector,
            "seq": np.round(norm_seq, 4).tolist(),
            "health_note": tip,
            "source_frame_index": selected_index
        })

    package = {
        "meta": {
            "version": "competition-v1.0",
            "note": "竞赛｜太极二十四式，自动提取动作极点（变化最小定式帧）"
        },
        "templates": templates_out
    }

    # 紧凑输出：含120帧序列数据，indent会膨胀到十几MB
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False, separators=(",", ":"))

    import os
    size_kb = os.path.getsize(out_file) / 1024
    print(f"✅ 生成完成，一共 {len(templates_out)} 个招式")
    print(f"✅ 文件输出：{out_file}（{size_kb:.0f} KB）")
    # 抽查
    for t in templates_out[:3]:
        seq_shape = f"{len(t['seq'])}x{len(t['seq'][0])}" if t.get("seq") else "无"
        print(f"  {t['template_name']}: 定式帧#{t['source_frame_index']}  向量{len(t['vector'])}维  序列{seq_shape}  提示:{t['health_note'][:12]}...")


if __name__ == "__main__":
    main()