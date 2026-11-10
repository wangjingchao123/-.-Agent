# -*- coding: utf-8 -*-
"""
太智伴·太极AI陪练（竞赛定稿·单文件版）
项目结构：app.py + taiji_keyframes.json
  Tab1 定式识别与安全评估：单帧余弦匹配定式 + 膝关节角度风险检测 + 健康标签提醒
  Tab2 序列动作评测：DTW对齐 + 双门槛错误片段 + 全过程抽样图 + 自动识别招式
数据：taiji_keyframes.json（24式：定式向量99维 + 归一化120帧序列 + 安全提示）
"""
import json
import cv2
import numpy as np
import gradio as gr
import mediapipe as mp
from scipy.spatial.distance import cosine, euclidean
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from collections import Counter
from pathlib import Path

# ===================== 数据文件 =====================
DATA_FILE = Path("taiji_keyframes.json")

# ===================== 共用常量 =====================
NUM_KEYPOINTS = 33
TARGET_SEQ_LEN = 120
MAX_FRAMES_READ = 800

# MediaPipe 33点索引备注：
# 11左肩 12右肩 | 23左髋 24右髋 | 25左膝 26右膝 | 27左踝 28右踝
# 29左脚跟 30右脚跟 | 31左脚尖 32右脚尖

# ===================== Tab1 常量：定式识别与安全评估 =====================
FIXED_STEP = 3              # 隔帧采样
MIN_VALLEY_DIST = 10        # 谷值间最小间隔（采样帧），find_peaks找定式停顿点
SIM_MIN_ABSOLUTE = 0.35     # 最佳相似度绝对门槛（放宽，靠特征校验把关）
SIM_GAP_MIN = 0.03          # 第一与最相似不同模板的差距门槛（防歧义；同模板多谷值=共识不算歧义）
LANDMARK_CONF_THRESHOLD = 0.5  # 帧骨架平均置信度门槛（低于则整帧舍弃）
ANGLE_WARN_THRESHOLD = 95.0 # 膝关节角度风险线（<95°判风险）
KNEE_OVER_TOL = 0.05        # 膝盖超脚尖容差
WEIGHT_FULL = 1.5           # 躯干关键点权重（定式识别看上身）
WEIGHT_LOWER = 0.6          # 下肢关键点权重（压低，防腿部噪声干扰）
LEFT_LEG = (23, 25, 27)     # 左：髋-膝-踝
RIGHT_LEG = (24, 26, 28)    # 右：髋-膝-踝

# ===================== Tab2 常量：序列动作评测 =====================
VISIBILITY_THRESHOLD = 0.5  # 低置信整帧舍弃
SKIP_STEP_DTW = 2           # 隔帧采样
DANGER_DIST = 0.45          # 得分归零线
TOP_K_KEYPOINTS = 5         # 红圈标注偏差最大关键点数
CIRCLE_RADIUS = 30          # 红圈半径（像素）

# 腿部防误报（已锁定）：只保留髋/膝/踝，删脚跟脚尖脚掌
LEG_IDS = [23, 24, 25, 26, 27, 28]
UPPER_IDS = [i for i in range(33) if i not in LEG_IDS]
THRESH_UPPER = 0.25         # 上身偏差门槛
THRESH_LEG = 0.42           # 腿部偏差门槛（更高=更宽松）
MIN_FRAME_UPPER = 5         # 上身需连续≥5帧才算真错
MIN_FRAME_LEG = 10          # 腿部需连续≥10帧才算真错
WEIGHT_UPPER = 0.75         # 上身主导打分
WEIGHT_LEG = 0.25           # 腿部压低权重
SMOOTH_WIN = 3              # 腿部平滑窗口

MATCH_DIST_THRESHOLD = 2.0  # 整体匹配门槛（超过判不匹配，不进入局部检测）
RECOG_THRESH = 0.20         # 自动识别阈值（降采样DTW粗筛）
SAMPLE_RATIO = 0.10         # 抽样比例
MAX_SAMPLE_IMAGES = 50      # 抽样图上限

# ===================== MediaPipe 全局 =====================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose_global = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ===================== 数据加载 =====================
keyframe_templates = []     # 定式列表：[{template_name, vector, health_note, ...}]
seq_library = {}            # 序列模板库：{招式名: np.array(120,99) 已归一化}


def apply_vector_weights(vec):
    """对99维定式向量应用躯干/下肢权重（保证模板与用户帧加权一致，余弦不失真）"""
    pts = np.array(vec, dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
    weights = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    for idx in [25, 26, 27, 28, 29, 30, 31, 32]:
        weights[idx] = WEIGHT_LOWER
    for idx in range(11, 25):
        weights[idx] = WEIGHT_FULL
    return (pts * weights[:, None]).flatten().tolist()


def load_data():
    """加载 taiji_keyframes.json → 定式列表(向量加权) + 序列模板库（启动时调用一次）"""
    global keyframe_templates, seq_library
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    keyframe_templates = data["templates"]
    seq_library = {}
    for t in keyframe_templates:
        t["vector"] = apply_vector_weights(t["vector"])
        if "seq" in t and t["seq"]:
            seq_library[t["template_name"]] = np.array(t["seq"], dtype=np.float32)
    print(f"[数据] 加载完成：{len(keyframe_templates)} 个定式，{len(seq_library)} 个序列模板")


# ===================== 共用：关键点提取（dict格式） =====================
def extract_pose_landmarks(frame):
    """提骨架 → dict列表 + results对象（None, None 表示未检出）"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose_global.process(rgb)
    if not res.pose_landmarks:
        return None, None
    out = []
    for idx, lm in enumerate(res.pose_landmarks.landmark):
        out.append({
            "idx": idx,
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        })
    return out, res


def filter_low_confidence(landmarks):
    """一帧内大量关键点置信度不够，直接舍弃这一帧"""
    if landmarks is None:
        return None
    low_count = sum(1 for p in landmarks if p["visibility"] < VISIBILITY_THRESHOLD)
    if low_count > NUM_KEYPOINTS // 3:
        return None
    return landmarks


def landmarks_to_array(landmarks):
    arr = np.zeros((NUM_KEYPOINTS, 3))
    for p in landmarks:
        arr[p["idx"]] = [p["x"], p["y"], p["z"]]
    return arr


def build_feature_mat(landmark_list):
    """有效关键点拼成特征矩阵 (帧数, 99)"""
    mats = []
    for lm in landmark_list:
        if lm is None:
            continue
        mats.append(landmarks_to_array(lm).flatten())
    if not mats:
        return None
    return np.array(mats, dtype=np.float32)


# ===================== Tab1：定式识别与安全评估 =====================
def normalize_and_flat_landmarks(landmark_list):
    """单帧髋部归一化 + 躯干/下肢加权 + 展平99维（输入为mediapipe关键点列表）"""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)
    origin = (pts[23] + pts[24]) / 2.0
    scale = float(np.linalg.norm(pts[12] - pts[11]))
    if scale < 1e-8:
        scale = 1.0
    normalized = (pts - origin) / scale
    weights = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    for idx in [25, 26, 27, 28, 29, 30, 31, 32]:
        weights[idx] = WEIGHT_LOWER
    for idx in range(11, 25):
        weights[idx] = WEIGHT_FULL
    return (normalized * weights[:, None]).flatten()


def calculate_angle(a, b, c):
    """三点夹角（度），用于髋-膝-踝膝关节角度"""
    ba = a - b
    bc = c - b
    cos_a = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def check_knee_risk(points_2d, hip_idx, knee_idx, ankle_idx):
    """膝关节风险检测：角度<95°或膝盖超脚尖判风险；关键点异常返回安全标记防崩溃"""
    if not (0.0 <= points_2d[knee_idx, 0] <= 1.0):
        return None, False
    a = points_2d[hip_idx]
    b = points_2d[knee_idx]
    c = points_2d[ankle_idx]
    angle = calculate_angle(a, b, c)
    over_toe = bool(b[0] > c[0] + KNEE_OVER_TOL)
    risky = bool(angle < ANGLE_WARN_THRESHOLD or over_toe)
    return angle, risky


def frame_is_reliable(landmark_list):
    """帧骨架平均置信度>=门槛才算可信（遮挡/模糊帧舍弃）"""
    if not landmark_list:
        return False
    avg_conf = float(np.mean([lm.visibility for lm in landmark_list]))
    return avg_conf >= LANDMARK_CONF_THRESHOLD


def is_gongbu(lm):
    """弓步判定：双脚左右分开>=0.15 且至少一膝弯曲<145°"""
    al, ar = lm[27], lm[28]
    if abs(al.x - ar.x) < 0.15:
        return False
    ang_l = calculate_angle(np.array([lm[23].x, lm[23].y]),
                            np.array([lm[25].x, lm[25].y]),
                            np.array([al.x, al.y]))
    ang_r = calculate_angle(np.array([lm[24].x, lm[24].y]),
                            np.array([lm[26].x, lm[26].y]),
                            np.array([ar.x, ar.y]))
    return bool((ang_l < 145) or (ang_r < 145))


def is_xubu(lm):
    """虚步判定：至少一膝近伸直>150°（重心偏一腿，另一腿虚点）"""
    ang_l = calculate_angle(np.array([lm[23].x, lm[23].y]),
                            np.array([lm[25].x, lm[25].y]),
                            np.array([lm[27].x, lm[27].y]))
    ang_r = calculate_angle(np.array([lm[24].x, lm[24].y]),
                            np.array([lm[26].x, lm[26].y]),
                            np.array([lm[28].x, lm[28].y]))
    return bool((ang_l > 150) or (ang_r > 150))


def is_pubu(lm):
    """仆步判定：髋部平均y>0.62（深蹲 lowering）"""
    hl, hr = lm[23], lm[24]
    return bool((hl.y + hr.y) / 2.0 > 0.62)


def is_ti_tui(lm):
    """提腿判定：至少一脚踝高于膝-0.08（脚离地）"""
    al, ar = lm[27], lm[28]
    kl, kr = lm[25], lm[26]
    return bool((al.y < kl.y - 0.08) or (ar.y < kr.y - 0.08))


def is_hand_near_knee(hand, knee, threshold=0.30):
    """手是否靠近膝盖（区分野马分鬃vs搂膝拗步）"""
    dist = np.sqrt((hand.x - knee.x) ** 2 + (hand.y - knee.y) ** 2)
    return bool(dist < threshold)


def is_hand_forward_to_face(hand, nose):
    """手是否前伸（z方向比鼻尖更靠前）"""
    return bool(hand.z < nose.z - 0.04)


def is_hand_side_out(hand, nose):
    """手是否侧展（x方向偏离鼻尖>0.22）"""
    return bool(abs(hand.x - nose.x) > 0.22)


def feature_verify(move_name, landmarks):
    """全套招式标志性特征校验：步型判定+几何规则过滤余弦误匹配（无规则招式放行）"""
    if not landmarks:
        return False
    lm = landmarks
    nose = lm[0]
    shl, shr = lm[11], lm[12]
    wl, wr = lm[15], lm[16]
    hl, hr = lm[23], lm[24]
    kl, kr = lm[25], lm[26]
    al, ar = lm[27], lm[28]
    ear_l, ear_r = lm[7], lm[8]
    name = move_name

    if name == "起势":
        same_h = abs(wl.y - wr.y) < 0.18
        shoulder_h = (shl.y - 0.20 < wl.y < shl.y + 0.20
                      and shr.y - 0.20 < wr.y < shr.y + 0.20)
        return same_h and shoulder_h

    if "野马分鬃" in name:
        if not is_gongbu(lm):
            return False
        dy = abs(wl.y - wr.y)
        big_gap = dy > 0.18
        one_high = min(wl.y, wr.y) < nose.y + 0.12
        one_low = max(wl.y, wr.y) > hr.y - 0.12
        no_knee_near = not (is_hand_near_knee(wl, kl) or is_hand_near_knee(wl, kr)
                            or is_hand_near_knee(wr, kl) or is_hand_near_knee(wr, kr))
        side_open = is_hand_side_out(wl, nose) or is_hand_side_out(wr, nose)
        return big_gap and one_high and one_low and no_knee_near and side_open

    if name == "白鹤亮翅":
        if not is_xubu(lm):
            return False
        one_up = (wl.y < nose.y - 0.05) or (wr.y < nose.y - 0.05)
        one_down = (wl.y > hr.y + 0.05) or (wr.y > hr.y + 0.05)
        return one_up and one_down

    if "搂膝拗步" in name:
        if not is_gongbu(lm):
            return False
        near_knee = (is_hand_near_knee(wl, kl) or is_hand_near_knee(wl, kr)
                     or is_hand_near_knee(wr, kl) or is_hand_near_knee(wr, kr))
        push_forward = (is_hand_forward_to_face(wl, nose)
                        or is_hand_forward_to_face(wr, nose))
        spread = abs(wl.x - wr.x) > 0.15
        return near_knee and push_forward and spread

    if name == "手挥琵琶":
        if not is_xubu(lm):
            return False
        close_w = abs(wl.x - wr.x) < 0.28 and abs(wl.y - wr.y) < 0.28
        chest_h = (nose.y + 0.03 < wl.y < hl.y - 0.03
                   and nose.y + 0.03 < wr.y < hr.y - 0.03)
        return close_w and chest_h

    if "倒卷肱" in name:
        near_ear = ((abs(wl.x - ear_l.x) < 0.25 and abs(wl.y - ear_l.y) < 0.25)
                    or (abs(wr.x - ear_r.x) < 0.25 and abs(wr.y - ear_r.y) < 0.25))
        forward_push = (wl.z < -0.06) or (wr.z < -0.06)
        spread_hands = abs(wl.x - wr.x) > 0.18
        return near_ear and forward_push and spread_hands

    if name == "左揽雀尾":
        if not is_gongbu(lm):
            return False
        close_wrists = abs(wl.x - wr.x) < 0.28
        mid_h = (nose.y + 0.03 < wl.y < hl.y - 0.03
                 and nose.y + 0.03 < wr.y < hr.y - 0.03)
        left_bias = (wl.x + wr.x) / 2.0 < (nose.x - 0.04)
        return close_wrists and mid_h and left_bias

    if name == "右揽雀尾":
        if not is_gongbu(lm):
            return False
        close_wrists = abs(wl.x - wr.x) < 0.28
        mid_h = (nose.y + 0.03 < wl.y < hl.y - 0.03
                 and nose.y + 0.03 < wr.y < hr.y - 0.03)
        right_bias = (wl.x + wr.x) / 2.0 > (nose.x + 0.04)
        return close_wrists and mid_h and right_bias

    if name == "单鞭（一）":
        wide_span = abs(wl.x - wr.x) > 0.30
        one_low = (wl.y > hr.y + 0.08) or (wr.y > hr.y + 0.08)
        first_dir = (wl.x + wr.x) / 2.0 > nose.x
        return wide_span and one_low and first_dir

    if name == "单鞭（二）":
        wide_span = abs(wl.x - wr.x) > 0.30
        one_low = (wl.y > hr.y + 0.08) or (wr.y > hr.y + 0.08)
        second_dir = (wl.x + wr.x) / 2.0 < nose.x
        return wide_span and one_low and second_dir

    if name == "云手":
        wide_feet = abs(al.x - ar.x) > 0.25
        cross_range = abs(wl.x - wr.x) < 0.40
        return wide_feet and cross_range

    if name == "高探马":
        if not is_xubu(lm):
            return False
        eye_h = (abs(wl.y - nose.y) < 0.15) or (abs(wr.y - nose.y) < 0.15)
        forward_extend = (wl.z < -0.06) or (wr.z < -0.06)
        return eye_h and forward_extend

    if name == "右蹬脚":
        if not is_ti_tui(lm):
            return False
        wide_hands = abs(wl.x - wr.x) > 0.25
        right_lift = ar.y < kr.y - 0.08
        return wide_hands and right_lift

    if name == "转身左蹬脚":
        if not is_ti_tui(lm):
            return False
        wide_hands = abs(wl.x - wr.x) > 0.25
        left_lift = al.y < kl.y - 0.08
        return wide_hands and left_lift

    if name == "双峰贯耳":
        if not is_gongbu(lm):
            return False
        near_nose = abs(wl.y - nose.y) < 0.18 and abs(wr.y - nose.y) < 0.18
        wide = abs(wl.x - wr.x) > 0.18
        return near_nose and wide

    if name == "左下势独立":
        deep = is_pubu(lm)
        lift_knee = is_ti_tui(lm)
        left_lift = al.y < kl.y - 0.08
        return (deep or lift_knee) and left_lift

    if name == "右下势独立":
        deep = is_pubu(lm)
        lift_knee = is_ti_tui(lm)
        right_lift = ar.y < kr.y - 0.08
        return (deep or lift_knee) and right_lift

    if "穿梭" in name:
        if not is_gongbu(lm):
            return False
        one_head = (wl.y < nose.y + 0.03) or (wr.y < nose.y + 0.03)
        one_chest = ((nose.y + 0.03 < wl.y < hl.y - 0.03)
                     or (nose.y + 0.03 < wr.y < hr.y - 0.03))
        return one_head and one_chest

    if name == "海底针":
        hip_avg_y = (hl.y + hr.y) / 2.0
        forward_bend = nose.y > hip_avg_y
        hand_down = (wl.y > kl.y) or (wr.y > kr.y)
        return forward_bend and hand_down

    if name == "闪通臂":
        if not is_gongbu(lm):
            return False
        one_over_head = (wl.y < nose.y - 0.03) or (wr.y < nose.y - 0.03)
        other_chest = ((nose.y + 0.03 < wl.y < hl.y - 0.03)
                       or (nose.y + 0.03 < wr.y < hr.y - 0.03))
        return one_over_head and other_chest

    if name == "转身搬拦捶":
        mid_line = (wl.x + wr.x) / 2.0
        punch_near_mid = (abs(wl.x - mid_line) < 0.25) or (abs(wr.x - mid_line) < 0.25)
        forward_extend = (wl.z < -0.06) or (wr.z < -0.06)
        return punch_near_mid and forward_extend

    if name == "如封似闭":
        if not is_gongbu(lm):
            return False
        mid_h = (nose.y + 0.03 < wl.y < hl.y - 0.03
                 and nose.y + 0.03 < wr.y < hr.y - 0.03)
        wider = abs(wl.x - wr.x) > 0.28
        return wider and mid_h

    if name == "十字手":
        close_wrists = abs(wl.x - wr.x) < 0.28
        close_ankles = abs(al.x - ar.x) < 0.28
        return close_wrists and close_ankles

    if name == "收势":
        body_up = (hl.y < 0.55) and (hr.y < 0.55)
        hands_low = (wl.y > hr.y + 0.05) and (wr.y > hr.y + 0.05)
        return body_up and hands_low

    return True


def run_recognize(video_file, health_label):
    """Tab1 主入口：视频 → 隔帧采样 → 谷值检测找定式停顿帧 → 各谷值匹配最佳模板投票
    → 双重置信保护（绝对门槛+歧义门槛）→ 膝关节安全检测 → 最佳帧画骨架"""
    if video_file is None:
        return None, "请上传视频文件。", 0.0, "未知"
    if not keyframe_templates:
        return None, "定式模板未加载。", 0.0, "未知"

    frames_data = []
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        smooth_landmarks=True, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_file)
    frame_idx = 0
    while cap.isOpened() and frame_idx < MAX_FRAMES_READ:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FIXED_STEP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                if frame_is_reliable(lm):
                    vec = normalize_and_flat_landmarks(lm)
                    frames_data.append({
                        "raw_frame": frame.copy(),
                        "pose_lm": res.pose_landmarks,
                        "landmarks": lm,
                        "vec": vec,
                    })
        frame_idx += 1
    cap.release()
    pose.close()

    if len(frames_data) < 5:
        return None, "视频太短，或者没有检测到有效的人体骨骼。", 0.0, "未知"

    # 谷值检测：帧间变化最小的点 = 定式停顿帧
    vecs = np.array([item["vec"] for item in frames_data])
    diffs = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
    valleys, _ = find_peaks(-diffs, distance=MIN_VALLEY_DIST)
    if len(valleys) == 0:
        valleys = np.array([int(np.argmin(diffs))])

    # 每个谷值帧与24式匹配，取各自最佳模板入候选池
    candidate_pool = []
    for v_idx in valleys:
        target_vec = vecs[v_idx]
        score_dict = {}
        best_sim = -1.0
        best_tmpl = None
        for tmpl in keyframe_templates:
            sim = 1.0 - cosine(target_vec, np.array(tmpl["vector"], dtype=np.float32))
            score_dict[tmpl["template_name"]] = float(sim)
            if sim > best_sim:
                best_sim = sim
                best_tmpl = tmpl
        # 每一轮比对时打印
        print("=====本轮比对信息=====")
        print("抓到的极点帧索引：", v_idx)
        print("当前帧相似度字典：", score_dict)
        sorted_score = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        print("TOP1：", sorted_score[0])
        print("TOP2：", sorted_score[1])
        print("校验feature_verify结果：", feature_verify(sorted_score[0][0], frames_data[v_idx]["landmarks"]))
        if best_tmpl is not None:
            candidate_pool.append({
                "template": best_tmpl,
                "similarity": best_sim,
                "frame_item": frames_data[v_idx],
            })

    # 兜底：全帧最佳匹配帧也入候选池（防谷值检测漏掉最佳姿态帧）
    gb_sim, gb_idx, gb_tmpl = -1.0, -1, None
    for fi, fvec in enumerate(vecs):
        for tmpl in keyframe_templates:
            sim = 1.0 - cosine(fvec, np.array(tmpl["vector"], dtype=np.float32))
            if sim > gb_sim:
                gb_sim, gb_idx, gb_tmpl = sim, fi, tmpl
    if gb_idx >= 0 and gb_tmpl is not None:
        already = any(c["frame_item"] is frames_data[gb_idx] for c in candidate_pool)
        if not already:
            candidate_pool.append({
                "template": gb_tmpl,
                "similarity": gb_sim,
                "frame_item": frames_data[gb_idx],
            })

    if not candidate_pool:
        return None, "未能匹配到任何已知招式。", 0.0, "未知"

    # ===== 全套招式标志性特征校验：过滤余弦误匹配 =====
    filtered_pool = []
    for cand in candidate_pool:
        name = cand["template"]["template_name"]
        if feature_verify(name, cand["frame_item"]["landmarks"]):
            filtered_pool.append(cand)
    if not filtered_pool:
        return None, "初步匹配成功，但动作形态和该招式标志性特征不符，无法确认，请重新录制。", 0.0, "未知"
    candidate_pool = filtered_pool

    candidate_pool.sort(key=lambda x: x["similarity"], reverse=True)
    vote_counter = Counter()
    for item in candidate_pool:
        vote_counter[item["template"]["template_name"]] += 1

    first = candidate_pool[0]
    best_score = first["similarity"]
    best_template = first["template"]
    best_item = first["frame_item"]

    # 歧义判断：只与不同模板的候选比较（同模板多谷值命中=共识，非歧义）
    gap = 1.0
    for item in candidate_pool[1:]:
        if item["template"]["template_name"] != best_template["template_name"]:
            gap = best_score - item["similarity"]
            break

    # 双重置信保护
    if best_score < SIM_MIN_ABSOLUTE:
        return None, "匹配分数不足，无法确认招式，请录制完整标准动作。", 0.0, "未知"
    if gap < SIM_GAP_MIN:
        return None, "识别存在歧义，两个招式相似度接近，无法确定，请重新录制。", 0.0, "未知"

    score = float(np.clip(best_score * 100, 0, 100))
    move_name = best_template["template_name"]
    health_note = best_template.get("health_note", "动作整体姿态正常。")

    # 膝关节安全检测
    pts_2d = np.array([[lm.x, lm.y] for lm in best_item["landmarks"]], dtype=np.float32)
    l_angle, l_risky = check_knee_risk(pts_2d, *LEFT_LEG)
    r_angle, r_risky = check_knee_risk(pts_2d, *RIGHT_LEG)
    risk_parts = []
    if l_risky and l_angle is not None:
        risk_parts.append(f"左膝：{int(l_angle)}°，偏低或超过脚尖")
    if r_risky and r_angle is not None:
        risk_parts.append(f"右膝：{int(r_angle)}°，偏低或超过脚尖")
    risk_text = "🚨 安全提醒：" + "；".join(risk_parts) if risk_parts else "✅ 当前关节角度处于安全范围。"

    comment = (
        f"【识别招式】{move_name}\n"
        f"【匹配得分】{score:.1f} 分\n"
        f"【动作提示】{health_note}\n"
        f"【安全评估】{risk_text}"
    )
    if health_label == "高血压":
        comment += "\n\n⚠️ 练习时保持平稳呼吸，不要憋气、猛然发力。"
    elif health_label == "膝关节炎":
        comment += "\n\n⚠️ 如果膝盖出现酸痛，降低下蹲幅度，以无痛为首要原则。"

    # 最佳定式帧画骨架（mediapipe官方绘图工具）
    best_image = None
    try:
        draw_img = best_item["raw_frame"].copy()
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        mp_draw.draw_landmarks(draw_img, best_item["pose_lm"], mp_pose.POSE_CONNECTIONS)
        best_image = draw_img
    except Exception:
        best_image = cv2.cvtColor(best_item["raw_frame"], cv2.COLOR_BGR2RGB)

    return best_image, comment, score, move_name


# ===================== Tab2：序列动作评测 =====================
def normalize_sequence(seq, target_len=TARGET_SEQ_LEN):
    """等长重采样（n帧 → 120帧，逐维线性插值）"""
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


def normalize_hip_center(seq):
    """逐帧髋中心+肩宽归一化（输入/输出均为 (帧数, 99)）"""
    n = seq.shape[0]
    out = seq.reshape(n, NUM_KEYPOINTS, 3).copy()
    for f in range(n):
        center = (out[f, 23] + out[f, 24]) / 2.0
        shoulder = float(np.linalg.norm(out[f, 11] - out[f, 12]))
        if shoulder < 1e-6:
            shoulder = 1e-6
        out[f] = (out[f] - center) / shoulder
    return out.reshape(n, -1)


def smooth_leg_sequence(seq):
    """仅对腿部关键点做滑动平均，上身不动"""
    out = seq.copy()
    n_frame = seq.shape[0]
    reshaped = seq.reshape(n_frame, NUM_KEYPOINTS, 3)
    out_r = out.reshape(n_frame, NUM_KEYPOINTS, 3)
    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    half = SMOOTH_WIN // 2
    for pid in LEG_IDS:
        for d in range(3):
            vals = reshaped[:, pid, d]
            padded = np.pad(vals, half, mode="edge")
            out_r[:, pid, d] = np.convolve(padded, kernel, mode="valid")
    return out


def quick_compare(a, b):
    """降采样轻量DTW粗筛（自动识别用，均值绝对偏差）"""
    step_a = max(1, a.shape[0] // 24)
    step_b = max(1, b.shape[0] // 24)
    a_s = a[::step_a]
    b_s = b[::step_b]
    n, m = len(a_s), len(b_s)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = float(np.mean(np.abs(a_s[i - 1] - b_s[j - 1])))
            d[i, j] = c + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])
    return d[n, m] / (n + m)


def auto_select_template(user_seq):
    """自动识别：降采样轻量DTW粗筛 → 最小距离招式（>RECOG_THRESH判失败）"""
    best_name, best_dist = None, float("inf")
    for name, tpl in seq_library.items():
        d = quick_compare(user_seq, tpl)
        if d < best_dist:
            best_dist, best_name = d, name
    if best_name is None or best_dist > RECOG_THRESH:
        return None, best_dist
    return best_name, best_dist


def dtw_with_path(seq_a, seq_b):
    """带回溯DTW，分体加权：上身0.75 + 腿部0.25"""
    na, nb = seq_a.shape[0], seq_b.shape[0]
    upper_cols = [c for kp in UPPER_IDS for c in (kp * 3, kp * 3 + 1, kp * 3 + 2)]
    leg_cols = [c for kp in LEG_IDS for c in (kp * 3, kp * 3 + 1, kp * 3 + 2)]
    ua, la = seq_a[:, upper_cols], seq_a[:, leg_cols]
    ub, lb = seq_b[:, upper_cols], seq_b[:, leg_cols]
    cost_mat = np.zeros((na, nb), dtype=np.float64)
    dmat = np.full((na + 1, nb + 1), np.inf)
    dmat[0, 0] = 0.0
    for i in range(na):
        for j in range(nb):
            d_up = euclidean(ua[i], ub[j])
            d_leg = euclidean(la[i], lb[j])
            c = WEIGHT_UPPER * d_up + WEIGHT_LEG * d_leg
            cost_mat[i, j] = c
            dmat[i + 1, j + 1] = c + min(dmat[i, j], dmat[i + 1, j], dmat[i, j + 1])
    path = []
    i, j = na, nb
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        if dmat[i - 1, j - 1] <= dmat[i - 1, j] and dmat[i - 1, j - 1] <= dmat[i, j - 1]:
            i, j = i - 1, j - 1
        elif dmat[i - 1, j] <= dmat[i, j - 1]:
            i, j = i - 1, j
        else:
            i, j = i, j - 1
    path.reverse()
    pair_costs = np.array([cost_mat[a, b] for a, b in path], dtype=np.float64)
    pair_costs_upper = np.array([euclidean(ua[a], ub[b]) for a, b in path], dtype=np.float64)
    pair_costs_leg = np.array([euclidean(la[a], lb[b]) for a, b in path], dtype=np.float64)
    kp_costs = []
    for a, b in path:
        fa = seq_a[a].reshape(NUM_KEYPOINTS, 3)
        fb = seq_b[b].reshape(NUM_KEYPOINTS, 3)
        kp_costs.append(np.sqrt(np.sum((fa - fb) ** 2, axis=1)))
    kp_costs = np.array(kp_costs, dtype=np.float64)
    dist = float(dmat[na, nb]) / (na + nb)
    return dist, path, pair_costs, pair_costs_upper, pair_costs_leg, kp_costs


def score_from_distance(dist):
    """评分公式：100*(1-dist/DANGER_DIST)，超过DANGER_DIST归零"""
    dist = max(0.0, float(dist))
    if dist >= DANGER_DIST:
        return 0.0
    return float(np.clip(100.0 * (1.0 - dist / DANGER_DIST), 0.0, 100.0))


def judge_level(score):
    """四档：优秀≥80 / 尚可60~79 / 重点调整40~59 / 需重练<40"""
    if score >= 80:
        return "优秀", "走向正确，连贯顺滑，仅存在细碎小差异，继续保持。"
    elif score >= 60:
        return "尚可", "主体动作正确，局部小幅偏移，无断停，注意保持连贯。"
    elif score >= 40:
        return "重点调整", "方向或连贯出现明显问题，建议回看错误片段对照修正。"
    else:
        return "需重练", "走向严重偏离，频繁卡顿、断续，建议从基础动作重新练习。"


def _close_segment(start, end, is_leg, path, valid_count, valid_frame_nums,
                   pair_costs_upper, pair_costs_leg, kp_costs, seg_list):
    """结束一段连续偏移区间，满足最短帧数才入库"""
    span = end - start + 1
    need_len = MIN_FRAME_LEG if is_leg else MIN_FRAME_UPPER
    if span < need_len:
        return
    dev_seq = pair_costs_leg if is_leg else pair_costs_upper
    total = float(np.sum(dev_seq[start:end + 1]))
    mid_k = (start + end) // 2
    ua = path[mid_k][0]
    eff_idx = min(max(int(round(ua / (TARGET_SEQ_LEN - 1) * (valid_count - 1))), 0), valid_count - 1)
    region_ids = LEG_IDS if is_leg else UPPER_IDS
    ranked = sorted(region_ids, key=lambda k: kp_costs[mid_k][k], reverse=True)
    top_kp = ranked[:TOP_K_KEYPOINTS]
    ua_s, ua_e = path[start][0], path[end][0]
    eff_s = min(max(int(round(ua_s / (TARGET_SEQ_LEN - 1) * (valid_count - 1))), 0), valid_count - 1)
    eff_e = min(max(int(round(ua_e / (TARGET_SEQ_LEN - 1) * (valid_count - 1))), 0), valid_count - 1)
    seg_list.append({
        "orig_start": valid_frame_nums[eff_s],
        "orig_end": valid_frame_nums[eff_e],
        "is_leg": is_leg,
        "total_dev": total,
        "mean_dev": total / span,
        "mid_eff_idx": eff_idx,
        "orig_frame": valid_frame_nums[eff_idx],
        "top_kp": top_kp,
        "top_kp_dev": [float(kp_costs[mid_k][k]) for k in top_kp],
    })


def locate_error_segments(path, valid_count, valid_frame_nums,
                          pair_costs_upper, pair_costs_leg, kp_costs):
    """双门槛片段检测：上身>0.25连续≥5帧 / 腿部>0.42连续≥10帧才算真错"""
    n = len(pair_costs_upper)
    if n == 0 or valid_count < 2:
        return []
    seg_list = []
    current_start = None
    current_is_leg = False
    for f in range(n):
        hit_upper = pair_costs_upper[f] > THRESH_UPPER
        hit_leg = pair_costs_leg[f] > THRESH_LEG
        if hit_leg:
            if current_start is None:
                current_start = f
                current_is_leg = True
        elif hit_upper:
            if current_start is None:
                current_start = f
                current_is_leg = False
        else:
            if current_start is not None:
                _close_segment(current_start, f - 1, current_is_leg,
                               path, valid_count, valid_frame_nums,
                               pair_costs_upper, pair_costs_leg, kp_costs, seg_list)
                current_start = None
                current_is_leg = False
    if current_start is not None:
        _close_segment(current_start, n - 1, current_is_leg,
                       path, valid_count, valid_frame_nums,
                       pair_costs_upper, pair_costs_leg, kp_costs, seg_list)
    seg_list.sort(key=lambda x: x["mean_dev"], reverse=True)
    return seg_list[:3]


def sample_indices_by_percent(n_frame, percent=SAMPLE_RATIO, max_images=MAX_SAMPLE_IMAGES):
    """均匀抽样：首尾必选，封顶max_images"""
    if n_frame <= 0:
        return []
    k = max(1, int(np.ceil(n_frame * percent)))
    k = min(k, max_images)
    if k >= n_frame:
        return list(range(n_frame))
    indices = np.linspace(0, n_frame - 1, num=k, endpoint=True)
    return sorted(set(int(round(i)) for i in indices))


def extract_sample_frames(video_path, sample_eff_indices, valid_frame_nums,
                          all_landmarks, top_segs):
    """回原视频帧画骨架：绿=正常，红圈=错误片段偏差突出部位"""
    images = []
    if not sample_eff_indices:
        return images
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return images
    for eff_idx in sample_eff_indices:
        if eff_idx >= len(all_landmarks):
            continue
        orig_f = valid_frame_nums[eff_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_f)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        lm = all_landmarks[eff_idx]
        for (a, b) in mp_pose.POSE_CONNECTIONS:
            pa, pb = lm[a], lm[b]
            cv2.line(frame, (int(pa["x"] * w), int(pa["y"] * h)),
                     (int(pb["x"] * w), int(pb["y"] * h)), (0, 255, 0), 2)
        for p in lm:
            cv2.circle(frame, (int(p["x"] * w), int(p["y"] * h)), 4, (0, 255, 0), -1)
        hit_seg = None
        for seg in top_segs:
            if seg["orig_start"] <= orig_f <= seg["orig_end"]:
                hit_seg = seg
                break
        label = f"frame:{orig_f}"
        if hit_seg is not None:
            for kp_idx in hit_seg["top_kp"]:
                p = lm[kp_idx]
                cx, cy = int(p["x"] * w), int(p["y"] * h)
                cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, (0, 0, 255), 4)
                cv2.putText(frame, str(kp_idx), (cx + CIRCLE_RADIUS + 5, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            kind = "Leg" if hit_seg["is_leg"] else "Upper"
            label += f" | ERROR[{kind}] dev:{hit_seg['mean_dev']:.3f}"
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)
        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return images


def extract_seq_from_video(video_path):
    """视频 → 提骨架 → 低置信过滤 → 髋部归一化 → 重采样120帧（未平滑）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    all_landmarks = []
    valid_frame_nums = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < MAX_FRAMES_READ:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % SKIP_STEP_DTW != 0:
            frame_idx += 1
            continue
        lm, _ = extract_pose_landmarks(frame)
        lm = filter_low_confidence(lm)
        if lm is not None:
            all_landmarks.append(lm)
            valid_frame_nums.append(frame_idx)
        frame_idx += 1
    cap.release()
    if len(all_landmarks) < 20:
        return None
    feat_mat = build_feature_mat(all_landmarks)
    if feat_mat is None:
        return None
    feat_mat = normalize_hip_center(feat_mat)
    norm_seq = normalize_sequence(feat_mat, target_len=TARGET_SEQ_LEN)
    return norm_seq, all_landmarks, valid_frame_nums


def analyze(action_name, video_path):
    """Tab2 主入口：选招式/自动识别 → 提骨架 → 平滑 → DTW → 匹配门槛 → 评分 → 片段 → 抽样图"""
    if not video_path:
        return 0.0, "未上传", "请上传练习视频。", []
    result = extract_seq_from_video(video_path)
    if result is None:
        return 0.0, "识别失败", (
            "识别失败：检测有效关键点数量不足。\n"
            "常见原因：画面遮挡、背光、距离太远、多人同时入镜。\n"
            "请重新录制，保证画面内仅有练习者一人。"
        ), []
    norm_user_raw, all_landmarks, valid_frame_nums = result
    valid_count = len(all_landmarks)

    matched_by_auto = False
    match_dist = 0.0
    target_name = action_name
    if action_name == "自动识别":
        target_name, match_dist = auto_select_template(norm_user_raw)
        if target_name is None:
            return 0.0, "匹配失败", (
                f"自动匹配失败（最小距离 {match_dist:.4f}，阈值 {RECOG_THRESH}），"
                f"没有找到对应招式，请手动选择。"
            ), []
        matched_by_auto = True

    template = seq_library.get(target_name)
    if template is None:
        return 0.0, "模板缺失", f"招式 [{target_name}] 模板缺失。", []

    norm_user = smooth_leg_sequence(norm_user_raw)
    dist, path, pair_costs, pc_upper, pc_leg, kp_costs = dtw_with_path(norm_user, template)

    if dist > MATCH_DIST_THRESHOLD:
        report = (
            f"【招式】：{target_name}\n"
            f"【整体差异距离】：{dist:.4f}（匹配门槛 {MATCH_DIST_THRESHOLD}）\n\n"
            f"【判定】：动作与标准模板不匹配\n"
            f"整体动作形态/节奏与「{target_name}」对不上，不进行局部错误检测。\n"
            f"可能原因：选错招式、视频内容不是该招式、动作严重偏离。\n"
            f"建议：确认招式选择，或重新录制。"
        )
        return 0.0, "不匹配", report, []

    score = score_from_distance(dist)
    level, tip = judge_level(score)
    top_segs = locate_error_segments(path, valid_count, valid_frame_nums,
                                     pc_upper, pc_leg, kp_costs)
    sample_idx = sample_indices_by_percent(valid_count)
    gallery_images = extract_sample_frames(video_path, sample_idx, valid_frame_nums,
                                           all_landmarks, top_segs)

    if top_segs:
        seg_lines = []
        for rank, seg in enumerate(top_segs, 1):
            kind = "下肢（腿部）" if seg["is_leg"] else "上身/躯干"
            seg_lines.append(
                f"  第{rank}段：第 {seg['orig_start']} ~ {seg['orig_end']} 帧"
                f"（{kind}，平均偏差 {seg['mean_dev']:.4f}，代表帧 {seg['orig_frame']}）"
            )
        seg_text = "\n".join(seg_lines)
    else:
        seg_text = "  未检出重点错误片段（个别点位短暂偏移属于识别噪声，不计为练习错误）"

    auto_line = ""
    if matched_by_auto:
        auto_line = f"【自动识别匹配】：{target_name}（匹配距离 {match_dist:.4f}）\n"
    report = (
        f"{auto_line}"
        f"【招式】：{target_name}\n"
        f"【综合评分】：{score:.1f} / 100\n"
        f"【判定等级】：{level}\n"
        f"【整体差异距离】：{dist:.4f}\n\n"
        f"【判定说明与修正建议】\n{tip}\n\n"
        f"【重点错误片段（前三大）】\n{seg_text}\n\n"
        f"【标记样图】：下方为全过程均匀抽样（约{int(SAMPLE_RATIO * 100)}%，最多{MAX_SAMPLE_IMAGES}张），"
        f"绿色=正常关节骨架，红圈=错误片段内偏差突出部位\n"
        f"（注：腿部点位需连续{MIN_FRAME_LEG}帧偏离才标记，上身需连续{MIN_FRAME_UPPER}帧，瞬时识别漂移已过滤）"
    )
    return round(score, 1), level, report, gallery_images


# ===================== Gradio 界面 =====================
def build_app():
    action_opts = ["自动识别"] + list(seq_library.keys())
    with gr.Blocks(title="太智伴·太极AI陪练") as demo:
        gr.Markdown("# 太智伴 · 老年太极AI陪练")
        gr.Markdown("> 定式识别与安全评估 + 序列动作评测，双模式辅助练习")
        with gr.Tabs():
            # —— Tab1：定式识别与安全评估 ——
            with gr.TabItem("定式识别与安全评估"):
                gr.Markdown("上传练习视频，系统提取最接近标准定式的关键帧，识别招式，检测膝关节风险。")
                with gr.Row():
                    with gr.Column():
                        vid1 = gr.Video(label="上传练习视频")
                        health_in = gr.Dropdown(
                            choices=["无特殊情况", "高血压", "膝关节炎"],
                            value="无特殊情况", label="使用者健康情况")
                        btn1 = gr.Button("开始识别打分", variant="primary")
                    with gr.Column():
                        preview_out = gr.Image(label="匹配到的最佳定式（骨架）")
                        score1 = gr.Number(label="匹配得分（0-100）")
                        action1 = gr.Textbox(label="识别招式名称")
                        comment1 = gr.Textbox(label="教练评语与安全评估", lines=7)
                btn1.click(fn=run_recognize, inputs=[vid1, health_in],
                           outputs=[preview_out, comment1, score1, action1])
            # —— Tab2：序列动作评测 ——
            with gr.TabItem("序列动作评测"):
                gr.Markdown("选择招式（或自动识别）→ 上传练习视频 → DTW比对 → 得分等级 + 错误片段 + 抽样图。")
                sel_action = gr.Dropdown(choices=action_opts, value="自动识别",
                                         label="选择目标招式模板")
                vid2 = gr.Video(label="上传练习视频")
                btn2 = gr.Button("开始评测", variant="primary")
                score2 = gr.Number(label="得分", precision=1)
                level2 = gr.Textbox(label="判定等级")
                report2 = gr.Textbox(label="完整报告", lines=18)
                gallery2 = gr.Gallery(
                    label="全过程抽样样图（绿色=正常骨架，红圈=错误片段偏差突出部位）",
                    columns=3, height=650)
                btn2.click(fn=analyze, inputs=[sel_action, vid2],
                           outputs=[score2, level2, report2, gallery2])
    return demo


if __name__ == "__main__":
    load_data()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)