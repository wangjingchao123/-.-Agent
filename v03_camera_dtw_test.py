"""
【已核验｜竞赛 v03 最终版】
校验点清单
  ✅ ① adaptive_score：k=2.2 × 招式专属 d_bad（无自引用）
  ✅ ② JSON：兼容 dict{"frames","d_bad"} / list
  ✅ ③ Gradio：sources=["webcam"]，适配新版
  ✅ ④ DTW：髋部中心+肩宽归一化（消除体型/距离差异），累积距离与 d_bad 同口径
  ✅ ⑤ 触发阈值：缓冲 ≥ SCORE_TRIGGER_MIN(30) 即评分
  ✅ 队列积压：超时直接丢弃旧帧
  ✅ 隔帧 DTW，降低计算压力
  ✅ MediaPipe 轻量模型 + 降分辨率
  ✅ 关键点平滑，抑制分数抖动
  ✅ 单人锁定、遮挡判定、正反面、只读模板全部保留
"""
import json
import math
import time
import collections
import cv2
import numpy as np
from dtw import dtw
import mediapipe as mp
import gradio as gr

# ========== 竞赛固定配置 ==========
TEMPLATE_FILE = "saved_templates_final.json"
K = 2.2
BAD_RATIO_THRESHOLD = 0.40
CONTINUOUS_BAD_MAX = 8
WINDOW_SIZE = 120
MIN_TEMPLATE_FRAMES = 10
SCORE_TRIGGER_MIN = 30           # 缓冲达到此帧数即触发评分
DETECTION_CONF = 0.5
TRACKING_CONF = 0.5
VIS_VALID_THRESH = 0.40          # 归一化所需关键点最低置信度
SKIP_DTW_EVERY = 5               # 每 5 帧才跑一次 DTW
RESIZE_W, RESIZE_H = 640, 480    # 降分辨率减轻推理负载
STALE_FRAME_GAP = 0.35           # 积压判定：超过此间隔视为旧帧丢弃
# =================================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=DETECTION_CONF,
    min_tracking_confidence=TRACKING_CONF
)
mp_draw = mp.solutions.drawing_utils

# 关键点平滑缓冲区
landmark_smooth_buf = collections.deque(maxlen=5)


def load_template_library(filepath: str):
    """兼容两种模板存储：dict{frames,d_bad} / list（问题②）"""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    library = {}
    for name, item in raw_data.items():
        if isinstance(item, dict):
            if "frames" not in item:
                print(f"[跳过] {name}：字典缺少 frames")
                continue
            seq = item["frames"]
            d_bad = float(item.get("d_bad", 0.0))
        elif isinstance(item, list):
            seq = item
            d_bad = 0.0
        else:
            print(f"[跳过] {name}：未知数据类型")
            continue
        if len(seq) < MIN_TEMPLATE_FRAMES:
            print(f"[跳过] {name}：帧数不足")
            continue
        library[name] = {"frames": seq, "d_bad": d_bad}
    return library


def extract_33_keypoints(results):
    if not results.pose_landmarks:
        return None
    points_list = []
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        points_list.append({
            "idx": idx,
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility
        })
    return points_list


def smooth_points(points):
    """时序平滑，抑制画面、分数抖动"""
    if points is None:
        landmark_smooth_buf.clear()
        return None
    landmark_smooth_buf.append(points)
    if len(landmark_smooth_buf) < 3:
        return points
    avg_list = []
    for i in range(len(points)):
        xs = [frame[i]["x"] for frame in landmark_smooth_buf]
        ys = [frame[i]["y"] for frame in landmark_smooth_buf]
        vs = [frame[i]["visibility"] for frame in landmark_smooth_buf]
        avg_list.append({
            "idx": i,
            "x": float(np.mean(xs)),
            "y": float(np.mean(ys)),
            "z": points[i]["z"],
            "visibility": float(np.mean(vs))
        })
    return avg_list


def calculate_bad_ratio(frame_points):
    if frame_points is None:
        return 1.0
    low_vis = sum(1 for p in frame_points if p["visibility"] < 0.5)
    return low_vis / len(frame_points)


# ========== 归一化：髋部中心+肩宽（与 d_bad 同口径，问题④） ==========
def normalize_frame_33(frame_points):
    """33点帧归一化：髋部中心为原点，肩宽为单位长度
    消除体型/离镜头远近差异；髋部不可用返回 None
    """
    if frame_points is None or len(frame_points) < 33:
        return None
    lhip, rhip = frame_points[23], frame_points[24]
    if lhip["visibility"] < VIS_VALID_THRESH or rhip["visibility"] < VIS_VALID_THRESH:
        return None
    cx = (lhip["x"] + rhip["x"]) / 2.0
    cy = (lhip["y"] + rhip["y"]) / 2.0
    lsh, rsh = frame_points[11], frame_points[12]
    if lsh["visibility"] >= VIS_VALID_THRESH and rsh["visibility"] >= VIS_VALID_THRESH:
        scale = math.hypot(lsh["x"] - rsh["x"], lsh["y"] - rsh["y"]) + 1e-8
    else:
        scale = math.hypot(lhip["x"] - rhip["x"], lhip["y"] - rhip["y"]) * 2 + 1e-8
    return [((p["x"] - cx) / scale, (p["y"] - cy) / scale) for p in frame_points]


def flatten_sequence(seq):
    """归一化帧序列 [(nx,ny),...] → np.array (N, 66)"""
    vectors = []
    for frame in seq:
        vec = []
        for pt in frame:
            vec.append(float(pt[0]))
            vec.append(float(pt[1]))
        vectors.append(np.array(vec, dtype=np.float32))
    if not vectors:
        return np.empty((0, 66), dtype=np.float32)
    return np.stack(vectors, axis=0)


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


def compute_dtw_dist(user_seq, tmpl_seq):
    """DTW 累积距离（与 d_bad 同口径）：重采样到模板帧数后比对"""
    target_n = len(tmpl_seq)
    a_sampled = resample_sequence(user_seq, target_n)
    mat_a = flatten_sequence(a_sampled)
    mat_b = flatten_sequence(tmpl_seq)
    align = dtw(mat_a, mat_b, dist_method="euclidean")
    return float(align.distance)


def adaptive_score(dist, d_bad):
    """k=2.2 × 招式专属 d_bad（问题①）：dist=d_bad 时约 54.5 分"""
    if d_bad <= 1e-8:
        return 0.0
    max_allowed = K * d_bad
    if dist >= max_allowed:
        return 0.0
    score = 100.0 * (1.0 - dist / max_allowed)
    return float(np.clip(score, 0.0, 100.0))


def estimate_orientation(points):
    if points is None:
        return "unknown"
    nose = points[0]
    left_sh = points[11]
    right_sh = points[12]
    shoulder_center_x = (left_sh["x"] + right_sh["x"]) / 2
    if nose["x"] < shoulder_center_x - 0.15:
        return "back"
    return "front"


def select_single_human(results_list):
    best = None
    best_sum = -1.0
    for res in results_list:
        if not res.pose_landmarks:
            continue
        total_vis = sum(lm.visibility for lm in res.pose_landmarks.landmark)
        if total_vis > best_sum:
            best_sum = total_vis
            best = res
    return best


# ========== 全局状态变量 ==========
lib = load_template_library(TEMPLATE_FILE)
user_buffer = []              # 归一化后的帧缓冲（滑窗）
bad_streak = 0
current_template_seq = None   # 归一化后的模板帧序列
current_d_bad = 0.0           # 招式专属标尺
frame_counter = 0
last_score_text = "等待足够动作片段……"
last_process_time = time.time()


def set_selected_pose(name: str):
    """加载选中招式模板，预归一化，重置全部缓冲区（问题⑤）"""
    global current_template_seq, current_d_bad
    global user_buffer, bad_streak, frame_counter, last_score_text
    if name not in lib:
        return "❌ 当前选择招式不存在，请重新选择。"

    entry = lib[name]
    raw_frames = entry["frames"]
    norm_frames = []
    for fr in raw_frames:
        nf = normalize_frame_33(fr)
        if nf is not None:
            norm_frames.append(nf)
    if len(norm_frames) < MIN_TEMPLATE_FRAMES:
        return f"❌「{name}」模板归一化后有效帧不足"

    current_template_seq = norm_frames
    current_d_bad = entry["d_bad"]
    user_buffer.clear()
    bad_streak = 0
    frame_counter = 0
    last_score_text = "等待足够动作片段……"
    landmark_smooth_buf.clear()
    return f"✅ 已加载：{name}｜模板帧数：{len(norm_frames)}｜d_bad={current_d_bad:.2f}"


def process_frame(img):
    """单帧处理主逻辑：摄像头 → 骨架 → 判断 → DTW → 打分"""
    global user_buffer, bad_streak
    global frame_counter, last_score_text, last_process_time

    info_text = "等待采集动作片段……"
    score_text = last_score_text
    now = time.time()

    # 积压过重：直接丢弃旧帧，防止疯狂刷新
    if now - last_process_time > STALE_FRAME_GAP:
        last_process_time = now
        out = img.copy() if img is not None else None
        return out, "跳过积压旧帧，等待最新画面", score_text
    last_process_time = now

    if img is None:
        return None, info_text, score_text

    # 降分辨率，减轻推理负载（Gradio 传入已是 RGB）
    img = cv2.resize(img, (RESIZE_W, RESIZE_H))
    rgb_img = img
    result = pose.process(rgb_img)
    best_result = select_single_human([result])
    raw_kpts = extract_33_keypoints(best_result)
    kpts = smooth_points(raw_kpts)
    bad_rate = calculate_bad_ratio(kpts)

    if kpts is None:
        bad_streak += 1
        info_text = "未检测到人，请站到画面内。"
    elif bad_rate >= BAD_RATIO_THRESHOLD:
        bad_streak += 1
        info_text = f"可疑帧｜异常占比：{bad_rate * 100:.1f}%"
    else:
        bad_streak = 0
        norm_frame = normalize_frame_33(kpts)
        if norm_frame is None:
            info_text = "髋部关键点不清，请正对摄像头。"
        else:
            info_text = "采集正常，存入动作序列。"
            user_buffer.append(norm_frame)
            if len(user_buffer) > WINDOW_SIZE:
                user_buffer.pop(0)

    if bad_streak >= CONTINUOUS_BAD_MAX:
        info_text = "识别不稳！调整光线，避开叠手、急转身。"

    orient = estimate_orientation(kpts)
    if orient == "back":
        score_text = "方向不对，请转向正面练习。"
        last_score_text = score_text
    elif current_template_seq is not None:
        frame_counter += 1
        # 触发条件：缓冲 ≥ 保底帧数（问题⑤，max 会永不触发）
        if (frame_counter % SKIP_DTW_EVERY == 0) and (len(user_buffer) >= SCORE_TRIGGER_MIN):
            dist = compute_dtw_dist(user_buffer, current_template_seq)
            final_score = adaptive_score(dist, current_d_bad)

            if final_score >= 90:
                score_text = f"分数：{final_score:.1f}｜很好，继续保持！"
            elif final_score >= 70:
                score_text = f"分数：{final_score:.1f}｜大体到位，微调姿态。"
            elif final_score >= 40:
                score_text = f"分数：{final_score:.1f}｜存在明显偏移，请对照示范。"
            else:
                score_text = f"分数：{final_score:.1f}｜差距较大，请重新练习本段。"
            last_score_text = score_text

    out_img = img.copy()
    if best_result and best_result.pose_landmarks:
        mp_draw.draw_landmarks(
            out_img,
            best_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    return out_img, info_text, score_text


def build_app():
    """构建 Gradio｜新版摄像头参数（问题③）"""
    available_names = sorted(lib.keys())
    with gr.Blocks(title="竞赛 v03｜最终核验版") as demo:
        gr.Markdown("## 太极测试台（全部修复 + 流畅优化）")
        sel = gr.Dropdown(choices=available_names, label="选择招式")
        load_btn = gr.Button("加载模板")
        load_msg = gr.Textbox(label="加载状态")

        cam_in = gr.Image(
            sources=["webcam"],
            streaming=True,
            label="摄像头输入"
        )
        cam_out = gr.Image(label="带骨架画面")
        txt_info = gr.Textbox(label="状态提示")
        txt_score = gr.Textbox(label="打分结果")

        load_btn.click(
            set_selected_pose,
            inputs=sel,
            outputs=load_msg
        )
        cam_in.stream(
            process_frame,
            inputs=cam_in,
            outputs=[cam_out, txt_info, txt_score],
            time_limit=0.15
        )

    demo.queue(max_size=4)
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        show_error=False
    )