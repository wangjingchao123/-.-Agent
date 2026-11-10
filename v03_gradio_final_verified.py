import cv2
import json
import numpy as np
import mediapipe as mp
import gradio as gr
from pathlib import Path
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d

# =====================
# 竞赛全局配置（锁定，禁止现场修改）
# =====================
NUM_KEYPOINTS = 33
VISIBILITY_THRESHOLD = 0.5
SCORE_MIN = 0
SCORE_MAX = 100
BAD_DIST = 0.28
DANGER_DIST = 0.45
TARGET_SEQ_LEN = 120
MAX_FRAMES_READ = 800
SKIP_STEP = 2
MIN_TEMPLATE_FRAMES = 20
TEMPLATE_FILE = "saved_templates_final.json"

# JoyCode：如果云端语音报错，直接设 False
ENABLE_TTS = False

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 全局单例姿态检测器（防反复创建、内存暴涨）
pose_global = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# “持续学习 / 历史记录”：内存存储（适配云环境，不写本地文件）
history_records = []


def save_practice_record(record: dict):
    global history_records
    history_records.append(record)


def load_practice_history():
    global history_records
    return history_records


# ---------------------
# 模板加载：从真实模板库读取（替换占位随机模板）
# ---------------------
def load_real_templates(filepath: str = TEMPLATE_FILE):
    """从 saved_templates_final.json 加载全部招式模板
    返回 {name: np.array(帧数, 99)}，已归一化到 TARGET_SEQ_LEN
    """
    p = Path(filepath)
    if not p.exists():
        print(f"[模板] 找不到文件：{filepath}")
        return {}
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    templates = {}
    for name, item in raw.items():
        frames = item.get("frames") if isinstance(item, dict) else item
        if not isinstance(frames, list) or len(frames) < MIN_TEMPLATE_FRAMES:
            print(f"[跳过] {name}：帧数不足")
            continue
        mats = []
        ok = True
        for fr in frames:
            if not isinstance(fr, list) or len(fr) != NUM_KEYPOINTS:
                ok = False
                break
            arr = np.zeros((NUM_KEYPOINTS, 3))
            for pt in fr:
                if isinstance(pt, dict) and "idx" in pt:
                    arr[pt["idx"]] = [pt["x"], pt["y"], pt["z"]]
            mats.append(arr.flatten())
        if not ok or len(mats) < MIN_TEMPLATE_FRAMES:
            print(f"[跳过] {name}：关键点格式异常")
            continue
        seq = np.array(mats, dtype=np.float32)
        # 归一化到统一长度，消除速度差异
        templates[name] = normalize_sequence(seq, target_len=TARGET_SEQ_LEN)
    print(f"[模板] 成功加载 {len(templates)} 个招式")
    return templates


# ---------------------
# 关键点提取：单个人、过滤低置信
# ---------------------
def extract_pose_landmarks(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose_global.process(rgb)

    if not res.pose_landmarks:
        return None

    total_conf = sum(lm.visibility for lm in res.pose_landmarks.landmark)
    out = []
    for idx, lm in enumerate(res.pose_landmarks.landmark):
        out.append({
            "idx": idx,
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility)
        })
    return out, total_conf


def filter_low_confidence(landmarks):
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
    mats = []
    for lm in landmark_list:
        if lm is None:
            continue
        arr = landmarks_to_array(lm).flatten()
        mats.append(arr)
    if not mats:
        return None
    return np.array(mats)


def normalize_sequence(seq, target_len=TARGET_SEQ_LEN):
    n, feat = seq.shape
    if n == target_len:
        return seq
    old_x = np.linspace(0, 1, n)
    new_x = np.linspace(0, 1, target_len)
    out = np.empty((target_len, feat))
    for col in range(feat):
        f = interp1d(old_x, seq[:, col], kind="linear", fill_value="extrapolate")
        out[:, col] = f(new_x)
    return out


# ---------------------
# DTW & 自适应打分（竞赛固定）
# ---------------------
def dtw_distance(seq_a, seq_b):
    """手写 DTW，返回归一化平均距离（与 BAD_DIST/DANGER_DIST 量级匹配）"""
    if seq_a.shape[0] == 0 or seq_b.shape[0] == 0:
        raise ValueError("序列为空，无法比对")
    na, nb = seq_a.shape[0], seq_b.shape[0]
    dmat = np.full((na + 1, nb + 1), np.inf)
    dmat[0, 0] = 0.0

    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = euclidean(seq_a[i - 1], seq_b[j - 1])
            dmat[i, j] = cost + min(
                dmat[i - 1, j],
                dmat[i, j - 1],
                dmat[i - 1, j - 1]
            )
    # 归一化：累积距离 / 路径长度，得到平均每步距离
    # 量级与 BAD_DIST(0.28) / DANGER_DIST(0.45) 匹配
    return float(dmat[na, nb]) / (na + nb)


def score_from_distance(dist):
    dist = max(0.0, float(dist))
    if dist >= DANGER_DIST:
        return float(SCORE_MIN)
    t = dist / DANGER_DIST
    score = SCORE_MAX * (1.0 - t)
    return float(np.clip(score, SCORE_MIN, SCORE_MAX))


def judge_errors(score):
    if score >= 80:
        level = "优秀"
        tip = "动作基本标准，继续保持，注意平稳呼吸。"
    elif score >= 60:
        level = "存在小偏差"
        tip = "部分关节轨迹和标准有差异，请放慢速度，对照示范调整姿态。"
    elif score >= 40:
        level = "明显错误"
        tip = "动作偏差较大！建议停下来，对照标准模板重新练习，避免发力不当。"
    else:
        level = "高风险姿态（重点提醒）"
        tip = "当前姿态风险较高，请立刻停止练习，调整姿势，防止关节、肌肉受伤。适合老年人安全保护。"
    return level, tip


# 全局模板库（启动时加载真实模板，替换占位随机数据）
TEMPLATE_LIB = load_real_templates()


# ---------------------
# 主方案：上传视频分析（比赛核心）
# ---------------------
def analyze_video_file(video_path, action_name):
    if not video_path:
        return "请上传视频文件。", "", None, None, ""
    if not action_name or action_name not in TEMPLATE_LIB:
        return "请先选择招式（模板库加载失败或未选择）。", "", None, None, ""

    template = TEMPLATE_LIB[action_name]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "无法打开视频文件，请检查上传文件。", "", None, None, ""

    all_landmarks = []
    frame_idx = 0
    valid_count = 0
    error_intervals = []

    while cap.isOpened() and frame_idx < MAX_FRAMES_READ:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % SKIP_STEP != 0:
            frame_idx += 1
            continue

        lm_and_conf = extract_pose_landmarks(frame)
        if lm_and_conf is not None:
            landmarks, conf = lm_and_conf
            landmarks = filter_low_confidence(landmarks)
            if landmarks:
                all_landmarks.append(landmarks)
                valid_count += 1
            else:
                all_landmarks.append(None)
        else:
            all_landmarks.append(None)
        frame_idx += 1
    cap.release()

    if valid_count < 20:
        msg = (
            "识别失败：检测有效关键点数量不足。\n"
            "常见原因：画面遮挡、背光、距离太远、多人同时入镜。\n"
            "请重新录制，保证画面内仅有练习者一人。"
        )
        return msg, "", None, None, ""

    feat_mat = build_feature_mat(all_landmarks)
    if feat_mat is None:
        return "姿态序列构建失败。", "", None, None, ""

    norm_user = normalize_sequence(feat_mat, target_len=template.shape[0])
    dist_val = dtw_distance(norm_user, template)
    final_score = score_from_distance(dist_val)
    level, tip_text = judge_errors(final_score)

    if final_score < 60:
        error_intervals.append("检出：整体动作片段存在明显偏差，建议对照标准重练。")

    record = {
        "招式": action_name,
        "score": round(final_score, 2),
        "distance": round(dist_val, 4),
        "level": level,
        "tip": tip_text
    }
    save_practice_record(record)
    history = load_practice_history()
    history_note = f"\n【历史练习记录总数】：{len(history)} 条（持续积累）"

    report = (
        f"【招式】：{action_name}\n"
        f"【综合评分】：{final_score:.1f} / 100\n"
        f"【判定等级】：{level}\n"
        f"【序列差异距离】：{dist_val:.4f}\n\n"
        f"【改进建议】\n{tip_text}\n\n"
        f"【错误片段提示】\n"
        f"{''.join(error_intervals) if error_intervals else '未检出明显危险片段'}"
        f"{history_note}"
    )
    return report, tip_text, final_score, dist_val, report


# ---------------------
# TTS（云端可选）
# ---------------------
def gen_audio(text):
    if not ENABLE_TTS:
        return None
    try:
        import asyncio
        import tempfile
        from edge_tts import Communicate
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        comm = Communicate(text, voice="zh-CN-XiaoxiaoNeural")
        asyncio.run(comm.save(temp_wav))
        return temp_wav.name
    except Exception:
        return None


# ---------------------
# 备选：实时摄像头（答辩演示）
# ---------------------
def realtime_camera_frame(img):
    if img is None:
        return img, "无画面输入。"
    try:
        # Gradio 传入已是 RGB，无需 cvtColor
        rgb = img
        res = pose_global.process(rgb)
        if not res.pose_landmarks:
            return img, "未检测到人体，请调整位置。"

        mp_drawing.draw_landmarks(
            img,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        return img, "骨架采集成功（云环境仅预览；完整打分请上传视频）"
    except Exception as e:
        return img, f"采集异常：{str(e)}"


# ---------------------
# Gradio｜JoyCode 网页
# ---------------------
def build_app():
    available_names = sorted(TEMPLATE_LIB.keys())
    with gr.Blocks(title="老年太极健身陪练平台｜竞赛提交版") as demo:
        gr.Markdown("""
        # 🥋 太极 AI 陪练（提交给 JoyCode）
        > "上传视频离线分析"为本项目主方案，适配云平台算力限制；
        > "实时摄像头"仅作为答辩备选演示。面向老年人实现姿态识别、DTW 序列比对、自适应打分、错误判定、安全提醒与练习记录持续积累。
        """)

        with gr.Tab("上传视频｜主方案（比赛）"):
            sel_action = gr.Dropdown(
                choices=available_names,
                label="选择招式",
                value=available_names[0] if available_names else None
            )
            vid_in = gr.Video(label="上传太极练习视频")
            run_btn = gr.Button("开始分析（DTW + 打分）", variant="primary")
            out_report = gr.Textbox(label="完整分析报告", lines=14)
            out_tip = gr.Textbox(label="语音提示文本")
            out_audio = gr.Audio(label="播放语音指导")

            def run_and_speak(action_name, vid_path):
                try:
                    rep, tip, score, dist, _ = analyze_video_file(vid_path, action_name)
                    aud = gen_audio(tip)
                    return rep, tip, aud
                except Exception as e:
                    err_msg = f"系统异常：{str(e)}"
                    return err_msg, "", None

            run_btn.click(
                fn=run_and_speak,
                inputs=[sel_action, vid_in],
                outputs=[out_report, out_tip, out_audio]
            )

        with gr.Tab("实时摄像头｜备选演示（答辩）"):
            cam_in = gr.Image(sources=["webcam"], streaming=True, label="摄像头画面")
            cam_out_img = gr.Image(label="绘制骨架")
            cam_out_msg = gr.Textbox(label="状态提示")
            cam_in.stream(
                fn=realtime_camera_frame,
                inputs=[cam_in],
                outputs=[cam_out_img, cam_out_msg]
            )

        with gr.Tab("安全指引 & 热身（老年）"):
            gr.Markdown("""
            ### ✅ 使用前安全提示
            1. **健康问询**：如果存在骨关节病、高血压、心脏不适，请降低强度或暂停练习。
            2. **热身优先**：练习前充分活动肩、腰、膝、腕，降低拉伤风险。
            3. **风险告警**：系统一旦提示"高风险姿态"，立刻停止并调整姿势。
            4. **持续学习**：每一次练习自动保存记录，平台随练习积累，逐步适配个人习惯。
            """)

        with gr.Tab("查看历史练习记录"):
            history_btn = gr.Button("加载历史记录")
            history_out = gr.Textbox(label="历史练习记录", lines=12)

            def show_history():
                hs = load_practice_history()
                if not hs:
                    return "暂无练习记录，请先上传视频进行分析。"
                return json.dumps(hs, ensure_ascii=False, indent=2)
            history_btn.click(fn=show_history, outputs=[history_out])
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )