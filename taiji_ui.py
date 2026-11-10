# -*- coding: utf-8 -*-
"""
银发太极安全防伤评估系统（AI应用创新大赛版·UI增强）
- 前置健康建档筛查：严重心血管疾病/关节剧痛直接拦截
- 上传练习视频 → 提取骨架 → 定式极点检测 → 与模板余弦相似度比对
- 动态安全阈值引擎：膝关节炎患者阈值更严格
- 膝盖角度可视化标注 + HTML分级预警报告（动态彩色边框）
- 亲情健康周报生成（模拟推送子女）
- 自定义CSS：Hero横幅+卡片式布局+渐变按钮
"""
import json
import cv2
import numpy as np
import gradio as gr
import mediapipe as mp
from scipy.spatial.distance import cosine

# ===================== 初始化配置 =====================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
JSON_PATH = "upgrade_taiji.json"
VIS_THRESHOLD = 0.55

# MediaPipe 33关键点索引
KP = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# 加载模板库 + 动态生成招式选项（避免硬编码"揽雀尾"匹配失败）
with open(JSON_PATH, "r", encoding="utf-8") as f:
    TAIJI_TEMPLATES = json.load(f)["templates"]
template_map = {t["template_name"]: t for t in TAIJI_TEMPLATES}
action_options = [t["template_name"] for t in TAIJI_TEMPLATES]
DEFAULT_ACTION = "野马分鬃_左" if "野马分鬃_左" in template_map else action_options[0]

# 全局缓存最后一次评估数据，用于周报
last_eval_data = {}


# ===================== 工具函数 =====================
def calculate_angle(a, b, c):
    """三点夹角计算 a-b-c，b为顶点"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    return float(np.degrees(np.arccos(cos_angle)))


def normalize_vector(pts_xyz):
    """
    髋部归一化 + 展平，与模板生成时完全一致
    原点=双髋中点，尺度=双肩距离，否则与模板值域不同相似度失真
    pts_xyz: np.array (33, 3)
    """
    hip_center = (pts_xyz[KP["left_hip"]] + pts_xyz[KP["right_hip"]]) / 2.0
    normalized = pts_xyz - hip_center
    scale = np.linalg.norm(pts_xyz[KP["right_shoulder"]] - pts_xyz[KP["left_shoulder"]])
    if scale < 1e-8:
        scale = 1.0
    normalized = normalized / scale
    return normalized.flatten().astype(np.float32)


def process_video(video_path):
    """
    提取视频骨架关键点
    【关键】立即拷贝为numpy数组并绘制骨架，不保留mediapipe对象引用
    ——landmark对象在下一帧process后可能失效
    返回: draw_list(已绘骨架RGB帧), vec_list(归一化向量), pts_list(numpy(33,4))
    """
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    draw_list, vec_list, pts_list = [], [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if res.pose_landmarks:
            # 立即拷贝为numpy数组，脱离mediapipe对象生命周期
            pts = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility]
                 for lm in res.pose_landmarks.landmark],
                dtype=np.float32
            )
            vec = normalize_vector(pts[:, :3])
            draw_img = frame.copy()
            mp_draw.draw_landmarks(draw_img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            draw_list.append(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
            vec_list.append(vec)
            pts_list.append(pts)
    cap.release()
    pose.close()
    return draw_list, vec_list, pts_list


def find_peak_frame(vec_list):
    """
    极点查找：跳过开头20%、结尾10%的发呆帧
    从start+1开始计算相邻帧变化，避免start帧diff=0抢占极点
    返回极点帧索引
    """
    n = len(vec_list)
    if n < 5:
        return n // 2
    start = int(n * 0.2)
    end = int(n * 0.9)
    if start >= end:
        start, end = 0, n
    change_list = []
    for i in range(start + 1, end):
        change_list.append(cosine(vec_list[i], vec_list[i - 1]))
    if len(change_list) == 0:
        return n // 2
    local_idx = int(np.argmin(change_list))
    return start + 1 + local_idx


# ===================== 报告HTML渲染（动态左侧彩色边框） =====================
def build_report_html(age, chronic_list, recent_status, taiji_name, similarity,
                      warn_list, red_thresh):
    red_cnt = sum(1 for s in warn_list if "" in s)
    yellow_cnt = sum(1 for s in warn_list if "🟡" in s)

    if red_cnt > 0:
        bar, bg, badge, badge_bg = "#EF4444", "#FEF2F2", "高危风险", "#EF4444"
    elif yellow_cnt > 0:
        bar, bg, badge, badge_bg = "#F59E0B", "#FFFBEB", "需注意", "#F59E0B"
    else:
        bar, bg, badge, badge_bg = "#10B981", "#F0FDF4", "安全通过", "#10B981"

    chronic_text = "、".join(chronic_list) if chronic_list else "无基础慢病"
    items_html = ""
    for s in warn_list:
        if "🔴" in s:
            c, ibg = "#B91C1C", "#FEE2E2"
        elif "🟡" in s:
            c, ibg = "#B45309", "#FEF3C7"
        else:
            c, ibg = "#047857", "#D1FAE5"
        items_html += (
            f'<div style="background:{ibg};color:{c};border-radius:8px;'
            f'padding:10px 14px;margin-bottom:10px;font-size:16px;line-height:1.6;'
            f'font-weight:500;">{s}</div>'
        )

    return f"""
<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;
            box-shadow:0 6px 24px rgba(15,23,42,0.08);overflow:hidden;">
  <div style="border-left:8px solid {bar};background:{bg};padding:20px 24px;">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
      <div style="font-size:22px;font-weight:800;color:#0F172A;letter-spacing:0.5px;">
        📋 适老太极安全评估报告
      </div>
      <span style="background:{badge_bg};color:#FFFFFF;font-size:15px;font-weight:700;
                   padding:6px 16px;border-radius:999px;">{badge}</span>
    </div>
    <div style="margin-top:14px;font-size:16px;color:#334155;line-height:1.9;">
      <div><b>年龄：</b>{age} 岁　<b>慢病：</b>{chronic_text}</div>
      <div><b>近期状态：</b>{recent_status}</div>
      <div><b>评估招式：</b>{taiji_name}　<b>姿态相似度：</b>{similarity:.2%}</div>
      <div><b>本次动态膝盖高危阈值：</b>{red_thresh}°</div>
    </div>
  </div>
  <div style="padding:20px 24px;">
    <div style="font-size:18px;font-weight:700;color:#0F172A;margin-bottom:14px;">
      ⚠️ 风险检测结果
    </div>
    {items_html}
    <div style="margin-top:8px;font-size:14px;color:#94A3B8;">
      本报告由 AI 姿态识别生成，仅作为居家练习参考，不替代医嘱。
    </div>
  </div>
</div>
"""


# ===================== 核心评估函数（前置拦截+动态阈值） =====================
def evaluate(age, chronic_list, recent_status, taiji_name, video_file):
    global last_eval_data

    # ========== 前置准入拦截 ==========
    if "严重心血管疾病" in chronic_list or recent_status == "是，存在关节剧痛/头晕":
        intercept_html = """
<div style="background:#FEF2F2;border:1px solid #FCA5A5;border-left:8px solid #EF4444;
            border-radius:12px;box-shadow:0 6px 24px rgba(239,68,68,0.15);padding:24px;">
  <div style="font-size:22px;font-weight:800;color:#B91C1C;margin-bottom:12px;">
    🛑 运动风险拦截提示
  </div>
  <div style="font-size:16px;line-height:1.9;color:#7F1D1D;">
    🔴 检测到您存在 <b>极高运动风险</b>（严重心血管疾病 / 近期关节剧痛、头晕）。<br>
    本系统暂不适合您使用，请务必遵从医生指导，不要自行练习太极。
  </div>
</div>
"""
        return None, intercept_html, ""

    if not video_file:
        return None, '<div style="font-size:16px;color:#B45309;">⚠️ 请先上传练习视频</div>', ""

    # ========== 视频处理、提取极点帧 ==========
    draw_list, vec_list, pts_list = process_video(video_file)
    if len(vec_list) == 0:
        return None, '<div style="font-size:16px;color:#B45309;">⚠️ 未能提取有效人体姿态，请重新拍摄</div>', ""

    peak_idx = find_peak_frame(vec_list)

    # 可见度兜底：关键关节置信度低则退回中间帧
    KEY_JOINTS = [KP["left_shoulder"], KP["right_shoulder"],
                  KP["left_hip"], KP["right_hip"],
                  KP["left_knee"], KP["right_knee"]]
    vis_score = float(np.mean(pts_list[peak_idx][KEY_JOINTS, 3]))
    if vis_score < VIS_THRESHOLD:
        peak_idx = len(pts_list) // 2

    peak_vec = vec_list[peak_idx]
    peak_pts = pts_list[peak_idx]
    show_img = draw_list[peak_idx].copy()

    # ========== 匹配模板 ==========
    target_template = template_map.get(taiji_name)
    if target_template is None:
        return None, f'<div style="font-size:16px;color:#B45309;">⚠️ 招式「{taiji_name}」暂无标准模板</div>', ""
    similarity = 1 - cosine(peak_vec, np.array(target_template["vector"], dtype=np.float32))

    # ========== 动态安全阈值引擎 ==========
    if "膝关节炎" in chronic_list:
        warn_thresh_red, warn_thresh_yellow = 100.0, 110.0
    else:
        warn_thresh_red, warn_thresh_yellow = 90.0, 100.0

    # 计算双膝角度（像素2D坐标，numpy索引）
    h, w = show_img.shape[:2]
    lh = [peak_pts[KP["left_hip"], 0] * w, peak_pts[KP["left_hip"], 1] * h]
    lk = [peak_pts[KP["left_knee"], 0] * w, peak_pts[KP["left_knee"], 1] * h]
    la = [peak_pts[KP["left_ankle"], 0] * w, peak_pts[KP["left_ankle"], 1] * h]
    knee_left_angle = calculate_angle(lh, lk, la)
    rh = [peak_pts[KP["right_hip"], 0] * w, peak_pts[KP["right_hip"], 1] * h]
    rk = [peak_pts[KP["right_knee"], 0] * w, peak_pts[KP["right_knee"], 1] * h]
    ra = [peak_pts[KP["right_ankle"], 0] * w, peak_pts[KP["right_ankle"], 1] * h]
    knee_right_angle = calculate_angle(rh, rk, ra)

    # ========== 画面标注（骨架已绘制，追加膝盖角度） ==========
    # show_img为RGB通道：红=(255,0,0)，绿=(0,200,0)
    def draw_knee_label(img, pt, angle_val):
        color = (255, 0, 0) if angle_val < warn_thresh_red else (0, 200, 0)
        cv2.putText(img, f"{angle_val:.1f}", (int(pt[0]) + 10, int(pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    draw_knee_label(show_img, lk, knee_left_angle)
    draw_knee_label(show_img, rk, knee_right_angle)

    # ========== 风险判定 ==========
    warn_list = []
    if knee_left_angle < warn_thresh_red:
        warn_list.append(f"🔴【高危】左膝角度 {knee_left_angle:.1f}°，低于安全阈值 {warn_thresh_red:.0f}°，下蹲过大，易损伤膝盖")
    elif knee_left_angle < warn_thresh_yellow:
        warn_list.append(f"🟡【注意】左膝角度 {knee_left_angle:.1f}°，建议抬高重心，减少下蹲幅度")
    else:
        warn_list.append(f"🟢【安全】左膝角度 {knee_left_angle:.1f}°")

    if knee_right_angle < warn_thresh_red:
        warn_list.append(f"🔴【高危】右膝角度 {knee_right_angle:.1f}°，低于安全阈值 {warn_thresh_red:.0f}°，下蹲过大，易损伤膝盖")
    elif knee_right_angle < warn_thresh_yellow:
        warn_list.append(f"🟡【注意】右膝角度 {knee_right_angle:.1f}°，建议抬高重心，减少下蹲幅度")
    else:
        warn_list.append(f"🟢【安全】右膝角度 {knee_right_angle:.1f}°")

    if "高血压" in chronic_list and taiji_name == "海底针":
        warn_list.insert(0, "🔴【慢病专属提醒】高血压人群练习海底针，俯身低头幅度过大容易诱发脑部供血异常，请放缓动作！")

    report_html = build_report_html(age, chronic_list, recent_status, taiji_name,
                                    similarity, warn_list, warn_thresh_red)

    last_eval_data = {
        "age": age,
        "chronic_list": chronic_list,
        "recent_status": recent_status,
        "taiji_name": taiji_name,
        "similarity": similarity,
        "warn_list": warn_list,
        "red_thresh": warn_thresh_red,
    }
    return show_img, report_html, ""


# ===================== 亲情周报 =====================
def generate_family_report():
    if not last_eval_data:
        return "暂无评估记录，请先完成一次太极练习评估。"

    red_cnt = sum(1 for s in last_eval_data["warn_list"] if "🔴" in s)
    yellow_cnt = sum(1 for s in last_eval_data["warn_list"] if "🟡" in s)
    if red_cnt > 0:
        risk = "存在高危风险动作，建议暂停练习并咨询医生"
    elif yellow_cnt > 0:
        risk = "有轻微风险提示，建议调整动作幅度"
    else:
        risk = "动作安全规范，请继续保持"

    report = f"""
### 🧡 亲情健康周报（模拟推送子女）

**长辈年龄**：{last_eval_data['age']} 岁
**慢病情况**：{"、".join(last_eval_data['chronic_list']) if last_eval_data['chronic_list'] else "无"}
**最近练习招式**：{last_eval_data['taiji_name']}（相似度 {last_eval_data['similarity']:.2%}）
**综合风险提示**：{risk}

**详细风险条目**：
"""
    for s in last_eval_data["warn_list"]:
        report += f"- {s}\n"
    report += "\n> 本周报由 AI 自动生成，已模拟推送至子女微信，让关爱不缺席。"
    return report


# ===================== 自定义 CSS =====================
custom_css = """
/* ===== 全局字体与背景 ===== */
body, .gradio-container {
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif !important;
    background: #F1F5F9 !important;
}

/* ===== Hero 横幅 ===== */
.hero-banner {
    background: linear-gradient(135deg, #15803D 0%, #16A34A 50%, #4ADE80 100%) !important;
    border-radius: 16px !important;
    padding: 28px 32px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 10px 30px rgba(22, 163, 74, 0.25) !important;
}
.hero-banner h1 {
    color: #FFFFFF !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: 1px !important;
}
.hero-banner p {
    color: #DCFCE7 !important;
    font-size: 15px !important;
    margin: 0 !important;
    line-height: 1.6 !important;
}

/* ===== 卡片容器 ===== */
.card-ui {
    background: #FFFFFF !important;
    border-radius: 14px !important;
    padding: 20px 24px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06) !important;
    border: 1px solid #E2E8F0 !important;
    margin-bottom: 16px !important;
}

/* ===== 主按钮：渐变绿 ===== */
.primary-btn {
    background: linear-gradient(135deg, #16A34A, #22C55E) !important;
    color: #FFFFFF !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    box-shadow: 0 6px 20px rgba(22, 163, 74, 0.35) !important;
    transition: all 0.25s ease !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(22, 163, 74, 0.45) !important;
}

/* ===== 次要按钮：白底描边 ===== */
.secondary-btn {
    background: #FFFFFF !important;
    color: #15803D !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border: 2px solid #16A34A !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    transition: all 0.25s ease !important;
}
.secondary-btn:hover {
    background: #F0FDF4 !important;
    transform: translateY(-1px) !important;
}

/* ===== 报告区域 ===== */
.report-host {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ===== 周报区域 ===== */
.weekly-area {
    background: #FFFBEB !important;
    border: 1px solid #FDE68A !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}

/* ===== 隐藏页脚 ===== */
footer { display: none !important; }
"""


# ===================== Gradio UI =====================
with gr.Blocks(title="银发太极安全防伤评估平台") as demo:
    # Hero 横幅
    gr.HTML("""
<div class="hero-banner">
    <h1>🥋 银发太极 · 安全防伤评估平台</h1>
    <p>AI 姿态识别守护长辈健康 —— 健康建档、动态阈值、风险预警、亲情周报，一站式守护</p>
</div>
""")

    with gr.Row(equal_height=False):
        # ============ 左侧：健康建档 + 视频上传 ============
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["card-ui"]):
                gr.Markdown("### 👤 长辈健康建档")
                age_input = gr.Slider(minimum=50, maximum=95, value=65, step=1,
                                      label="年龄", elem_classes=["card-ui"])
                chronic_input = gr.CheckboxGroup(
                    choices=["高血压", "膝关节炎", "严重心血管疾病", "糖尿病", "骨质疏松"],
                    label="基础慢病（可多选）",
                )
                status_input = gr.Radio(
                    choices=["状态良好", "轻微疲劳", "是，存在关节剧痛/头晕"],
                    value="状态良好",
                    label="近期身体状态",
                )

            with gr.Group(elem_classes=["card-ui"]):
                gr.Markdown("### 🎬 上传练习视频")
                taiji_select = gr.Dropdown(
                    choices=action_options,
                    value=DEFAULT_ACTION,
                    label="请选择练习的太极招式",
                )
                vid_upload = gr.Video(label="上传太极练习视频",
                                      sources=["upload"], elem_classes=["media-area"])
                btn_eval = gr.Button("🧠 开始安全评估", variant="primary",
                                     elem_classes=["primary-btn"])

        # ============ 右侧：评估结果面板 ============
        with gr.Column(scale=2):
            with gr.Group(elem_classes=["card-ui"]):
                gr.Markdown("### 🖼️ 骨架姿态与膝盖角度标注")
                out_img = gr.Image(label="极点帧标注图", type="numpy")

            with gr.Group(elem_classes=["card-ui"]):
                gr.Markdown("### 📋 安全评估报告")
                out_report = gr.HTML(elem_classes=["report-host"])

            with gr.Group(elem_classes=["card-ui"]):
                gr.Markdown("### 🧡 亲情健康周报")
                btn_week = gr.Button("生成亲情周报（模拟推送子女）",
                                     elem_classes=["secondary-btn"])
                out_week_text = gr.Markdown("暂无周报，请先完成评估。",
                                            label="亲情周报预览",
                                            elem_classes=["weekly-area"])

    # 事件绑定：评估 → (标注图, HTML报告, 清空周报)
    btn_eval.click(
        fn=evaluate,
        inputs=[age_input, chronic_input, status_input, taiji_select, vid_upload],
        outputs=[out_img, out_report, out_week_text],
    )
    # 周报按钮 → 周报文本
    btn_week.click(
        fn=generate_family_report,
        inputs=[],
        outputs=[out_week_text],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Soft(), css=custom_css)



