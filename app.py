# -*- coding: utf-8 -*-
"""
银发太极 · 安全防伤评估平台（Warm Medical Tech 版）
依赖：gradio  mediapipe  opencv-python  numpy  scipy（版本见 requirements.txt）
同目录需放：taiji_24.json（或自己用 build_24_templates.py 生成的库）

打分口径（这是本项目最容易被做错的地方）：
    不用余弦相似度，改成「相对骨架 + 逐点平均距离」
    ① x、z 按视频宽高比还原，让 x、y 同量纲
    ② 平移到髋中心，站位/镜头位置不再影响
    ③ 除以躯干长度，身材/机位远近不再影响
    ④ 只比 x、y 平面（MediaPipe 的 z 是单目估深，噪声比 xy 大得多）

安全检查口径：
    safety_checks 模板驱动引擎，按 type 分发：
    knee_angle / knee_over_toe / hand_height_diff / nose_hip_y / balance
    膝关节炎患者 knee_angle 阈值自动加严 +10°
"""
import html
import os
import sys
from pathlib import Path
import cv2
import gradio as gr
import json
import mediapipe as mp
import numpy as np

# ===================== 初始化 =====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 模板库文件：【JoyCode 上只改这一行】
#   taiji_24.json      = 用你的视频重新生成的库（推荐）
#   upgrade_taiji.json = 旧的 6 条示例库（点位已中心化，不建议再用）
DATA_FILE = os.environ.get("TAIJI_DATA", "taiji_24.json")

# JoyCode / 云容器的工作目录不一定是脚本所在目录，
# 这里按「脚本自身所在目录」拼绝对路径，避免"找不到 JSON"导致启动即退出
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(DATA_FILE) if os.path.isabs(DATA_FILE) else BASE_DIR / DATA_FILE
if not DATA_PATH.exists():
    if "--doctor" in sys.argv:
        TAIJI_TEMPLATES, TEMPLATE_NAMES = [], []
    else:
        raise SystemExit(
            f"找不到模板库文件：{DATA_PATH}\n"
            f"请确认 {DATA_FILE} 和 app.py 在同一个文件夹里。")

if DATA_PATH.exists():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, dict) or not isinstance(raw_data.get("templates"), list):
        raise SystemExit(f"模板库格式错误：{DATA_PATH} 须包含 templates 列表")
    TAIJI_TEMPLATES = raw_data["templates"]
    TEMPLATE_NAMES = [t["template_name"] for t in TAIJI_TEMPLATES]
    print(f"[启动] 模板库 {DATA_FILE}｜招式 {len(TEMPLATE_NAMES)} 条｜"
          f"已录入向量 {sum(1 for t in TAIJI_TEMPLATES if t.get('vector'))} 条")

# 关键点索引
KP = {
    "left_hip": 23, "left_knee": 25, "left_ankle": 27,
    "right_hip": 24, "right_knee": 26, "right_ankle": 28,
}

# 骨架绘制样式（OpenCV 是 BGR）
POINT_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
LINE_SPEC = mp_drawing.DrawingSpec(color=(166, 184, 20), thickness=2, circle_radius=2)  # 青绿

DEFAULT_ACTION = "野马分鬃_左" if "野马分鬃_左" in TEMPLATE_NAMES else (TEMPLATE_NAMES[0] if TEMPLATE_NAMES else None)

last_eval_data = {}


# ===================== 基础工具 =====================
def calculate_angle(a, b, c):
    """三点夹角，b 为顶点，返回 0~180"""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8 or np.any(np.isnan([a, b, c])):
        return None
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1, 1)
    return float(np.degrees(np.arccos(cos_angle)))


# ===================== 姿势比对：相对骨架表示 =====================
# 只比较这些关键点，取躯干 + 四肢主干，避免手指脚尖等噪声
CORE_POINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0, 9, 10]

# 分数换算常数：平均逐点距离达到 D0 时分数归零
DIST_SCALE = 0.50
# 视频默认宽高比，模板未记录 aspect 时使用
DEFAULT_ASPECT = 16.0 / 9.0


def _to_points(vec):
    """
    把模板 / 实时向量统一成 33×3 点位矩阵。
    兼容三种常见录入格式：
        99 维 = 33 点 × (x, y, z)
        66 维 = 33 点 × (x, y)          —— 只录了平面点位，z 补 0
        [33][2] / [33][3] 嵌套列表      —— 直接转换
    """
    v = np.asarray(vec, dtype=float)
    if v.ndim == 2 and v.shape[0] == 33:
        P = np.zeros((33, 3))
        P[:, :v.shape[1]] = v
        return P
    flat = v.reshape(-1)
    if flat.size == 99:
        return flat.reshape(33, 3)
    if flat.size == 66:
        P = np.zeros((33, 3))
        P[:, :2] = flat.reshape(33, 2)
        return P
    raise ValueError(f"点位维度不支持：{flat.size}（应为 66、99 或 33×2 / 33×3）")


def vector_ok(vec):
    """判断一条模板向量能不能用来比对"""
    if vec is None:
        return False
    try:
        _to_points(vec)
        return True
    except Exception:
        return False


def to_relative(vec, aspect=None):
    """
    把点位还原成「相对骨架」：
      1) x、z 按宽高比还原到同一量纲（x 按宽归一、y 按高归一，不还原就会因画幅比例漂移）
      2) 平移到以髋中心为原点（站位、镜头位置都不再影响）
      3) 除以躯干长度归一化（身材、机位远近都不再影响）
    """
    if aspect is None or aspect <= 0:
        aspect = DEFAULT_ASPECT
    P = _to_points(vec).copy()
    P[:, 0] *= aspect
    P[:, 2] *= aspect
    hip = (P[23] + P[24]) / 2.0
    P -= hip
    shoulder = (P[11] + P[12]) / 2.0
    scale = float(np.linalg.norm(shoulder))
    if scale < 1e-6:
        scale = 1.0
    P /= scale
    return P


def skeleton_distance(vec_a, vec_b, aspect_a=None, aspect_b=None):
    """
    相对骨架之间的平均逐点距离，越小越像。
    只取 x、y 平面：MediaPipe 的 z 是单目深度估计，量级可达 ±1，噪声远大于 xy。
    """
    A = to_relative(vec_a, aspect_a)
    B = to_relative(vec_b, aspect_b)
    return float(np.linalg.norm(A[CORE_POINTS, :2] - B[CORE_POINTS, :2], axis=1).mean())


def match_score(vec_a, vec_b, aspect_a=None, aspect_b=None):
    """返回 0~1 的动作标准度。1.0 为完全一致，0 为完全不像"""
    d = skeleton_distance(vec_a, vec_b, aspect_a, aspect_b)
    return float(np.clip(1.0 - d / DIST_SCALE, 0.0, 1.0))


def template_aspect(tpl):
    """
    决定一条模板该按什么宽高比解释：
      - 库里写了 aspect 就照它；
      - 没写、但坐标已经是「髋中心为原点」的归一化库，按 1.0 处理；
      - 其余情况（MediaPipe 原始图像坐标）按 16:9 兜底。
    """
    a = tpl.get("aspect")
    if a:
        try:
            return float(a)
        except (TypeError, ValueError):
            pass
    try:
        P = _to_points(tpl["vector"])
        torso = float(np.linalg.norm((P[11] + P[12]) / 2 - (P[23] + P[24]) / 2))
        hip_y = abs(float((P[23][1] + P[24][1]) / 2))
        if torso > 1e-6 and hip_y < 0.2 * torso and P[:, 1].min() < -0.2 * torso:
            return 1.0
    except Exception:
        pass
    return DEFAULT_ASPECT


def rank_templates(vec, aspect=None):
    """把当前姿势和库里所有招式比一遍，返回按相似度降序的 [(招式名, 分数)]"""
    out = []
    for t in TAIJI_TEMPLATES:
        if not vector_ok(t.get("vector")):
            continue
        out.append((t["template_name"],
                    match_score(vec, t["vector"], aspect, template_aspect(t))))
    out.sort(key=lambda x: -x[1])
    return out


def extract_frame_keypoints(frame, pose_inst):
    """关键点整体可见度过低才丢帧；绝不清零坐标，否则骨架线会塌到左上角"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose_inst.process(rgb)
    if not res.pose_landmarks:
        return None, None
    lms = res.pose_landmarks.landmark
    core = [11, 12, 23, 24, 25, 26, 27, 28]
    if np.mean([lms[i].visibility for i in core]) < 0.4:
        return None, None
    vec = [c for lm in lms for c in (lm.x, lm.y, lm.z)]
    return res.pose_landmarks, vec


def process_video(video_path):
    """
    每段视频新建一个 Pose 实例。
    复用同一个实例会让上一段视频的追踪状态串到下一段。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")
    frames, vecs, lms = [], [], []
    pose_local = None
    try:
        pose_local = mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as exc:
        cap.release()
        raise RuntimeError(f"姿态模型初始化失败：{exc}") from exc
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmark, vec = extract_frame_keypoints(frame, pose_local)
            if landmark is not None:
                frames.append(frame.copy())
                vecs.append(vec)
                lms.append(landmark)
    except Exception as exc:
        raise RuntimeError(f"视频分析失败：{exc}") from exc
    finally:
        cap.release()
        if pose_local is not None:
            pose_local.close()
    return frames, vecs, lms


def find_peak_frame(frames, vecs, lms, aspect=None):
    """极点查找：跳过开头 20%、结尾 10%，取定式帧前后均值抑制抖动"""
    n = len(vecs)
    if n < 5:
        return None, None, None
    start, end = int(n * 0.2), int(n * 0.9)
    if start >= end:
        start, end = 0, n
    rel = [to_relative(v, aspect) for v in vecs]
    change = []
    for i in range(start + 1, end):
        d = np.linalg.norm(rel[i][CORE_POINTS, :2] - rel[i - 1][CORE_POINTS, :2], axis=1).mean()
        change.append(float(d))
    if not change:
        return None, None, None
    peak_idx = start + 1 + int(np.argmin(change))

    # 定式帧不是单点，取它前后做平均：抑制关键点抖动，比单帧稳得多
    lo = max(start, peak_idx - 1)
    hi = min(end, peak_idx + 2)
    neighborhood = np.array(vecs[lo:hi], dtype=float)
    stable_vec = neighborhood.mean(axis=0).tolist()
    return frames[peak_idx], stable_vec, lms[peak_idx]


_CN_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]
_CN_FONT_PATH = None
for _fp in _CN_FONT_CANDIDATES:
    if os.path.exists(_fp):
        _CN_FONT_PATH = _fp
        break


def draw_warning_dot(img, pt_norm, angle, is_danger, h, w, side_cn=""):
    """在关节旁绘制半透明圆点 + 中文角度标注（PIL渲染，支持中文）
    side_cn: 中文侧别（"左膝"/"右膝"）。无中文字体时回退Hershey只画数字"""
    if angle is None:
        return
    cx = int(np.clip(pt_norm[0] * w, 10, w - 10))
    cy = int(np.clip(pt_norm[1] * h, 10, h - 10))

    # 半透明圆点
    dot_color = (60, 60, 235) if is_danger else (80, 200, 120)  # BGR 红/绿
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), 22, dot_color, -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # 角度文字
    if _CN_FONT_PATH:
        from PIL import Image, ImageDraw, ImageFont
        text = f"{side_cn} {angle:.0f}°"
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(_CN_FONT_PATH, 26)
        except Exception:
            font = ImageFont.load_default()
        draw.text((cx + 28, cy - 18), text, font=font,
                  fill=(220, 38, 38) if is_danger else (5, 150, 105),
                  stroke_width=3, stroke_fill=(255, 255, 255))
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        text = f"{'R ' if '右' in side_cn else 'L '}{angle:.0f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.72, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x = min(cx + 28, max(6, w - tw - 14))
        y = max(cy + 6, th + 14)
        cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 8), (25, 27, 32), -1)
        cv2.putText(img, text, (x, y), font, scale,
                    (86, 86, 245) if is_danger else (140, 210, 90), thick, cv2.LINE_AA)


# ===================== safety_checks 通用安全引擎 =====================
def run_safety_checks(lm, checks, chronic_list):
    """
    遍历模板 safety_checks 列表，按 type 分发（坐标取原始归一化值）：
    - knee_angle:      髋-膝-踝夹角 < min → 🔴；min ≤ 角度 < min+10 → 🟡（膝关节炎 min+10°）
    - knee_over_toe:   |膝x - 踝x| > max_offset → 🔴
    - hand_height_diff: 双腕(15/16)y差 < min_gap → 🟡
    - nose_hip_y:      鼻y - 髋中点y > min_gap → 🟠（俯身过度）
    - balance:         |髋中点x - 支撑踝x| > max_offset → 🟠（单腿重心偏移）
    """
    warn_list = []
    chronic_list = chronic_list or []
    if lm is None or len(lm) < 29:
        return ["🟡【注意】关键点数据不完整，无法完成安全检查"]

    def visible(index):
        value = getattr(lm[index], "visibility", 1.0)
        try:
            return float(value) >= 0.4
        except (TypeError, ValueError):
            return False

    def rule_number(rule, *keys, default):
        """兼容旧模板字段名，避免规则升级后悄悄失效。"""
        for key in keys:
            if key in rule:
                try:
                    return float(rule[key])
                except (TypeError, ValueError):
                    continue
        return default

    checks = checks if isinstance(checks, list) else []
    has_alert = False
    for chk in checks:
        if not isinstance(chk, dict):
            continue
        ctype = chk.get("type", "")
        msg = chk.get("msg", "")
        if ctype == "knee_angle":
            min_deg = rule_number(chk, "min", "min_angle", default=90.0)
            if "膝关节炎" in chronic_list:
                min_deg += 10.0
            for side, hi, ki, ai in (("左", 23, 25, 27), ("右", 24, 26, 28)):
                if not all(visible(index) for index in (hi, ki, ai)):
                    warn_list.append(f"🟡【注意】{side}膝关键点未识别到，请调整拍摄角度或光线")
                    continue
                ang = calculate_angle(
                    (lm[hi].x, lm[hi].y), (lm[ki].x, lm[ki].y), (lm[ai].x, lm[ai].y))
                if ang is None:
                    warn_list.append(f"\U0001F7E1【注意】{side}膝关键点未识别到，请调整拍摄角度或光线")
                    continue
                if ang < min_deg:
                    has_alert = True
                    warn_list.append(
                        f"\U0001F534【高危】{side}膝角度 {ang:.1f}°，低于安全下限 {min_deg:.0f}°。{msg}")
                elif ang < min_deg + 10:
                    warn_list.append(
                        f"\U0001F7E1【注意】{side}膝角度 {ang:.1f}°，建议抬高重心，减小下蹲幅度")
                else:
                    warn_list.append(f"\U0001F7E2【安全】{side}膝角度 {ang:.1f}°，在安全范围内")
        elif ctype == "knee_over_toe":
            max_off = rule_number(chk, "max_offset", "threshold", default=0.05)
            for side, ki, ai in (("左", 25, 27), ("右", 26, 28)):
                if not all(visible(index) for index in (ki, ai)):
                    continue
                offset = lm[ki].x - lm[ai].x
                if abs(offset) > max_off:
                    has_alert = True
                    warn_list.append(
                        f"\U0001F534【高危】{side}膝超过脚尖（偏移 {abs(offset):.2f}）。{msg}")
        elif ctype == "hand_height_diff":
            min_gap = rule_number(chk, "min_gap", "min_diff", default=0.10)
            if not all(visible(index) for index in (15, 16)):
                warn_list.append("🟡【注意】手腕关键点未识别到，无法判断双手高度")
                continue
            diff = abs(lm[15].y - lm[16].y)
            if diff < min_gap:
                has_alert = True
                warn_list.append(
                    f"\U0001F7E1【提示】两手高度差仅 {diff:.2f}（要求≥{min_gap:.2f}）。{msg}")
        elif ctype == "nose_hip_y":
            min_gap = rule_number(chk, "min_gap", "threshold", default=0.35)
            if not all(visible(index) for index in (0, 23, 24)):
                warn_list.append("🟡【注意】头部或髋部关键点未识别到，无法判断前倾")
                continue
            nose_y = lm[0].y
            hip_mid_y = (lm[23].y + lm[24].y) / 2.0
            if nose_y - hip_mid_y > min_gap:
                has_alert = True
                warn_list.append(
                    f"\U0001F7E0【风险】上身过度前倾（鼻-髋落差 {nose_y - hip_mid_y:.2f}）。{msg}")
        elif ctype == "balance":
            max_off = rule_number(chk, "max_offset", "threshold", default=0.08)
            if not all(visible(index) for index in (23, 24, 27, 28)):
                warn_list.append("🟡【注意】髋部或踝部关键点未识别到，无法判断平衡")
                continue
            # 支撑腿 = 踝y更大的那侧（脚在地面，y 向下为正）
            if lm[27].y >= lm[28].y:
                support_x, support_name = lm[27].x, "左"
            else:
                support_x, support_name = lm[28].x, "右"
            hip_mid_x = (lm[23].x + lm[24].x) / 2.0
            offset = abs(hip_mid_x - support_x)
            if offset > max_off:
                has_alert = True
                warn_list.append(
                    f"\U0001F7E0【跌倒风险】单腿支撑重心偏移 {offset:.2f}（{support_name}腿支撑）。{msg}")
    if not warn_list:
        warn_list.append("\U0001F7E2 姿态安全，各项检查均通过")
    return warn_list


# ===================== 报告卡片 =====================
def build_report_html(age, chronic_list, recent_status, taiji_name, similarity,
                      warn_list, red_thresh):
    chronic_list = chronic_list or []
    warn_list = warn_list or []
    age_text = html.escape(str(age))
    status_text = html.escape(str(recent_status or "未填写"))
    action_text = html.escape(str(taiji_name or "未选择"))
    chronic_text = ("、".join(html.escape(str(item)) for item in chronic_list)
                    if chronic_list else "无基础慢病")
    red_cnt = sum(1 for s in warn_list if "\U0001F534" in s)
    yellow_cnt = sum(1 for s in warn_list if "\U0001F7E1" in s)

    if red_cnt > 0:
        bar, soft, badge, badge_bg = "#EF4444", "#FEF2F2", "高危风险", "#EF4444"
    elif yellow_cnt > 0:
        bar, soft, badge, badge_bg = "#F59E0B", "#FFFBEB", "需注意", "#F59E0B"
    else:
        bar, soft, badge, badge_bg = "#10B981", "#F0FDF4", "安全通过", "#10B981"

    if similarity is None:
        sim_html = ('<div style="color:#64748B;font-size:15.5px;">'
                    '本次未做模板比对，仅执行安全评估</div>')
    else:
        if similarity >= 0.85:
            sim_tip, sim_word = "动作与标准模板高度吻合", "高度契合"
        elif similarity >= 0.70:
            sim_tip, sim_word = "动作与标准模板基本吻合", "比较接近"
        elif similarity >= 0.55:
            sim_tip, sim_word = "动作与标准模板略有差异，安全检测照常进行", "略有偏差"
        else:
            sim_tip, sim_word = "动作与所选招式差别较大，可以再确认一下招式是否选对", "差异明显"
        sim_html = (f'<div><b>动作标准度</b>　{sim_word}　'
                    f'<span style="color:#475569;font-size:15.5px;">{similarity:.0%}，{sim_tip}</span></div>')

    items = ""
    for s in warn_list:
        s = html.escape(str(s))
        if "\U0001F534" in s:
            c, ibg = "#991B1B", "#FEE2E2"
        elif "\U0001F7E1" in s:
            c, ibg = "#92400E", "#FEF3C7"
        else:
            c, ibg = "#065F46", "#D1FAE5"
        items += (
            f'<div style="display:inline-block;background:{ibg};color:{c};'
            f'border-radius:999px;padding:9px 16px;margin:5px 4px;'
            f'font-size:15.5px;line-height:1.6;font-weight:600;">{s}</div>'
        )

    return f"""
<div style="background:#FFFFFF;border-radius:16px;overflow:hidden;
            box-shadow:0 4px 20px rgba(0,0,0,0.05);border:1px solid #EEF2F7;">
  <div style="border-left:6px solid {bar};background:{soft};padding:22px 26px;">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
      <div style="font-size:22px;font-weight:800;color:#0F172A;letter-spacing:.3px;">
        评估报告
      </div>
      <span style="background:{badge_bg};color:#fff;font-size:14px;font-weight:700;
                   padding:6px 16px;border-radius:999px;">{badge}</span>
    </div>
    <div style="margin-top:16px;font-size:17.5px;color:#0F172A;line-height:2.0;">
      <div><b>年龄</b>　{age_text} 岁　｜　<b>慢病</b>　{chronic_text}</div>
      <div><b>近期状态</b>　{status_text}</div>
      <div><b>评估招式</b>　{action_text}</div>
      {sim_html}
      <div><b>本次安全下限</b>　{red_thresh:.0f}°（随健康档案动态调整）</div>
    </div>
  </div>
  <div style="padding:22px 26px 24px;">
    <div style="font-size:16px;font-weight:700;color:#0F172A;margin-bottom:12px;">
      风险检测结果
    </div>
    <div>{items}</div>
    <div style="margin-top:14px;font-size:13px;color:#94A3B8;line-height:1.7;">
      本报告由 AI 姿态识别生成，仅作居家练习参考，不替代医嘱。
    </div>
  </div>
</div>
"""


# ===================== 输入适配 =====================
def _video_source(video_file):
    """兼容 Gradio 不同版本返回的字符串、Path 和文件信息字典。"""
    if isinstance(video_file, dict):
        video_file = video_file.get("path") or video_file.get("name")
    if not video_file:
        return None
    source = Path(str(video_file)).expanduser()
    return source if source.is_file() else None


# ===================== 主评估 =====================
def evaluate(video_file, taiji_name, age, chronic_list, recent_status):
    """
    多帧时序采样版：
      每隔 SAMPLE_STEP 帧采样一帧 → 逐帧做膝角 / 超脚尖 / 支撑腿锁死判定
      → 汇总错误次数 → 取"蹲得最深"的最差帧画图 → 输出结构化诊断报告
    前置拦截、动态阈值逻辑与旧版完全一致。
    """
    global last_eval_data
    chronic_list = chronic_list or []

    # ================= 前置准入拦截（原样保留） =================
    if "严重心血管疾病" in chronic_list or recent_status == "是，存在关节剧痛/头晕":
        return None, """
<div style="background:#FFFFFF;border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,0.05);
            border-left:6px solid #EF4444;padding:26px;">
  <div style="font-size:20px;font-weight:800;color:#B91C1C;margin-bottom:12px;">运动风险拦截</div>
  <div style="font-size:15.5px;line-height:2.0;color:#7F1D1D;">
    检测到 <b>极高运动风险</b>（严重心血管疾病 / 近期关节剧痛、头晕）。<br>
    本系统暂不适合使用，请务必遵从医生指导，不要自行练习太极。
  </div>
</div>""", ""

    video_path = _video_source(video_file)
    if video_path is None:
        return None, '<div style="color:#B45309;font-size:16px;padding:8px 0;">请先上传一段练习视频</div>', ""

    # ================= 动态安全阈值（健康档案驱动，原样保留） =================
    if "膝关节炎" in chronic_list:
        red_thresh, yellow_thresh = 100.0, 110.0
    else:
        red_thresh, yellow_thresh = 90.0, 100.0

    TOE_LIMIT = 0.02      # 膝-踝横向偏移超过该值判超脚尖
    LOCK_ANGLE = 175.0    # 膝角大于该值判支撑腿锁死（过伸）
    SAMPLE_STEP = 8       # 每隔多少帧采样一次

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, '<div style="color:#B45309;font-size:16px;padding:8px 0;">视频文件打不开，请重新上传</div>', ""

    # 计数表：左/右分别统计
    cnt = {"deep_l": 0, "deep_r": 0, "mid_l": 0, "mid_r": 0,
           "toe_l": 0, "toe_r": 0, "lock_l": 0, "lock_r": 0}
    samples, skipped = 0, 0
    worst = None          # (角度, 侧别, 帧号)
    worst_pack = None     # (原始帧, pose_landmarks, 角度, 侧别)

    # 单段视频使用一个 Pose 实例即可（跨视频复用才会串扰）
    pose_local = None
    try:
        pose_local = mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as exc:
        cap.release()
        return None, (
            '<div style="color:#B45309;font-size:16px;padding:8px 0;">'
            f'姿态模型初始化失败：{html.escape(str(exc))}</div>'
        ), ""
    frame_idx = 0
    # 每侧的关键点编号与计数键
    sides = [("左", "left_hip", "left_knee", "left_ankle", "l"),
             ("右", "right_hip", "right_knee", "right_ankle", "r")]
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % SAMPLE_STEP == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose_local.process(rgb)
                plm = res.pose_landmarks

                if plm is None:
                    skipped += 1
                else:
                    lm = plm.landmark
                    core = [KP["left_hip"], KP["right_hip"], KP["left_knee"], KP["right_knee"],
                            KP["left_ankle"], KP["right_ankle"]]
                    if np.mean([lm[i].visibility for i in core]) < 0.4:
                        skipped += 1
                    else:
                        samples += 1
                        for side_name, hip_k, knee_k, ankle_k, tag in sides:
                            ang = calculate_angle(
                                (lm[KP[hip_k]].x, lm[KP[hip_k]].y),
                                (lm[KP[knee_k]].x, lm[KP[knee_k]].y),
                                (lm[KP[ankle_k]].x, lm[KP[ankle_k]].y))
                            if ang is None:
                                continue

                            # 1) 深蹲过深（红）/ 临界区间（黄）
                            if ang < red_thresh:
                                cnt["deep_" + tag] += 1
                            elif ang < yellow_thresh:
                                cnt["mid_" + tag] += 1

                            # 2) 膝盖超脚尖（图像坐标横向偏移）
                            toe_offset = abs(lm[KP[knee_k]].x - lm[KP[ankle_k]].x)
                            if toe_offset > TOE_LIMIT:
                                cnt["toe_" + tag] += 1

                            # 3) 支撑腿锁死（膝过伸）
                            if ang > LOCK_ANGLE:
                                cnt["lock_" + tag] += 1

                            # 4) 记录最差帧（角度最小 = 蹲得最深）
                            if worst is None or ang < worst[0]:
                                worst = (ang, side_name, frame_idx)
                                worst_pack = (frame.copy(), plm, ang, side_name)
            frame_idx += 1
    except Exception as exc:
        return None, (
            '<div style="color:#B45309;font-size:16px;padding:8px 0;">'
            f'视频分析失败：{html.escape(str(exc))}</div>'
        ), ""
    finally:
        cap.release()
        if pose_local is not None:
            pose_local.close()

    if samples < 3:
        return None, ('<div style="color:#B45309;font-size:16px;padding:8px 0;">'
                      f'有效采样仅 {samples} 帧（跳过 {skipped} 帧），请确保全身入镜、光线均匀后重拍</div>'), ""

    # ================= 在"最差帧"上画骨架 + 红色大字 =================
    worst_frame, worst_lm, worst_ang, worst_side = worst_pack
    h, w = worst_frame.shape[:2]
    draw_img = worst_frame.copy()
    mp_drawing.draw_landmarks(
        draw_img, worst_lm, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=POINT_SPEC,
        connection_drawing_spec=LINE_SPEC,
    )
    knee_key = "left_knee" if worst_side == "左" else "right_knee"
    # 只有真正低于安全下限时才标红，否则用绿色（无违规时"最差帧"不应误导）
    _is_red = worst_ang < red_thresh
    draw_warning_dot(draw_img,
                     (worst_lm.landmark[KP[knee_key]].x, worst_lm.landmark[KP[knee_key]].y),
                     worst_ang, _is_red, h, w, side_cn=f"{worst_side}膝")
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

    # ================= 汇总数字 =================
    deep_total = cnt["deep_l"] + cnt["deep_r"]
    mid_total = cnt["mid_l"] + cnt["mid_r"]
    toe_total = cnt["toe_l"] + cnt["toe_r"]
    lock_total = cnt["lock_l"] + cnt["lock_r"]
    red_total = deep_total
    yellow_total = mid_total + toe_total + lock_total

    # ================= 医生建议（按计数与慢病分支） =================
    advice = []
    if deep_total > 0:
        advice.append("出现多次深蹲过深，请务必抬高重心、减小下蹲幅度，避免半月板与髌股关节持续受压。")
    if toe_total > 0:
        advice.append("存在多次膝盖超脚尖，练习时注意膝盖与脚尖方向一致，不要让膝盖顶到脚尖前面。")
    if lock_total > 0:
        advice.append("支撑腿出现完全锁死（过伸），建议保持微屈，给膝关节留缓冲。")
    if "膝关节炎" in chronic_list:
        advice.append("考虑到膝关节炎病史，安全下限已提高到 100°，建议以高位动作练习为主，减少低位定式。")
    if "高血压" in chronic_list:
        advice.append("高血压人群请放慢起身与俯身节奏，避免突然低头，出现头晕立刻停下休息。")
    if not advice:
        advice.append("本次练习姿态规范，未发现明显风险，继续保持温和练习即可。")

    # ================= 面向老人的友好报告（无技术黑话） =================
    issue_items = [
        ("左膝下蹲过深", cnt["deep_l"], "#EF4444"),
        ("右膝下蹲过深", cnt["deep_r"], "#EF4444"),
        ("左膝超过脚尖", cnt["toe_l"],  "#F59E0B"),
        ("右膝超过脚尖", cnt["toe_r"],  "#F59E0B"),
        ("左腿撑得太直", cnt["lock_l"], "#F59E0B"),
        ("右腿撑得太直", cnt["lock_r"], "#F59E0B"),
    ]
    issue_items = sorted([x for x in issue_items if x[1] > 0], key=lambda x: -x[1])
    top3 = issue_items[:3]
    _maxv = top3[0][1] if top3 else 1

    def _bar(name, n, color):
        pct = max(6, int(round(n / _maxv * 100)))
        return f"""
    <div style="margin:13px 0;">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  font-size:16px;font-weight:700;color:#0F172A;">
        <span>{name}</span><span style="color:{color};font-weight:800;">{n} 次</span>
      </div>
      <div style="background:#F1F5F9;border-radius:999px;height:12px;margin-top:7px;overflow:hidden;">
        <div style="width:{pct}%;background:{color};height:100%;border-radius:999px;"></div>
      </div>
    </div>"""

    bars_html = "".join(_bar(n, v, c) for n, v, c in top3)
    if not bars_html:
        bars_html = ('<div style="font-size:16px;color:#047857;font-weight:700;padding:10px 0;">'
                     '本次没有发现需要调整的动作，您做得很稳。</div>')

    if red_total > 0:
        diag_emoji, diag_title, diag_desc = "🔴", "发现需要重点关注的动作", \
            "本次已为您完整扫描整套动作，综合风险偏高，下面是最需要留意的地方。"
        head_bg = "#FEF2F2"; head_bar = "#EF4444"
    elif yellow_total > 0:
        diag_emoji, diag_title, diag_desc = "🟡", "整体不错，有几个小细节", \
            "本次已为您完整扫描整套动作，没有大问题，只有几个小地方可以再调整。"
        head_bg = "#FFFBEB"; head_bar = "#F59E0B"
    else:
        diag_emoji, diag_title, diag_desc = "🟢", "动作很标准", \
            "本次已为您完整扫描整套动作，没有发现明显风险，继续保持。"
        head_bg = "#F0FDF4"; head_bar = "#10B981"

    if top3:
        _top_name = top3[0][0]
        coach_msg = (f"您这次练习很认真，不过在做「{taiji_name}」的时候，"
                     f"{_top_name}的次数偏多。别灰心，下次把重心稍微抬高一点、慢慢来，我一直陪着您。")
    else:
        coach_msg = "您这套动作打得很稳，几乎挑不出毛病，继续保持就好！"

    report = f"""
<div style="background:#FFFFFF;border-radius:16px;overflow:hidden;
            box-shadow:0 4px 20px rgba(0,0,0,0.06);border:1px solid #E2E8F0;">
  <div style="border-left:6px solid {head_bar};background:{head_bg};padding:20px 24px 18px;">
    <div style="font-size:21px;font-weight:800;color:#0F172A;line-height:1.5;">
      综合诊断：{diag_emoji} {diag_title}
    </div>
    <div style="margin-top:8px;font-size:16px;color:#475569;line-height:1.8;">{diag_desc}</div>
  </div>
  <div style="padding:18px 24px 10px;">
    <div style="display:flex;gap:14px;">
      <div style="flex:1;background:#EF4444;border:none;
                  border-radius:14px;padding:16px 10px;text-align:center;">
        <div style="font-size:36px;font-weight:800;color:#FFFFFF;line-height:1.1;">⚠️ {red_total}</div>
        <div style="color:#FFFFFF;font-size:15px;font-weight:700;margin-top:4px;">需要重点关注</div>
      </div>
      <div style="flex:1;background:#FEF3C7;border:1px solid #FDE68A;
                  border-radius:14px;padding:16px 10px;text-align:center;">
        <div style="font-size:36px;font-weight:800;color:#B45309;line-height:1.1;">💡 {yellow_total}</div>
        <div style="color:#B45309;font-size:15px;font-weight:700;margin-top:4px;">建议调整</div>
      </div>
    </div>
    <div style="font-size:18px;font-weight:800;color:#0F172A;margin:18px 0 2px;">最需要留意的 3 个地方</div>
    {bars_html}
    <div style="margin-top:16px;background:#ECFDF5;border:1px solid #A7F3D0;
                border-radius:14px;padding:16px 18px;">
      <div style="font-size:16px;font-weight:800;color:#065F46;margin-bottom:6px;">💬 教练寄语</div>
      <div style="font-size:16px;color:#064E3B;line-height:1.9;">{coach_msg}</div>
    </div>
    <div style="margin-top:14px;font-size:14.5px;color:#64748B;line-height:1.7;">
      本报告由 AI 姿态识别生成，仅作居家练习参考，不替代医生的专业诊断。
    </div>
  </div>
</div>
"""

    # ================= 供亲情周报使用的汇总条目（带 emoji 以便周报计数） =================
    warn_list = []
    if cnt["deep_l"]:
        warn_list.append(f"\U0001F534【高危】左膝深蹲过深 {cnt['deep_l']} 次，低于安全下限 {red_thresh:.0f}°")
    if cnt["deep_r"]:
        warn_list.append(f"\U0001F534【高危】右膝深蹲过深 {cnt['deep_r']} 次，低于安全下限 {red_thresh:.0f}°")
    if cnt["toe_l"]:
        warn_list.append(f"\U0001F7E1【注意】左膝超脚尖 {cnt['toe_l']} 次")
    if cnt["toe_r"]:
        warn_list.append(f"\U0001F7E1【注意】右膝超脚尖 {cnt['toe_r']} 次")
    if cnt["lock_l"]:
        warn_list.append(f"\U0001F7E1【注意】左腿支撑锁死 {cnt['lock_l']} 次")
    if cnt["lock_r"]:
        warn_list.append(f"\U0001F7E1【注意】右腿支撑锁死 {cnt['lock_r']} 次")
    if not warn_list:
        warn_list.append("\U0001F7E2【安全】多帧采样未发现明显风险，动作规范")

    last_eval_data = {
        "age": age, "chronic_list": chronic_list, "recent_status": recent_status,
        "taiji_name": taiji_name, "similarity": None,
        "warn_list": warn_list, "red_thresh": red_thresh, "best_match": None,
    }

    return draw_img, report, generate_family_report()




# ===================== 亲情周报 =====================
def generate_family_report():
    """微信气泡式周报：紧凑、浅绿底，标题居中，不再是空旷的框。"""
    d = last_eval_data
    if not d:
        return '<div align="center">📱 完成一次评估后，这里会生成给子女的消息</div>'

    red = sum(1 for x in d["warn_list"] if "\U0001F534" in x)
    yellow = sum(1 for x in d["warn_list"] if "\U0001F7E1" in x)
    chronic = ("、".join(html.escape(str(item)) for item in d["chronic_list"])
               if d["chronic_list"] else "无基础慢病")
    age = html.escape(str(d.get("age", "未填写")))
    action = html.escape(str(d.get("taiji_name") or "未选择"))

    if red > 0:
        mood = "今天有几个动作需要多留意，我会慢慢练，你们放心。"
    elif yellow > 0:
        mood = "今天整体不错，只有几个小细节要调整。"
    else:
        mood = "今天动作很标准，身体也没有不舒服，你们别担心。"

    return f"""<div align="center"><b>📱 模拟推送给子女的微信消息</b></div>

孩子，跟你说下我今天的练习情况：

- 我今年 **{age} 岁**，身体情况：{chronic}
- 今天练的是 **{action}**
- 需要重点关注：<b>{red}</b> 项　｜　建议调整：<b>{yellow}</b> 项
- {mood}

<div style="margin-top:8px;color:#047857;font-size:14.5px;">—— 银发太极陪练 · 模拟推送，不会真实发送</div>
"""


# ===================== 高级 CSS（干净后台风） =====================
custom_css = """
/* ---------- 全局：浅灰背景 + 清爽排版 ---------- */
.gradio-container {
    background: #F8FAFC !important;
    font-family: "PingFang SC", "Microsoft YaHei", "Hiragino Sans GB", Arial, sans-serif !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
    color: #334155 !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 12px 16px 28px !important;
}
html, body, gradio-app { background: #F8FAFC !important; }

/* ---------- 顶部 Banner：蓝绿渐变 ---------- */
#hero-banner {
    background: linear-gradient(135deg, #0EA5E9 0%, #10B981 100%);
    border-radius: 18px;
    padding: 40px 30px;
    margin: 8px 0 26px 0;
    text-align: center;
    box-shadow: 0 10px 28px rgba(16, 185, 129, 0.18);
}
#hero-banner .hero-title {
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    letter-spacing: 1.2px;
    line-height: 1.35;
    margin: 0 0 12px 0;
}
#hero-banner .hero-sub {
    font-size: 16.5px !important;
    font-weight: 700;
    color: #FFFFFF !important;
    letter-spacing: 0.4px;
    margin: 0;
}

/* ---------- 白卡片：16px 圆角 + 柔和阴影 ---------- */
.card-ui {
    background: #FFFFFF !important;
    border: 1px solid #EEF2F7 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
    padding: 26px 24px !important;
    margin-bottom: 20px !important;
}
.card-title h3, .card-title h2 {
    font-size: 18px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    border-left: 4px solid #10B981;
    padding-left: 12px;
    margin: 0 0 20px 0 !important;
    line-height: 1.4;
}

/* ---------- 主按钮：绿色渐变 + 悬停微放大 ---------- */
.primary-btn {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
    color: #FFFFFF !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    letter-spacing: 1px;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 20px !important;
    box-shadow: 0 6px 18px rgba(16, 185, 129, 0.28) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}
.primary-btn:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.38) !important;
}
.secondary-btn {
    background: #FFFFFF !important;
    color: #059669 !important;
    border: 1.5px solid #10B981 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 12px 18px !important;
    box-shadow: none !important;
    transition: background-color .18s ease !important;
}
.secondary-btn:hover { background: #ECFDF5 !important; }

/* ---------- 亲情周报：浅绿微信气泡 ---------- */
.weekly-area {
    background: #ECFDF5 !important;
    border: 1px solid #A7F3D0 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04) !important;
    padding: 20px !important;
    min-height: 160px;
    font-size: 16.5px !important;
    line-height: 1.85 !important;
}
.weekly-area .prose, .weekly-area .prose *,
.weekly-area p, .weekly-area span, .weekly-area div {
    color: #0F172A !important;
    font-size: 16.5px !important;
    line-height: 1.85 !important;
    background: transparent !important;
}
.weekly-area strong, .weekly-area b { color: #065F46 !important; font-weight: 800 !important; }

/* ---------- 卡片内部：去掉 Soft 主题的灰底(.styler)与标签紫底 ---------- */
.card-ui .styler,
.card-ui .styler > div,
.card-ui .block,
.card-ui .form,
.card-ui .wrap,
.card-ui .gr-box,
.card-ui fieldset,
.card-ui span,
.card-ui label {
    background: transparent !important;
    border-color: transparent !important;
}
.card-ui label,
.card-ui label span,
.card-ui span,
.card-ui .label-text,
.card-ui .block-info {
    color: #334155 !important;
}
.card-ui label { font-weight: 600 !important; }
.card-ui input,
.card-ui select,
.card-ui textarea {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}
.card-title, .card-title > *, .card-title .prose { background: transparent !important; }

/* 媒体区浅灰底，空态不发黑 */
.card-ui .image-container,
.card-ui .video-container,
.card-ui .media-container {
    background: #F1F5F9 !important;
    border-radius: 12px !important;
}


/* ---------- 按钮占满整栏 + 视频区压缩高度 ---------- */
.full-btn {
    width: 100% !important;
    min-height: 58px !important;
    display: block !important;
}
.left-col  { flex: 1 1 0 !important; }
.right-col { flex: 1.2 1 0 !important; }
.card-ui .image-container { height: 350px !important; }
.card-ui .image-container img { object-fit: contain !important; }
.card-ui .video-container { height: 240px !important; }

footer { display: none !important; }
"""

# ===================== Gradio 界面布局 =====================
with gr.Blocks(title="银发太极 · 安全防伤评估平台") as demo:
    gr.HTML("""
    <div id="hero-banner">
        <div class="hero-title">银发太极 · 安全防伤评估平台</div>
        <div class="hero-sub">AI 姿态识别守护长辈健康 —— 健康建档 · 动态阈值 · 风险预警 · 亲情周报</div>
    </div>
    """)
    with gr.Row(equal_height=False, elem_classes=["main-row"]):
        # 左：窄栏
        with gr.Column(scale=1, min_width=360, elem_classes=["left-col"]):
            with gr.Group(elem_classes=["card-ui"]):
                gr.Markdown("### 步骤 1 ｜ 长辈健康建档", elem_classes=["card-title"])
                age_slider = gr.Slider(minimum=50, maximum=95, value=65, step=1, label="老人年龄（岁）")
                chronic_box = gr.Dropdown(choices=["高血压", "膝关节炎", "糖尿病", "严重心血管疾病", "骨质疏松"],
                                          multiselect=True, label="基础慢病（可多选）", value=[])
                recent_radio = gr.Dropdown(choices=["状态良好", "轻微疲劳", "是，存在关节剧痛/头晕"],
                                           multiselect=False, label="近期身体状态", value="状态良好")
            with gr.Group(elem_classes=["card-ui"]):
                gr.Markdown("### 步骤 2 ｜ 上传练习视频", elem_classes=["card-title"])
                vid_in = gr.Video(label="上传太极练习视频", sources=["upload"], height=240)
                taiji_select = gr.Dropdown(
                    choices=TEMPLATE_NAMES, label="请选择练习的太极招式",
                    value=DEFAULT_ACTION)
                btn_start = gr.Button("开始安全评估", elem_classes=["primary-btn", "full-btn"])
        # 右：宽栏
        with gr.Column(scale=1.2, min_width=520, elem_classes=["right-col"]):
            with gr.Group(elem_classes=["card-ui"]):
                gr.HTML('<div class="card-title"><h3>骨架与关节角度标注</h3></div>')
                out_img = gr.Image(label="定式帧标注图", type="numpy",
                                   value=None, interactive=False,
                                   height=350)
            with gr.Group(elem_classes=["card-ui"]):
                gr.HTML('<div class="card-title"><h3>评估报告</h3></div>')
                risk_html = gr.HTML(
                    '<div style="color:#94A3B8;font-size:15px;padding:12px 0;">'
                    '完成评估后，此处将显示风险检测报告</div>')
            with gr.Group(elem_classes=["card-ui"]):
                gr.HTML('<div class="card-title"><h3>亲情健康周报</h3></div>')
                report_btn = gr.Button("生成亲情周报（模拟推送子女）", elem_classes="secondary-btn")
                report_out = gr.Markdown(
                    value='<div style="color:#94A3B8;font-size:15px;padding:12px 0;">'
                          '完成评估后，此处将显示亲情周报</div>',
                    label="", elem_classes=["weekly-area"])

    btn_start.click(
        fn=evaluate,
        inputs=[vid_in, taiji_select, age_slider, chronic_box, recent_radio],
        outputs=[out_img, risk_html, report_out])
    report_btn.click(
        fn=generate_family_report,
        outputs=[report_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Soft(), css=custom_css)
