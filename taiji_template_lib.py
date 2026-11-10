import json
import math
import os
import cv2
import mediapipe as mp
from pathlib import Path

# ===================== 配置常量 =====================
CONF_THRESH_HIGH = 0.60
CONF_THRESH_MID  = 0.40
MAX_JUMP_PIXEL   = 65
BAD_CONTINUOUS_FRAME = 12
# ✅ 放宽：允许到 42% 异常帧（适配下蹲、低头遮挡）
BAD_RATIO_THRESHOLD = 0.42

# ✅ 更新完整 24 式（含两个单鞭）
TAIJI_24 = [
    {"id": 1,  "name": "起势"},
    {"id": 2,  "name": "左右野马分鬃"},
    {"id": 3,  "name": "白鹤亮翅"},
    {"id": 4,  "name": "左右搂膝拗步"},
    {"id": 5,  "name": "手挥琵琶"},
    {"id": 6,  "name": "倒卷肱"},
    {"id": 7,  "name": "左揽雀尾"},
    {"id": 8,  "name": "右揽雀尾"},
    {"id": 9,  "name": "单鞭（一）"},
    {"id": 10, "name": "云手"},
    {"id": 11, "name": "单鞭（二）"},
    {"id": 12, "name": "高探马"},
    {"id": 13, "name": "右蹬脚"},
    {"id": 14, "name": "双峰贯耳"},
    {"id": 15, "name": "转身左蹬脚"},
    {"id": 16, "name": "左下势独立"},
    {"id": 17, "name": "右下势独立"},
    {"id": 18, "name": "左右穿梭"},
    {"id": 19, "name": "海底针"},
    {"id": 20, "name": "闪通臂"},
    {"id": 21, "name": "转身搬拦捶"},
    {"id": 22, "name": "如封似闭"},
    {"id": 23, "name": "十字手"},
    {"id": 24, "name": "收势"},
]

# 采集脚本可能从任意工作目录启动，数据文件应始终相对本模块定位。
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = BASE_DIR / "taiji_template_lib.json"
saved_templates = dict()

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def point_distance(p1, p2):
    return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])


def filter_keypoint(x, y, conf):
    if conf >= CONF_THRESH_HIGH:
        return {"x": x, "y": y, "conf": conf, "valid": True, "estimated": False}
    elif conf >= CONF_THRESH_MID:
        return {"x": x, "y": y, "conf": conf, "valid": True, "estimated": True}
    else:
        return {"x": x, "y": y, "conf": conf, "valid": False, "estimated": False}


def smooth_sequence(seq, prev_seq):
    if prev_seq is None or len(seq) != len(prev_seq):
        return seq
    out = []
    for new_pt, old_pt in zip(seq, prev_seq):
        if not new_pt["valid"]:
            out.append(old_pt)
        elif old_pt["valid"] and point_distance(new_pt, old_pt) > MAX_JUMP_PIXEL:
            out.append(old_pt)
        else:
            out.append(new_pt)
    return out


def extract_video_keypoints(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_list = []
    preview_imgs = []
    prev_kpts = None
    total_frames = 0
    bad_frames = 0
    continuous_bad = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            bad_frames += 1
            continuous_bad += 1
            if continuous_bad >= BAD_CONTINUOUS_FRAME:
                cap.release()
                pose.close()
                raise RuntimeError("连续多帧丢失人体，视频质量太差")
            frame_list.append(None)
            preview_imgs.append(frame.copy())
            continue

        continuous_bad = 0
        kpts = []
        for lm in res.pose_landmarks.landmark:
            kpts.append(filter_keypoint(lm.x * w, lm.y * h, lm.visibility))
        kpts = smooth_sequence(kpts, prev_kpts)
        prev_kpts = kpts

        out_img = frame.copy()
        mp_draw.draw_landmarks(
            out_img,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        frame_list.append(kpts)
        preview_imgs.append(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))

    cap.release()
    pose.close()

    if total_frames <= 0:
        raise RuntimeError("视频为空")

    bad_rate = bad_frames / total_frames
    ok_flag = bad_rate < BAD_RATIO_THRESHOLD

    return {
        "sequence": frame_list,
        "preview_images": preview_imgs,
        "total": total_frames,
        "bad": bad_frames,
        "bad_rate": bad_rate * 100,
        "ok": ok_flag
    }


def load_template_lib():
    global saved_templates
    saved_templates = dict()
    if TEMPLATE_FILE.exists():
        with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for entry in raw:
            saved_templates[entry["name"]] = entry
    for item in TAIJI_24:
        name = item["name"]
        if name not in saved_templates:
            saved_templates[name] = {
                "id": item["id"],
                "name": name,
                "keypoint_sequence": [],
                "note": "尚未采集"
            }


def save_template_lib():
    out_list = list(saved_templates.values())
    with open(TEMPLATE_FILE, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)


def save_one_template(name, seq):
    if name not in saved_templates:
        saved_templates[name] = {
            "id": -1,
            "name": name,
            "keypoint_sequence": [],
            "note": ""
        }
    saved_templates[name]["keypoint_sequence"] = seq
    saved_templates[name]["note"] = "已采集"
    save_template_lib()


# 启动加载
load_template_lib()
