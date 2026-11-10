# -*- coding: utf-8 -*-
"""从 taiji_keyframes.json 生成 upgrade_taiji.json（添加相似度阈值+安全检查规则）"""
import json

# ---------- 通用安全规则生成器 ----------
def knee(target, min_deg, msg):
    return {"type": "knee_angle", "target": target, "min": min_deg, "msg": msg}

def toe(side, th, msg):
    return {"type": "knee_over_toe", "side": side, "threshold": th, "msg": msg}

def nose_hip(th, msg):
    return {"type": "nose_hip_y", "threshold": th, "msg": msg}

def hand_diff(min_diff, msg):
    return {"type": "hand_height_diff", "min_diff": min_diff, "msg": msg}

KNEE_L_WARN = "⚠️左膝弯曲过度（角度偏低），有半月板损伤风险，请注意膝盖不超过脚尖、弓步不要过低"
KNEE_R_WARN = "⚠️右膝弯曲过度（角度偏低），有半月板损伤风险，请注意膝盖不超过脚尖、弓步不要过低"
TOE_L_WARN = "⚠️左膝超过脚尖，膝关节压力过大，易损伤半月板，请收髋坐胯"
TOE_R_WARN = "⚠️右膝超过脚尖，膝关节压力过大，易损伤半月板，请收髋坐胯"

# ---------- 按招式类别定义安全规则 ----------
# 弓步类（前腿弯曲，min角度110°）
bow_stance = [
    knee("left_knee", 110, KNEE_L_WARN),
    knee("right_knee", 110, KNEE_R_WARN),
    toe("left", 0.05, TOE_L_WARN),
    toe("right", 0.05, TOE_R_WARN),
]

# 站立类（膝盖微弯，min角度150°）
standing = [
    knee("left_knee", 150, KNEE_L_WARN),
    knee("right_knee", 150, KNEE_R_WARN),
    toe("left", 0.05, TOE_L_WARN),
    toe("right", 0.05, TOE_R_WARN),
]

# 虚步类（重心在后腿，min角度125°）
empty_stance = [
    knee("left_knee", 125, KNEE_L_WARN),
    knee("right_knee", 125, KNEE_R_WARN),
    toe("left", 0.05, TOE_L_WARN),
    toe("right", 0.05, TOE_R_WARN),
]

# 仆步/独立类（下蹲较深但保护膝盖，min角度95°）
crouch_stance = [
    knee("left_knee", 95, KNEE_L_WARN),
    knee("right_knee", 95, KNEE_R_WARN),
    toe("left", 0.05, TOE_L_WARN),
    toe("right", 0.05, TOE_R_WARN),
    nose_hip(0.35, "⚠️上身过度前倾，腰椎受压过大，请保持立身中正"),
]

# ---------- 每个招式的专属规则 ----------
SAFETY_MAP = {
    "起势": standing,
    "左右野马分鬃": bow_stance,
    "白鹤亮翅": [
        knee("support_knee", 140, "⚠️支撑腿（右腿）弯曲过度，单腿站立不稳，有跌倒和膝关节损伤风险"),
        knee("left_knee", 125, KNEE_L_WARN),
        toe("right", 0.05, TOE_R_WARN),
        hand_diff(0.10, "⚠️两手高度差不足，白鹤亮翅要求一手上托一手下按，请加大两臂展开幅度"),
    ],
    "左右搂膝拗步": bow_stance,
    "手挥琵琶": empty_stance,
    "倒卷肱（标准）": empty_stance,
    "左揽雀尾": bow_stance,
    "右揽雀尾": bow_stance,
    "单鞭（一）": bow_stance,
    "云手": [
        knee("left_knee", 130, KNEE_L_WARN),
        knee("right_knee", 130, KNEE_R_WARN),
        toe("left", 0.05, TOE_L_WARN),
        toe("right", 0.05, TOE_R_WARN),
    ],
    "单鞭（二）": bow_stance,
    "高探马": empty_stance,
    "右蹬脚": [
        knee("support_knee", 135, "⚠️支撑腿（右腿）弯曲过度，单腿站立不稳，有跌倒风险"),
        toe("right", 0.05, TOE_R_WARN),
    ],
    "双峰贯耳": bow_stance,
    "转身左蹬脚": [
        knee("support_knee", 135, "⚠️支撑腿弯曲过度，单腿站立不稳，有跌倒风险"),
        toe("right", 0.05, TOE_R_WARN),
    ],
    "左下势独立": crouch_stance,
    "右下势独立": crouch_stance,
    "左右穿梭": bow_stance + [
        hand_diff(0.12, "⚠️一手上架一手前推高度差不足，请加大动作幅度"),
    ],
    "海底针": [
        knee("left_knee", 125, KNEE_L_WARN),
        knee("right_knee", 125, KNEE_R_WARN),
        toe("left", 0.05, TOE_L_WARN),
        toe("right", 0.05, TOE_R_WARN),
        nose_hip(0.45, "⚠️俯身过度，腰椎与腘绳肌过度牵拉，老年人请量力而行、不要强求下插深度"),
        hand_diff(0.15, "⚠️两手高度差不足，海底针要求一手下插一手上提，请保持对拉拔长"),
    ],
    "闪通臂": bow_stance + [
        hand_diff(0.10, "⚠️一手上架一手前推高度差不足，请加大动作幅度"),
    ],
    "转身搬拦捶": bow_stance,
    "如封似闭": [
        knee("left_knee", 120, KNEE_L_WARN),
        knee("right_knee", 120, KNEE_R_WARN),
        toe("left", 0.05, TOE_L_WARN),
        toe("right", 0.05, TOE_R_WARN),
    ],
    "十字手": standing,
    "收势": standing,
}

# ---------- 相似度阈值 ----------
# 复杂动作阈值稍低，简单站立稍高
COMPLEX_MOVES = {"左下势独立", "右下势独立", "海底针", "右蹬脚", "转身左蹬脚", "白鹤亮翅"}
SIMPLE_MOVES = {"起势", "收势", "十字手"}

def get_threshold(name):
    if name in COMPLEX_MOVES:
        return 0.50
    if name in SIMPLE_MOVES:
        return 0.60
    return 0.55

# ---------- 主流程 ----------
with open("taiji_keyframes.json", "r", encoding="utf-8") as f:
    src = json.load(f)

out_templates = []
for tpl in src["templates"]:
    name = tpl["template_name"]
    out_templates.append({
        "template_name": name,
        "vector": tpl["vector"],
        "similarity_threshold": get_threshold(name),
        "safety_checks": SAFETY_MAP.get(name, bow_stance),
    })

out = {"meta": {"version": "upgrade-v1.0", "note": "太极二十四式升级版：按招式定制安全检查规则（损伤优先）"}, "templates": out_templates}

with open("upgrade_taiji.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print(f"OK: upgrade_taiji.json generated, {len(out_templates)} templates")
for t in out_templates:
    n_rules = len(t["safety_checks"])
    print(f"  {t['template_name']}: threshold={t['similarity_threshold']}, rules={n_rules}")