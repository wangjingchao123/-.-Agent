# upgrade.py
# 运行方式：本地直接执行 python upgrade.py
# 输入：taiji_keyframes.json
# 输出：upgrade_taiji.json
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# ===================== 三级风险分级 =====================
# 🔴 高危伤膝（半月板/剪切力） 🟠 风险（负荷偏高） 🟡 姿态提示（无急性损伤）

KNEE_L_HIGH = {
    "type": "knee_angle", "target": "left_knee", "min": 90,
    "msg": "🔴【高危伤膝】左膝角度小于90°，屈膝过深，极易磨损半月板！"
}
KNEE_R_HIGH = {
    "type": "knee_angle", "target": "right_knee", "min": 90,
    "msg": "🔴【高危伤膝】右膝角度小于90°，屈膝过深，极易磨损半月板！"
}
TOE_L_HIGH = {
    "type": "knee_over_toe", "side": "left", "threshold": 0.07,
    "msg": "🔴【高危伤膝】左膝向前超过脚尖，膝关节承受巨大剪切力！"
}
TOE_R_HIGH = {
    "type": "knee_over_toe", "side": "right", "threshold": 0.07,
    "msg": "🔴【高危伤膝】右膝向前超过脚尖，膝关节承受巨大剪切力！"
}
SUPPORT_KNEE_WARN = {
    "type": "knee_angle", "target": "support_knee", "min": 90,
    "msg": "🟠【风险】支撑腿屈膝角度小于90°，膝关节负荷偏高，老年人不宜蹲太深！"
}
SUPPORT_TOE_WARN = {
    "type": "knee_over_toe", "side": "right", "threshold": 0.07,
    "msg": "🟠【风险】支撑腿膝盖超过脚尖，单腿站立膝关节压力过大！"
}
NOSE_HIP_HIGH = {
    "type": "nose_hip_y", "threshold": 0.0,
    "msg": "🔴【高危风险】头部位置低于髋部，低头俯身过大，容易引发脑充血、高血压不适！"
}
NOSE_HIP_WARN = {
    "type": "nose_hip_y", "threshold": 0.35,
    "msg": "🟠【风险】上身过度前倾，腰椎受压过大，请保持立身中正！"
}

# ===================== 按招式定制安全规则 =====================
SAFETY_RULE_MAP = {
    # ---- 用户定制的高危招式（原样保留） ----
    "野马分鬃_左": [KNEE_L_HIGH, TOE_L_HIGH],
    "野马分鬃_右": [KNEE_R_HIGH, TOE_R_HIGH],
    "海底针": [NOSE_HIP_HIGH],
    "白鹤亮翅": [
        SUPPORT_KNEE_WARN,
        {
            "type": "hand_height_diff", "min_diff": 0.35,
            "msg": "🟡【姿态提示】两手高低差不足，白鹤亮翅定式形态不标准（无急性损伤风险）"
        }
    ],
    # ---- 补全：弓步类（前腿承重，膝盖高危） ----
    "左右搂膝拗步": [KNEE_L_HIGH, KNEE_R_HIGH, TOE_L_HIGH, TOE_R_HIGH],
    "左揽雀尾": [KNEE_L_HIGH, KNEE_R_HIGH, TOE_L_HIGH, TOE_R_HIGH],
    "右揽雀尾": [KNEE_L_HIGH, KNEE_R_HIGH, TOE_L_HIGH, TOE_R_HIGH],
    "单鞭（一）": [KNEE_L_HIGH, KNEE_R_HIGH, TOE_L_HIGH, TOE_R_HIGH],
    "单鞭（二）": [KNEE_L_HIGH, KNEE_R_HIGH, TOE_L_HIGH, TOE_R_HIGH],
    "双峰贯耳": [KNEE_L_HIGH, KNEE_R_HIGH, TOE_L_HIGH, TOE_R_HIGH],
    "转身搬拦捶": [KNEE_L_HIGH, KNEE_R_HIGH, TOE_L_HIGH, TOE_R_HIGH],
    "闪通臂": [KNEE_L_HIGH, KNEE_R_HIGH, TOE_L_HIGH, TOE_R_HIGH],
    # ---- 补全：独立/蹬脚类（单腿支撑，跌倒+膝盖风险） ----
    "左下势独立": [SUPPORT_KNEE_WARN, SUPPORT_TOE_WARN, NOSE_HIP_WARN],
    "右下势独立": [SUPPORT_KNEE_WARN, SUPPORT_TOE_WARN, NOSE_HIP_WARN],
    "右蹬脚": [SUPPORT_KNEE_WARN],
    "转身左蹬脚": [SUPPORT_KNEE_WARN],
    # ---- 补全：虚实步类（重心后坐，膝盖中度风险） ----
    "手挥琵琶": [KNEE_L_HIGH, KNEE_R_HIGH],
    "倒卷肱（标准）": [KNEE_L_HIGH, KNEE_R_HIGH],
    "高探马": [KNEE_L_HIGH, KNEE_R_HIGH],
    "如封似闭": [KNEE_L_HIGH, KNEE_R_HIGH],
    "云手": [KNEE_L_HIGH, KNEE_R_HIGH],
    # ---- 补全：穿梭（弓步+手上架，加姿态提示） ----
    "左右穿梭": [
        KNEE_L_HIGH, KNEE_R_HIGH, TOE_L_HIGH, TOE_R_HIGH,
        {
            "type": "hand_height_diff", "min_diff": 0.30,
            "msg": "🟡【姿态提示】一手上架一手前推高低差不足，动作幅度不够（无急性损伤风险）"
        }
    ],
    # ---- 补全：站立类（膝盖微弯即可，基础防护） ----
    "起势": [KNEE_L_HIGH, KNEE_R_HIGH],
    "十字手": [KNEE_L_HIGH, KNEE_R_HIGH],
    "收势": [KNEE_L_HIGH, KNEE_R_HIGH],
}

# 野马分鬃拆分为左/右两式（vector相同，规则分侧检测）
SPLIT_ACTION = {"左右野马分鬃"}


def main():
    INPUT_FILE = BASE_DIR / "taiji_keyframes.json"
    OUTPUT_FILE = BASE_DIR / "upgrade_taiji.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    new_template_list = []

    for item in raw_data["templates"]:
        tpl_name = item["template_name"]
        vec_1d = item["vector"]

        if tpl_name in SPLIT_ACTION:
            # 拆分为左式/右式，各自检测对应侧膝盖
            for side in ("左", "右"):
                new_name = f"野马分鬃_{side}"
                new_template_list.append({
                    "template_name": new_name,
                    "vector": vec_1d,
                    "similarity_threshold": 0.50,
                    "safety_checks": SAFETY_RULE_MAP.get(new_name, [])
                })
        else:
            new_template_list.append({
                "template_name": tpl_name,
                "vector": vec_1d,
                "similarity_threshold": 0.50,
                "safety_checks": SAFETY_RULE_MAP.get(tpl_name, [])
            })

    final_data = {
        "meta": {
            "version": "upgrade-v2.0",
            "note": "太极安全评估：三级风险分级(🔴高危伤膝/🟠风险/🟡姿态提示)，野马分鬃左右拆分，全招式覆盖"
        },
        "templates": new_template_list
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 升级完成！输出文件：{OUTPUT_FILE}")
    print(f"👉 原模板总数：{len(raw_data['templates'])}")
    print(f"👉 升级后模板总数：{len(new_template_list)}")
    # 统计规则覆盖
    no_rules = [t["template_name"] for t in new_template_list if not t["safety_checks"]]
    if no_rules:
        print(f"⚠️ 无安全规则的招式：{no_rules}")
    else:
        print("👉 全部招式均含安全规则 ✅")


if __name__ == "__main__":
    main()
