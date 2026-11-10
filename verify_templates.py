import json
import math
from pathlib import Path

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

TEMPLATE_FILE = Path("taiji_template_lib.json")


def check_frame(frame):
    """检查单帧数据完整性，返回 (ok, 问题列表)"""
    problems = []
    if frame is None:
        return False, ["帧为None（丢失人体）"]
    if not isinstance(frame, list):
        return False, [f"帧不是list: {type(frame)}"]
    if len(frame) not in (17, 33):
        problems.append(f"关键点数={len(frame)}（应为17或33）")
    for i, pt in enumerate(frame):
        if not isinstance(pt, dict):
            problems.append(f"点{i}不是dict")
            continue
        for key in ("x", "y", "conf"):
            if key not in pt:
                problems.append(f"点{i}缺字段{key}")
                continue
            v = pt[key]
            if v is None:
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                problems.append(f"点{i}.{key}=NaN/Inf")
            elif key in ("x", "y") and v < 0:
                problems.append(f"点{i}.{key}={v}<0（负坐标）")
    return len(problems) == 0, problems


def main():
    if not TEMPLATE_FILE.exists():
        print("❌ 文件不存在：taiji_template_lib.json")
        return

    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        lib = json.load(f)

    if isinstance(lib, list):
        lib = {entry["name"]: entry for entry in lib}

    print("=" * 78)
    print("【深度校验】逐帧检查关键点数据完整性")
    print("=" * 78)

    all_ok = True
    for item in TAIJI_24:
        name = item["name"]
        entry = lib.get(name, {})
        seq = entry.get("keypoint_sequence", [])
        n = len(seq)

        if n == 0:
            print(f"❌ [{item['id']:2d}] {name} → 空模板")
            all_ok = False
            continue
        if n <= 3:
            print(f"⚠️ [{item['id']:2d}] {name} → 仅{n}帧，疑似保存失败")
            all_ok = False
            continue

        # 逐帧校验
        bad_frames = 0
        none_frames = 0
        sample_problems = []
        for fi, frame in enumerate(seq):
            ok, problems = check_frame(frame)
            if not ok:
                bad_frames += 1
                if frame is None:
                    none_frames += 1
                if len(sample_problems) < 2:
                    sample_problems.append(f"帧{fi}: {'; '.join(problems[:3])}")

        if bad_frames == 0:
            print(f"✅ [{item['id']:2d}] {name} → {n}帧，数据完整")
        else:
            print(f"⚠️ [{item['id']:2d}] {name} → {n}帧，异常帧{bad_frames}（None帧{none_frames}）")
            for p in sample_problems:
                print(f"      └ {p}")
            if bad_frames / n > 0.5:
                all_ok = False

    print("=" * 78)
    if all_ok:
        print("🎉 深度校验通过！所有模板数据结构完整，可放心用于比对教练模块。")
    else:
        print("❗ 存在严重问题项，建议回到采集界面重新上传保存。")


if __name__ == "__main__":
    main()