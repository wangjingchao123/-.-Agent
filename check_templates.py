import json
from pathlib import Path

# 标准完整 24 式清单
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


def main():
    if not TEMPLATE_FILE.exists():
        print("❌ 文件不存在：taiji_template_lib.json")
        return

    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        lib = json.load(f)

    # 兼容 list 和 dict 两种存储格式
    if isinstance(lib, list):
        lib = {entry["name"]: entry for entry in lib}

    total = len(TAIJI_24)
    good = []
    bad_empty = []
    bad_short = []

    for item in TAIJI_24:
        name = item["name"]
        entry = lib.get(name, {})
        seq = entry.get("keypoint_sequence", [])
        n = len(seq)

        if n == 0:
            bad_empty.append((item["id"], name))
        elif n <= 3:
            bad_short.append((item["id"], name, n))
        else:
            good.append((item["id"], name, n))

    print("=" * 78)
    print(f"【严格校验】目标总数：{total}")
    print(f"✅ 有效模板（帧数 > 3）：{len(good)}")
    print(f"❌ 完全空模板：{len(bad_empty)}")
    print(f"⚠️ 太短（疑似保存失败）：{len(bad_short)}")
    print("=" * 78)

    if good:
        print("\n✅ 通过校验（有效）：")
        for idx, nm, frames in good:
            print(f"{idx:2d}｜{nm} → {frames:3d} 帧")

    if bad_empty:
        print("\n❌【空，必须重采】：")
        for idx, nm in bad_empty:
            print(f"{idx:2d}｜{nm}")

    if bad_short:
        print("\n⚠️【帧数极少，建议重采】：")
        for idx, nm, frames in bad_short:
            print(f"{idx:2d}｜{nm}，仅 {frames} 帧")

    if not bad_empty and not bad_short:
        print("\n🎉 全部通过！所有 24 式都有足量骨骼序列，可以进入比对教练模块。")
    else:
        print("\n❗ 有不合格项，回到采集界面重新上传保存。")


if __name__ == "__main__":
    main()