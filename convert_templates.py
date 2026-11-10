"""
模板格式转换脚本
将 taiji_template_lib.json（list格式，像素坐标，conf/valid/estimated字段）
转换为比赛版格式（dict格式，含d_bad，idx/x/y/z/visibility字段）

转换规则：
- 顶层 list[{id,name,keypoint_sequence,note}] → dict{name:{frames,d_bad}}
- keypoint_sequence → frames（去掉None帧）
- 每点 {x,y,conf,valid,estimated} → {idx,x,y,z:0.0,visibility:conf}
- d_bad 从 d_bad_cache.json 读取（缺失则用兜底值）
- 去重："左右倒卷肱"跳过，"倒卷肱"→"倒卷肱（标准）"
- 只读原始文件，输出到新文件 taiji_templates_final.json
"""
import json
from pathlib import Path

SRC_FILE = "taiji_template_lib.json"
D_BAD_FILE = "d_bad_cache.json"
OUT_FILE = "taiji_templates_final.json"
FALLBACK_D_BAD = 12.0


def convert_point(idx, pt):
    """单点格式转换 {x,y,conf,valid,estimated} → {idx,x,y,z,visibility}"""
    return {
        "idx": idx,
        "x": pt["x"],
        "y": pt["y"],
        "z": 0.0,                      # 原始采集未存z，填0
        "visibility": pt["conf"],      # conf 即 MediaPipe visibility
    }


def convert_frame(frame):
    """单帧转换：33点列表 → 新格式列表；None帧返回None"""
    if frame is None:
        return None
    return [convert_point(i, pt) for i, pt in enumerate(frame)]


def main():
    # 1) 加载原始模板
    print("加载原始模板 ...")
    with open(SRC_FILE, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    print(f"  原始条目：{len(raw_list)} 条")

    # 2) 加载 d_bad 缓存
    d_bad_map = {}
    if Path(D_BAD_FILE).exists():
        with open(D_BAD_FILE, "r", encoding="utf-8") as f:
            d_bad_map = json.load(f)
        print(f"  d_bad 缓存：{len(d_bad_map)} 个")

    # 3) 转换（去重 + 格式转换 + 去None帧）
    out = {}
    skipped_dup = []
    for entry in raw_list:
        name = entry["name"].strip()

        # 去重：倒卷肱/左右倒卷肱 → 倒卷肱（标准）
        if name in ("倒卷肱", "左右倒卷肱"):
            key = "倒卷肱（标准）"
            if key in out:
                skipped_dup.append(name)
                continue
        else:
            key = name
            if key in out:
                skipped_dup.append(name)
                continue

        seq = entry.get("keypoint_sequence", [])
        # 转换每帧，去掉 None 帧
        frames = []
        none_count = 0
        for frame in seq:
            converted = convert_frame(frame)
            if converted is None:
                none_count += 1
            else:
                frames.append(converted)

        d_bad = d_bad_map.get(key, FALLBACK_D_BAD)

        out[key] = {
            "frames": frames,
            "d_bad": round(float(d_bad), 4),
        }
        print(f"  ✅ {key}：{len(frames)} 帧（去除None {none_count}），d_bad={d_bad:.4f}")

    if skipped_dup:
        print(f"  ⚠️ 跳过重复：{skipped_dup}")

    # 4) 输出
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    size_mb = Path(OUT_FILE).stat().st_size / 1024 / 1024
    print(f"\n✅ 转换完成：{len(out)} 个模板 → {OUT_FILE}（{size_mb:.1f} MB）")
    print("   原始文件 taiji_template_lib.json 未修改")


if __name__ == "__main__":
    main()