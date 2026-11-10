import json
from pathlib import Path

# 实际存储的点字段（taiji_template_lib.py filter_keypoint 生成）
REQUIRED_KEYS = {"x", "y", "conf", "valid", "estimated"}


def load_templates_only(json_path: str):
    """
    比赛版：只读加载，不回写原始文件
    - 如果有重复"倒卷肱"，自动只保留一份
    - 校验帧数、每帧必须 33 点，字段完整
    - readonly=True，禁止写回
    """
    file = Path(json_path)

    if not file.exists():
        raise FileNotFoundError(f"找不到模板文件：{json_path}")

    try:
        with file.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError("JSON 文件损坏，无法解析！检查模板文件")

    # 实际文件顶层是 list[{id, name, keypoint_sequence, note}]，统一转成 {name: 序列}
    raw_templates = {}
    if isinstance(raw_data, list):
        for entry in raw_data:
            if isinstance(entry, dict) and "name" in entry:
                raw_templates[entry["name"]] = entry.get("keypoint_sequence", [])
    elif isinstance(raw_data, dict):
        raw_templates = dict(raw_data)
    else:
        raise ValueError("模板文件格式不对：顶层既不是列表也不是字典")

    invalid_items = []
    duplicate_items = []

    # 去重处理（只用于内存展示，不改原始文件）
    keep_names = set()
    cleaned_raw = {}

    for name, seq in raw_templates.items():
        norm_name = name.strip()
        # "倒卷肱"重复处理
        if norm_name in ("倒卷肱", "左右倒卷肱"):
            key = "倒卷肱（标准）"
            if key in keep_names:
                duplicate_items.append(name)
                continue
            keep_names.add(key)
            cleaned_raw[key] = seq
        else:
            if norm_name in keep_names:
                duplicate_items.append(name)
                continue
            keep_names.add(norm_name)
            cleaned_raw[norm_name] = seq

    templates = {}

    for name, seq in cleaned_raw.items():
        if not isinstance(seq, list):
            invalid_items.append((name, "不是帧序列（非列表）"))
            continue

        total_frames = len(seq)
        # None 帧 = 采集时丢人的帧，属正常现象，单独统计
        none_frames = sum(1 for f in seq if f is None)
        valid_frames = total_frames - none_frames

        # 比赛最低阈值：有效帧不少于 15 帧
        if valid_frames < 15:
            invalid_items.append(
                (name, f"有效帧不足，仅 {valid_frames} 帧（总{total_frames}，None帧{none_frames}）")
            )
            continue

        ok_frames = True
        for idx_frame, frame in enumerate(seq):
            if frame is None:
                continue  # None 帧跳过，不算校验失败
            if not isinstance(frame, list) or len(frame) != 33:
                got = len(frame) if isinstance(frame, list) else "非列表"
                invalid_items.append(
                    (name, f"第 {idx_frame} 帧：关键点数量不对（{got}，应为 33）")
                )
                ok_frames = False
                break

            for point in frame:
                if not REQUIRED_KEYS.issubset(point.keys()):
                    missing = REQUIRED_KEYS - set(point.keys())
                    invalid_items.append(
                        (
                            name,
                            f"第 {idx_frame} 帧：缺少必要字段 {sorted(missing)}（必须包含 {sorted(REQUIRED_KEYS)}）",
                        )
                    )
                    ok_frames = False
                    break

            if not ok_frames:
                break

        if ok_frames:
            templates[name] = seq

    result = {
        "readonly": True,
        "templates": templates,
        "invalid": invalid_items,
        "duplicates": duplicate_items,
        "original_names": list(raw_templates.keys()),
    }
    return result


def print_report(res):
    print("=" * 72)
    print("【模板库只读加载报告｜比赛版】")
    print("=" * 72)
    print(f"只读标记：{'启用（禁止写入）' if res['readonly'] else '关闭'}")
    print(f"原始文件内总条目：{len(res['original_names'])}")
    print(f"内存去重后可用：{len(res['templates'])}")
    print(f"校验失败 / 不可用：{len(res['invalid'])}")
    print(f"重复（已跳过）：{len(res['duplicates'])}")
    print()

    if res["duplicates"]:
        print("⚠️ 重复条目（内存跳过，原文件不动）：")
        for dup in res["duplicates"]:
            print(f"   • {dup}")
        print()

    if res["templates"]:
        print("✅ 通过校验，可以用于比对：")
        for name, frames in sorted(res["templates"].items()):
            none_n = sum(1 for f in frames if f is None)
            extra = f"，None帧 {none_n}" if none_n else ""
            print(f"   • {name} —— {len(frames)} 帧{extra}")
    else:
        print("❌ 没有任何可用模板！")

    if res["invalid"]:
        print("\n❌ 校验失败（跳过）：")
        for name, reason in res["invalid"]:
            print(f"   • {name}：{reason}")

    print("\n【示例：取第一个可用招式，打印前两帧，只展示前 5 点】")
    if res["templates"]:
        first_key = sorted(res["templates"])[0]
        sample_seq = res["templates"][first_key]
        # 找第一个非 None 帧
        sample_frame = next((f for f in sample_seq if f is not None), None)
        print(f"   招式：{first_key}")
        print(f"   总帧数：{len(sample_seq)}")
        if sample_frame:
            print("   > 首个有效帧：")
            for p in sample_frame[:5]:
                print(
                    f"        x:{p['x']:7.1f} | y:{p['y']:7.1f} | conf:{p['conf']:.3f} | valid:{p['valid']}"
                )
    else:
        print("   无可用样本")

    print("\n【重要提醒】")
    print("   > 只读模式：本脚本只读取，绝不修改模板文件！")
    print("   > 如果要永久清理重复项：只用开发后台手动删，比赛演示不要碰原文件。")


if __name__ == "__main__":
    json_file = "taiji_template_lib.json"
    try:
        data = load_templates_only(json_file)
        print_report(data)
    except Exception as e:
        print(f"\n❌ 加载失败：{e}")