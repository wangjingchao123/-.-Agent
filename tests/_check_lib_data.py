import json
import sys

# 检查 taiji_template_lib.json
with open("taiji_template_lib.json", "r", encoding="utf-8") as f:
    lib = json.load(f)

print("=== taiji_template_lib.json ===")
print("总条目:", len(lib))
for entry in lib:
    seq = entry.get("keypoint_sequence", [])
    note = entry.get("note", "")
    if len(seq) > 0:
        print(f"  [{entry['id']:2d}] {entry['name']}: {len(seq)}帧, note={note}")
    else:
        print(f"  [{entry['id']:2d}] {entry['name']}: 空, note={note}")

# 检查 taiji_templates/左右野马分鬃.json
print()
print("=== taiji_templates/左右野马分鬃.json ===")
with open("taiji_templates/左右野马分鬃.json", "r", encoding="utf-8") as f:
    d = json.load(f)
seq = d.get("keypoint_sequence", [])
print("动作名:", d.get("action_name"))
print("帧数:", len(seq))
if len(seq) > 0:
    print("第1帧关键点数:", len(seq[0]))
    print("第1帧第1个点:", seq[0][0])