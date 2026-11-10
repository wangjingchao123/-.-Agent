import sys
sys.path.insert(0, ".")
import taiji_base as tb
import taiji_coach_camera as tc

print("=== taiji_base ===")
names = tb.get_template_names()
print("template count:", len(names))
print("first3:", names[:3])

# 测试归一化 + 比对
lib = tb.load_templates()
name = names[0]
raw = lib[name]["keypoint_sequence"]
norm = tb.normalize_sequence(raw)
print(f"normalize {name}: {len(raw)} raw -> {len(norm)} norm frames")
print("norm frame shape:", norm[0].shape if norm else "empty")

# 自比对（模板和自己比，应该接近100分）
score, dist = tb.compare_sequences(norm, norm)
print(f"self-compare score: {score}, dist: {dist}")

print()
print("=== taiji_coach_camera ===")
print("demo title:", tc.demo.title)
print("status:", tc.on_select_action(name)[:50])