"""交叉比对验证：不同招式互比应得低分（区分度）"""
import sys
sys.path.insert(0, ".")
from taiji_score import load_templates_only, calc_dtw_score, TEMPLATE_FILE

lib = load_templates_only(TEMPLATE_FILE)
templates = lib["templates"]
names = sorted(templates)
print(f"模板数：{len(names)}")
print("=" * 70)

# 用 白鹤亮翅 的帧 去比 右下势独立（动作完全不同）
user_name = "白鹤亮翅"
target_name = "右下势独立"
user_seq = templates[user_name]["frames"]
tmpl = templates[target_name]
score, msg, dist = calc_dtw_score(user_seq, tmpl["frames"], target_name, tmpl["d_bad"])
print(f"[交叉] {user_name} vs {target_name}")
print(f"  距离={dist:.2f} d_bad={tmpl['d_bad']:.2f} 比={dist/tmpl['d_bad']:.2f} 分数={score}")
assert score == 0.0, f"交叉比对应0分，实际{score}"

# 自比对（白鹤亮翅 vs 白鹤亮翅）应100分
tmpl2 = templates[user_name]
score2, msg2, dist2 = calc_dtw_score(tmpl2["frames"], tmpl2["frames"], user_name, tmpl2["d_bad"])
print(f"[自比] {user_name} vs {user_name}")
print(f"  距离={dist2:.2f} d_bad={tmpl2['d_bad']:.2f} 分数={score2}")
assert score2 == 100.0, f"自比对应100分，实际{score2}"

# 再测一个：起势 vs 收势（动作相似但不同）
n1, n2 = "起势", "收势"
if n1 in templates and n2 in templates:
    t1, t2 = templates[n1], templates[n2]
    s3, m3, d3 = calc_dtw_score(t1["frames"], t2["frames"], n2, t2["d_bad"])
    print(f"[相似] {n1} vs {n2}")
    print(f"  距离={d3:.2f} d_bad={t2['d_bad']:.2f} 比={d3/t2['d_bad']:.2f} 分数={s3}")

print("=" * 70)
print("全部通过")