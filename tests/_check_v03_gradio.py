"""验证竞赛提交版：真实模板加载 + 手写DTW量级 vs BAD_DIST/DANGER_DIST"""
import sys
sys.path.insert(0, ".")
import numpy as np
import v03_gradio_final_verified as m

# 1) 真实模板加载（替换占位随机）
lib = m.TEMPLATE_LIB
print(f"1) 真实模板加载：{len(lib)} 个")
assert len(lib) == 24, f"应24个，实际{len(lib)}"
name = "白鹤亮翅"
tmpl = lib[name]
print(f"   {name}：shape={tmpl.shape}（期望(120,99)）")
assert tmpl.shape == (m.TARGET_SEQ_LEN, m.NUM_KEYPOINTS * 3)

# 2) 自比对 → 距离≈0 → 100分
dist_self = m.dtw_distance(tmpl, tmpl)
score_self = m.score_from_distance(dist_self)
print(f"2) 自比对：dist={dist_self:.6f}，score={score_self}")
assert score_self == 100.0, f"自比对应100，实际{score_self}"

# 3) 交叉比对 → dist >> DANGER_DIST → 0分
tmpl_b = lib["右下势独立"]
dist_cross = m.dtw_distance(tmpl, tmpl_b)
score_cross = m.score_from_distance(dist_cross)
print(f"3) 交叉比对 白鹤亮翅vs右下势独立：dist={dist_cross:.4f}，score={score_cross}")
print(f"   BAD_DIST={m.BAD_DIST} DANGER_DIST={m.DANGER_DIST}")
assert score_cross == 0.0, f"交叉比对应0，实际{score_cross}"

# 4) 噪声扰动 → 中间分数（验证量级区分度）
rng = np.random.default_rng(42)
noisy = tmpl + rng.normal(0, 0.01, tmpl.shape)
dist_noise = m.dtw_distance(noisy, tmpl)
score_noise = m.score_from_distance(dist_noise)
print(f"4) 噪声扰动(σ=0.01)：dist={dist_noise:.4f}，score={score_noise:.1f}")

# 5) 评分函数边界
assert m.score_from_distance(0.0) == 100.0
assert m.score_from_distance(m.DANGER_DIST) == 0.0
assert m.score_from_distance(999) == 0.0
print(f"5) 评分边界：dist=0→100，dist>=DANGER_DIST→0 ✓")

print("\n全部通过")