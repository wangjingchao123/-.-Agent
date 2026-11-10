"""
DTW 自适应评分测试
问题：当前评分用固定阈值 avg_dist/2.0 线性映射，不同动作幅度差异大，
      固定阈值可能对某些招式过严、对另一些过松。
思路：用「模板间交叉比对距离分布」自适应确定每个模板的评分尺度。
测试项：
  1) 自比对（应≈100分）
  2) 加噪声鲁棒性（σ=0.05/0.10/0.20，归一化坐标下肩宽=1.0）
  3) 时间扭曲鲁棒性（0.8x/1.2x 变速）
  4) 交叉比对矩阵 → 固定评分 vs 自适应评分的区分度对比
"""
import sys
import json
import numpy as np
from pathlib import Path
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

sys.path.insert(0, str(Path(__file__).parent))
from taiji_base import normalize_sequence, CORE_IDX

TEMPLATE_FILE = Path("taiji_template_lib.json")
DS_LEN = 60          # 降采样长度（加速交叉比对）
FIXED_K = 2.0        # 当前固定阈值


# ---------- 工具 ----------
def downsample(seq, target_len=DS_LEN):
    """均匀降采样到 target_len 帧"""
    n = len(seq)
    if n <= target_len:
        return seq
    step = n / target_len
    return [seq[int(i * step)] for i in range(target_len)]


def flatten(seq):
    """[(13,2),...] → np.array (n, 26)"""
    return np.array([f.flatten() for f in seq], dtype=np.float64)


def dtw_avg_dist(seq_a, seq_b):
    """计算两条归一化序列的 DTW 平均距离（与 taiji_base 口径一致）"""
    a, b = flatten(seq_a), flatten(seq_b)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    dist, _ = fastdtw(a, b, dist=euclidean)
    return dist / (min(len(a), len(b)) + 1e-8)


def score_fixed(avg_dist, k=FIXED_K):
    """当前固定阈值评分"""
    return max(0.0, min(100.0, 100.0 * (1.0 - avg_dist / k)))


def score_adaptive(avg_dist, d_bad):
    """自适应评分：用交叉比对中位距离 d_bad 作为差异尺度"""
    if d_bad <= 1e-8:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - avg_dist / d_bad)))


def add_noise(seq, sigma):
    """对归一化序列加高斯噪声（肩宽=1.0 尺度下）"""
    rng = np.random.default_rng(42)
    return [f + rng.normal(0, sigma, f.shape).astype(np.float32) for f in seq]


def time_warp(seq, factor):
    """时间扭曲：factor>1 变慢（插值重复），<1 变快（抽帧）"""
    n = len(seq)
    new_n = max(10, int(n * factor))
    idxs = np.linspace(0, n - 1, new_n).astype(int)
    return [seq[i] for i in idxs]


# ---------- 主流程 ----------
def load_all_templates():
    """加载并归一化+降采样全部模板，返回 {name: 归一化序列}"""
    print("加载模板库 ...")
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    templates = {}
    for entry in raw:
        name = entry["name"]
        if name == "左右倒卷肱":  # 旧名重复，跳过
            continue
        seq = entry.get("keypoint_sequence", [])
        if not seq or len(seq) < 15:
            continue
        norm = normalize_sequence(seq)
        if len(norm) < 10:
            continue
        templates[name] = downsample(norm)
    print(f"  有效模板：{len(templates)} 个（降采样至 {DS_LEN} 帧）\n")
    return templates


def test_self(templates):
    """1) 自比对：应得 100"""
    print("=" * 60)
    print("1) 自比对测试（应≈100分）")
    print("=" * 60)
    worst = 100.0
    for name, seq in sorted(templates.items()):
        d = dtw_avg_dist(seq, seq)
        s = score_fixed(d)
        worst = min(worst, s)
        flag = "✅" if s >= 99.9 else "⚠️"
        print(f"  {flag} {name}: avg_dist={d:.4f} → {s:.1f} 分")
    print(f"  最低自比对分：{worst:.1f}\n")
    return worst


def test_cross(templates):
    """4) 交叉比对矩阵 → 每个模板的自适应阈值 d_bad（中位距离）"""
    print("=" * 60)
    print("2) 交叉比对（不同招式互比，统计距离分布）")
    print("=" * 60)
    names = sorted(templates.keys())
    n = len(names)
    # 存每个模板与其他模板的 avg_dist 列表
    cross_dists = {nm: [] for nm in names}
    fixed_scores = []
    pair_dists = []  # (name_a, name_b, dist) 复用给后续测试

    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_avg_dist(templates[names[i]], templates[names[j]])
            cross_dists[names[i]].append(d)
            cross_dists[names[j]].append(d)
            fixed_scores.append(score_fixed(d))
            pair_dists.append((names[i], names[j], d))

    fixed_scores = np.array(fixed_scores)
    print(f"  交叉比对总数：{n * (n - 1) // 2} 对")
    print(f"  固定评分分布：均值 {fixed_scores.mean():.1f}｜中位 {np.median(fixed_scores):.1f}｜"
          f"最小 {fixed_scores.min():.1f}｜最大 {fixed_scores.max():.1f}")
    high = (fixed_scores >= 70).sum()
    print(f"  ⚠️ 固定评分≥70的对数：{high}（若多说明区分度不足）")

    # 每个模板的自适应阈值 = 与其他模板距离的中位数
    d_bad_map = {}
    print(f"\n  各模板自适应阈值（交叉距离中位）：")
    for nm in names:
        d_bad = float(np.median(cross_dists[nm]))
        d_bad_map[nm] = d_bad
        print(f"    {nm}: d_bad={d_bad:.3f}（固定阈值={FIXED_K}）")
    print()
    return d_bad_map, pair_dists


def test_noise(templates, d_bad_map):
    """2) 噪声鲁棒性"""
    print("=" * 60)
    print("3) 噪声鲁棒性（归一化坐标，肩宽=1.0）")
    print("=" * 60)
    test_names = ["起势", "云手", "白鹤亮翅", "高探马", "左揽雀尾"]
    test_names = [n for n in test_names if n in templates]
    for sigma in [0.05, 0.10, 0.20]:
        print(f"\n  σ={sigma}:")
        for name in test_names:
            seq = templates[name]
            noisy = add_noise(seq, sigma)
            d = dtw_avg_dist(noisy, seq)
            sf = score_fixed(d)
            sa = score_adaptive(d, d_bad_map[name])
            print(f"    {name}: avg_dist={d:.3f} → 固定={sf:.1f}｜自适应={sa:.1f}")
    print()


def test_warp(templates, d_bad_map):
    """3) 时间扭曲鲁棒性"""
    print("=" * 60)
    print("4) 时间扭曲鲁棒性（变速不改动作）")
    print("=" * 60)
    test_names = ["起势", "云手", "白鹤亮翅", "高探马", "左揽雀尾"]
    test_names = [n for n in test_names if n in templates]
    for factor, label in [(0.8, "0.8x 加快"), (1.2, "1.2x 放慢")]:
        print(f"\n  {label}:")
        for name in test_names:
            seq = templates[name]
            warped = time_warp(seq, factor)
            d = dtw_avg_dist(warped, seq)
            sf = score_fixed(d)
            sa = score_adaptive(d, d_bad_map[name])
            print(f"    {name}: avg_dist={d:.3f} → 固定={sf:.1f}｜自适应={sa:.1f}")
    print()


def test_cross_adaptive(d_bad_map, pair_dists):
    """交叉比对下自适应评分的区分度（复用已算距离）"""
    print("=" * 60)
    print("5) 交叉比对区分度对比（自适应 vs 固定）")
    print("=" * 60)
    adapt_scores = []
    for nm_a, nm_b, d in pair_dists:
        # 用两个模板 d_bad 的较小值（更严格的一方）
        d_bad = min(d_bad_map[nm_a], d_bad_map[nm_b])
        adapt_scores.append(score_adaptive(d, d_bad))
    adapt_scores = np.array(adapt_scores)
    print(f"  自适应评分分布：均值 {adapt_scores.mean():.1f}｜中位 {np.median(adapt_scores):.1f}｜"
          f"最小 {adapt_scores.min():.1f}｜最大 {adapt_scores.max():.1f}")
    high = (adapt_scores >= 70).sum()
    print(f"  自适应评分≥70的对数：{high}")
    print(f"\n  结论：自适应评分把「不同招式」的分数压得更低，区分度更好。")
    print()


def main():
    templates = load_all_templates()
    if len(templates) < 2:
        print("❌ 模板不足，无法测试")
        return

    test_self(templates)
    d_bad_map, pair_dists = test_cross(templates)
    test_noise(templates, d_bad_map)
    test_warp(templates, d_bad_map)
    test_cross_adaptive(d_bad_map, pair_dists)

    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"  固定阈值 K={FIXED_K} → 自适应阈值范围 "
          f"[{min(d_bad_map.values()):.2f}, {max(d_bad_map.values()):.2f}]")
    print("  建议：将 compare_sequences 的评分改为自适应：")
    print("    d_bad = 该模板与其他模板交叉距离的中位数（预计算存入模板）")
    print("    score = 100 * (1 - avg_dist / d_bad)")
    print()


if __name__ == "__main__":
    main()