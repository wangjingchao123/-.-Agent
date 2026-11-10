"""验证 k=2.2 缩放后的评分口径：标准动作+噪声应得约55分"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import taiji_score as ts


def main():
    lib = ts.load_templates_only(ts.TEMPLATE_FILE)
    T = lib["templates"]

    name = "白鹤亮翅"
    tmpl_raw = T[name]["frames"]
    d_bad = T[name]["d_bad"]
    print(f"招式：{name}，d_bad={d_bad:.4f}，k=2.2 → max_dist={2.2*d_bad:.2f}")

    # 1) 自比对 → 100分
    r3 = ts.calc_dtw_score(tmpl_raw, tmpl_raw, name, d_bad)
    print(f"\n1) 自比对：分数={r3[0]}，距离={r3[2]:.2f}")
    assert r3[0] == 100.0

    # 2) 原始坐标加噪（σ=肩宽×0.028，与d_bad口径一致）→ 应得约55分
    rng = np.random.default_rng(42)
    sample = tmpl_raw[0]
    shoulder_w = abs(sample[11]["x"] - sample[12]["x"])
    sigma_pixel = shoulder_w * 0.028
    print(f"\n2) 原始坐标加噪：肩宽≈{shoulder_w:.0f}px，σ≈{sigma_pixel:.2f}px")

    noisy_raw = []
    for frame in tmpl_raw:
        nf = []
        for pt in frame:
            nx = pt["x"] + rng.normal(0, sigma_pixel)
            ny = pt["y"] + rng.normal(0, sigma_pixel)
            nf.append({"idx": pt["idx"], "x": nx, "y": ny,
                       "z": pt["z"], "visibility": pt["visibility"]})
        noisy_raw.append(nf)

    r = ts.calc_dtw_score(noisy_raw, tmpl_raw, name, d_bad)
    ratio = r[2] / d_bad
    print(f"   加噪评分：分数={r[0]}，距离={r[2]:.2f}，距离/d_bad={ratio:.2f}")
    print(f"   期望：距离≈d_bad(比例≈1) → 分数≈55（100*(1-1/2.2)）")
    assert r[0] > 30, f"加噪后应得30分以上，实际{r[0]}（评分仍过严）"

    # 3) 交叉比对 → 0分
    r2 = ts.calc_dtw_score(T["云手"]["frames"], tmpl_raw, name, d_bad)
    print(f"\n3) 交叉比对 云手vs{name}：分数={r2[0]}，距离={r2[2]:.2f}，比={r2[2]/d_bad:.2f}")
    assert r2[0] == 0.0

    print("\n全部通过")


if __name__ == "__main__":
    main()