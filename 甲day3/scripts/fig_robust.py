"""
稳健性检验出图（读 json，不重跑任何计算）      [甲 · 验证脚本]

产物：figs/数据扰动稳健性图，（论文第6章 问题2-3·稳健性检验）.png / .pdf

设计
----
左：四组的 t* 分布（箱线 + 全部散点 + 基准线）。**必须画散点** ——
    箱线图会隐藏"样本只有 20 个"这件事。
右：把"数据扰动引起的散布"与"本文其他已知误差"放在同一根对数轴上比大小。
    这是本图最有用的一处：**让读者一眼看出哪一项才是主导**。

用法：py scripts/fig_robust.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOC = ROOT / "docs"
P = lambda *a: print(*a, flush=True)

NAME = {"del10": "A 随机删 10%\n(24/241 点)", "quant": "B1 量化噪声\n(附件末位精度)",
        "sensor": "B2 传感器噪声\n(假定 $\\sigma_T$=0.1℃)"}
COLOR = {"del10": "#1f77b4", "quant": "#2ca02c", "sensor": "#d62728"}


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    out = json.loads((DOC / "day3_robust_data.json").read_text(encoding="utf-8"))
    t0 = out["base"]["t_star_h"]
    runs = [r for r in out["runs"] if r.get("found")]
    groups, rescued = {}, {}
    for r in runs:
        k = r["kind"]
        if k not in NAME:
            continue
        (rescued if r.get("dt_min") not in (None, "prod") else groups
         ).setdefault(k, []).append(r["t_star_h"])
    keys = [k for k in ("del10", "quant", "sensor") if k in groups]

    from src.figures.plot_utils import save_fig, setup_style
    import matplotlib.pyplot as plt

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---------------- 左：分布 ----------------
    ax = axes[0]
    rng = np.random.default_rng(7)
    for i, k in enumerate(keys):
        v = np.array(groups[k]) * 3600.0            # 换成 s，纵轴数值更可读
        x = i + rng.uniform(-0.13, 0.13, size=v.size)
        ax.scatter(x, v, s=26, color=COLOR[k], alpha=0.75, zorder=3,
                   edgecolors="white", linewidths=0.6,
                   label="生产步长下限" if i == 0 else None)
        bp = ax.boxplot([v], positions=[i], widths=0.44, showfliers=False,
                        patch_artist=True, zorder=2,
                        medianprops=dict(color="k", lw=1.6))
        bp["boxes"][0].set(facecolor="none", edgecolor=COLOR[k], linewidth=1.6)
        # 用放宽步长下限抢救出来的样本**单独标出**：它们不是生产配置的解
        if k in rescued:
            rv = np.array(rescued[k]) * 3600.0
            xr = i + rng.uniform(-0.13, 0.13, size=rv.size)
            ax.scatter(xr, rv, s=54, facecolors="none", edgecolors=COLOR[k],
                       linewidths=1.7, zorder=4,
                       label=("放宽步长下限抢救\n（3/20，见 §2.1）"
                              if k == "del10" else None))
        ax.annotate(f"$\\sigma$={v.std(ddof=1):.1f} s\n极差 {np.ptp(v):.1f} s",
                    (i, v.max()), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8.5, color=COLOR[k])
    ax.axhline(t0 * 3600.0, color="k", ls="--", lw=1.2,
               label=f"基准 $t^*$ = {t0*3600:.1f} s")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([NAME[k] for k in keys], fontsize=9.5)
    ax.set_ylabel("$t^*$ / s", fontsize=11)
    ax.set_title("(a) 数据扰动下的 $t^*$ 分布（每点一次独立求解）", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.22)
    ax.set_ylim(min(v for k in keys for v in groups[k]) * 3600 - 20,
                max(v for k in keys for v in groups[k]) * 3600 + 30)

    # ---------------- 右：与其他误差项比大小（对数轴）----------------
    ax = axes[1]
    items = [
        ("事件定位（二分）", 5.0e-4, "#cccccc"),
        ("量化噪声（附件真实精度）", 2.0 * float(np.std(groups["quant"], ddof=1)) * 3600.0,
         "#2ca02c"),
        ("时间离散（$t^*$）", 1.86, "#999999"),
        ("空间离散（$t^*$，主导项）", 17.4, "#666666"),
    ]
    for k in keys:
        if k == "quant":
            continue
        items.append((NAME[k].replace("\n", " "),
                      2.0 * float(np.std(groups[k], ddof=1)) * 3600.0, COLOR[k]))
    items.append(("平台窗口取法（4 档极差）", 1090.0, "#8c564b"))
    labels = [a for a, _, _ in items][::-1]
    vals = [b for _, b, _ in items][::-1]
    cols = [c for _, _, c in items][::-1]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="white", linewidth=0.8)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.3g} s", (v, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 6e3)
    # 对数轴的默认刻度走 mathtext（$10^{-4}$），含 U+2212 而 SimHei 无该字形
    # → 手工给 ASCII 刻度标签（同样踩过的坑，见 06/07 号补丁的图）
    from matplotlib.ticker import FixedLocator, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]))
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1000"],
                       fontsize=9)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("量级 / s（对数轴）", fontsize=11)
    ax.set_title("(b) 各项误差/散布的量级对比", fontsize=11.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", which="both", alpha=0.25)

    fig.tight_layout()
    paths = save_fig(fig, "数据扰动稳健性", "论文第6章 问题2-3·稳健性检验")
    plt.close(fig)
    P(f"  → 已出图 {[str(x.name) for x in paths]}")
    return paths


if __name__ == "__main__":
    main()
