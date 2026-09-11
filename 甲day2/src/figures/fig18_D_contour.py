"""
FIG-18  扩散系数 D(C,T) 等值线图           [甲主 / 丙复核 · M0 · Day2]

图名与放置位置
--------------
    figs/扩散系数等值线图，（论文第5章 问题2·模型建立与量级分析）.png/.pdf

这张图要回答**两个问题**
------------------------
1. D 在状态空间里到底怎么变？—— **对数色标**是必须的，
   因为 D 跨越若干数量级，线性色标会把绝大部分区域压成一片。
2. 过程轨迹落在哪？—— 叠加模拟得到的 (C_surf, T_surf) 与 (C_center, T_center)
   轨迹，让读者看到"干燥后期材料跑进了 D 很小的角落"。

🔴 必须写明的一处**纠正**（[OWN-DRATIO]）
------------------------------------------
题面**没有**出现"265 倍"。该数字是事后计算的，且只对**附录2** 成立：
    附录2  D(2.55)/D(0.15) = e^(−0.89/2.55+0.89/0.15) = 266.2 倍
    附录3  D(2.55)/D(0.15) = e^(−0.45/2.55+0.45/0.15) =  16.8 倍   ← 问题2/3 用这个
    附录4  D(2.55)/D(0.15) = e^(−0.30/2.55+0.30/0.15) =   6.6 倍
文献研报 R3-F7 把"265 倍"写成赛题经验公式的性质，属于**对附录2 的事后计算**，
引用时必须写成"本文计算"并注明是附录2。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..config import FIG_DIR, R0
from ..numerics.properties import D_app2, D_app3, D_app4
from .plot_utils import plain_log_ticks, save_fig, setup_style

NAME = "扩散系数等值线"
LOC = "论文第5章 问题2·模型建立与量级分析"


def _ratio(C_hi, C_lo, fn):
    return float(fn(np.array([C_hi]))[0] / fn(np.array([C_lo]))[0])


def make(npz_path=None, outdir: Path | None = None):
    setup_style()
    outdir = Path(outdir) if outdir else FIG_DIR

    Cg = np.linspace(0.05, 2.60, 420)
    Tg = np.linspace(20.0, 60.0, 320)
    CC, TT = np.meshgrid(Cg, Tg)
    D = np.atleast_2d(np.asarray(D_app3(CC, TT), dtype=float))
    # D_app3 对下溢置 0；等值线用对数色标，需把 0 换成 NaN 才不污染
    Dp = np.where(D > 0, D, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3),
                             gridspec_kw={"width_ratios": [1.0, 1.12]})

    # ================= (a) 附录3 的 D(C,T) 等值线 =================
    ax = axes[0]
    im = ax.pcolormesh(CC, TT, Dp, norm=LogNorm(vmin=1e-10, vmax=2e-8),
                       cmap="viridis", shading="auto")
    cs = ax.contour(CC, TT, Dp, levels=[1e-9, 2e-9, 5e-9, 1e-8],
                    colors="w", linewidths=0.8, alpha=0.8)
    ax.clabel(cs, fmt=lambda v: f"{v:.0e}", fontsize=7.5, inline=True)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_ticks([1e-10, 1e-9, 1e-8])
    cb.set_ticklabels(["1e-10", "1e-9", "1e-8"])   # 纯文本，避免 mathtext 的 U+2212
    cb.set_label("$D$ / (m$^2$·s)（对数色标）", fontsize=9)

    # 过程轨迹（模拟结果）
    plotted = False
    if npz_path and Path(npz_path).exists():
        z = np.load(npz_path)
        Cs, Ts = z["C_surf"], z["T_surf"]
        Cc, Tc = z["C_center"], z["T_center"]
        m = (Cs >= 0.05) & (Ts >= 20.0) & (Ts <= 60.0)
        ax.plot(Cs[m], Ts[m], "-", color="#ff2d2d", lw=2.0,
                label="表面轨迹 $(C_s,T_s)$", zorder=5)
        m2 = (Cc >= 0.05) & (Tc >= 20.0) & (Tc <= 60.0)
        ax.plot(Cc[m2], Tc[m2], "-", color="#ffffff", lw=2.0, alpha=0.9,
                label="中心轨迹 $(C_c,T_c)$", zorder=5)
        ax.plot(Cs[m][0], Ts[m][0], "o", color="#ff2d2d", ms=6, zorder=6)
        plotted = True

    ax.axvline(0.15, color="w", lw=1.0, ls="--", alpha=0.9)
    ax.text(0.155, 21.0, "目标 $C=0.15$", color="w", fontsize=8, rotation=90,
            va="bottom")
    ax.set_xlabel("干基含水率 $C$ / (kg/kg)")
    ax.set_ylabel("温度 $T$ / °C")
    ax.set_title("(a) 附录3 的 $D(C,T)$（问题2/3 使用）", fontsize=10.5)
    if plotted:
        ax.legend(fontsize=8, loc="lower left", framealpha=0.85)

    # ================= (b) 三套公式的 C 方向对比 =================
    ax = axes[1]
    T_ref = 50.0
    d2 = np.atleast_1d(np.asarray(D_app2(Cg), dtype=float))
    d3 = np.atleast_1d(np.asarray(D_app3(Cg, T_ref), dtype=float))
    d4 = np.atleast_1d(np.asarray(D_app4(Cg, T_ref), dtype=float))
    for d, lab, col in ((d2, "附录2（问题1）", "#1f77b4"),
                        (d3, "附录3 @50 °C（问题2/3）", "#d62728"),
                        (d4, "附录4 @50 °C（问题4）", "#2ca02c")):
        m = d > 0
        ax.semilogy(Cg[m], d[m], "-", color=col, lw=1.9, label=lab)
    r2 = _ratio(2.55, 0.15, lambda c: D_app2(c))
    r3 = _ratio(2.55, 0.15, lambda c: np.atleast_1d(D_app3(c, T_ref)))
    r4 = _ratio(2.55, 0.15, lambda c: np.atleast_1d(D_app4(c, T_ref)))
    ax.axvspan(0.15, 2.55, color="grey", alpha=0.10)
    ax.text(1.35, 1e-13,
            f"全过程 $C$ 由 2.55 降到 0.15：\n"
            f"附录2 变化 {r2:.1f} 倍\n"
            f"附录3 变化 {r3:.1f} 倍\n"
            f"附录4 变化 {r4:.1f} 倍",
            fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))
    ax.axvline(0.15, color="k", lw=0.9, ls="--")
    ax.set_xlabel("干基含水率 $C$ / (kg/kg)")
    ax.set_ylabel("$D$ / (m$^2$·s)")
    ax.set_title("(b) 三套公式对比：$C$ 方向的变化幅度差一个量级", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    plain_log_ticks(ax, "y")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")

    fig.suptitle("图 扩散系数等值线与量级结构（题面未声明「265 倍」，见 [OWN-DRATIO]）",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    return save_fig(fig, NAME, LOC, outdir=outdir)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    for f in make(p):
        print("  ", f)
