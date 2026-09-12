"""
FIG-16  无量纲数分析图                    [甲 · M0 · Day2]

图名与放置位置（按分工的文件命名约定）
--------------------------------------
    figs/无量纲数分析图，（论文第5章 问题2·模型建立与量级分析）.png/.pdf

这张图要回答**一个问题**：本题能不能用集总参数模型？
答案由两个 Biot 数共同给出，所以图必须让读者一眼看到
"内外阻力都不可忽略"以及"过程**从混合控制转向内扩散控制**"。

构成
----
(a) 本工况各无量纲数的**取值范围**（对数横轴），
    并叠加文献报道区间做对照。
    🔴 Bi_m 处**必须同时画出两段互相冲突的文献区间**
       （R3-F5：16~160；R4-F20：5.0~15.0），
       并注明本项目一律以附录3 的 D(C,T) 自算 —— 不选边、不隐去冲突。
(b) 过程演化：Bi_m(t) 与 Lu(t)。这张子图论证"单一常数 Biot 数
    不足以刻画全过程"，因此**必须用 PDE + 变物性**。

数据来源
--------
    本项目自算：results 中 npz 的 diag_* 数组（由 scripts/run_day2.py 保存）
    文献对照：  src/refs.py 中登记的 [R3-F3] [R3-F5] [R3-F2] [R4-F20] [R4-F21]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..config import C_INIT, FIG_DIR, H_COEF, HM_COEF, R0
from ..numerics.properties import D_app3, cp_app3, k_app3, rho_app3
from .plot_utils import plain_log_ticks, save_fig, setup_style

NAME = "无量纲数分析"
LOC = "论文第5章 问题2·模型建立与量级分析"

# 文献报道区间（**标注出处，不是本项目结果**）
LIT_RANGES = [
    # (标签, 下界, 上界, 引用键, 是否冲突)
    ("$Bi_m$", 5.0, 15.0, "R4-F20", True),
    ("$Bi_m$", 16.0, 160.0, "R3-F5", True),
    ("$Lu$", 1e-4, 1e-3, "R3-F3", False),
    ("$Pn$", 1e-3, 1e-2, "R3-F3", False),
    ("$Ko$", 2.5, 4.5, "R3-F3", False),
    ("$N_{rc}=h_{rad}/h$", 0.26, 0.26, "R3-F4", False),
]


def compute_own(c_range=(0.05, 0.15, 0.5, 1.0, 2.55), T_c=(28.0, 50.0)):
    """本项目自算的各无量纲数（在状态区间上取范围）。"""
    C = np.array(c_range)
    out = {}
    for key, T in (("28C", T_c[0]), ("50C", T_c[1])):
        k_ = np.atleast_1d(k_app3(C))
        cap = np.atleast_1d(rho_app3(C) * cp_app3(C))
        D_ = np.atleast_1d(D_app3(C, T))
        out[f"Bi_T_{key}"] = H_COEF * R0 / k_
        out[f"Bi_m_{key}"] = HM_COEF * R0 / D_
        out[f"Lu_{key}"] = D_ / (k_ / cap)
        out[f"D_{key}"] = D_
        out[f"alpha_{key}"] = k_ / cap
    return out


def make(npz_path=None, outdir: Path | None = None):
    setup_style()
    outdir = Path(outdir) if outdir else FIG_DIR
    own = compute_own()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ================= (a) 取值范围对照 =================
    ax = axes[0]
    rows = [
        ("$Bi_T$", own["Bi_T_50C"].min(), own["Bi_T_50C"].max(), "#1f77b4",
         "本项目自算"),
        ("$Bi_m$", own["Bi_m_28C"].min(), own["Bi_m_50C"].max(), "#d62728",
         "本项目自算（全过程）"),
        ("$Lu$", own["Lu_50C"].min(), own["Lu_50C"].max(), "#2ca02c",
         "本项目自算"),
    ]
    ylabels, ypos = [], []
    y = 0
    for name, lo, hi, col, lab in rows:
        ax.barh(y, hi - lo, left=lo, height=0.42, color=col, alpha=0.85,
                edgecolor="k", linewidth=0.6, label=lab if y == 0 else None,
                zorder=3)
        ax.plot([lo, hi], [y, y], "k|", ms=8, zorder=4)
        ylabels.append(name); ypos.append(y); y += 1.0

    # 文献区间（虚线框）
    lit_y = {}
    for name, lo, hi, key, conflict in LIT_RANGES:
        if name not in [r[0] for r in rows]:
            continue
        yy = ypos[[r[0] for r in rows].index(name)]
        c = "#ff7f0e" if conflict else "#7f7f7f"
        off = 0.30 if conflict else -0.30
        ax.barh(yy + off, hi - lo, left=lo, height=0.16, color="none",
                edgecolor=c, linewidth=1.2, linestyle="--", zorder=3)
        ax.text(hi, yy + off,
                f" {key}" + ("（与另一报道冲突）" if conflict else ""),
                va="center", ha="left", fontsize=7.5, color=c)
        lit_y.setdefault(name, []).append(key)

    # Ko / Pn 只有文献值，另起一行
    for name, lo, hi, key, conflict in LIT_RANGES:
        if name in [r[0] for r in rows]:
            continue
        ax.barh(y, hi - lo, left=lo, height=0.30, color="none",
                edgecolor="#7f7f7f", linewidth=1.2, linestyle="--", zorder=3)
        ax.text(hi * 1.15, y, f" {name}：文献 {key} 报 {lo:g}~{hi:g}"
                              f"（本项目未自算，见 §文档）",
                va="center", ha="left", fontsize=7.5, color="#7f7f7f")
        ylabels.append(name); ypos.append(y); y += 1.0

    ax.set_yticks(ypos); ax.set_yticklabels(ylabels)
    ax.set_xscale("log")
    plain_log_ticks(ax, "x")
    ax.set_xlim(1e-4, 1e4)
    ax.set_xlabel("无量纲数取值（对数轴）")
    ax.set_title("(a) 本工况取值范围 vs 文献报道区间", fontsize=10.5)
    ax.grid(True, axis="x", alpha=0.4, linestyle=":")
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#d62728", alpha=.85, label="本项目自算"),
                       Patch(facecolor="none", edgecolor="#7f7f7f",
                             linestyle="--", label="文献报道（[R3-F3] 等）"),
                       Patch(facecolor="none", edgecolor="#ff7f0e",
                             linestyle="--", label="文献间存在冲突")],
              fontsize=8, loc="lower right")

    # ================= (b) 过程演化 =================
    ax = axes[1]
    if npz_path and Path(npz_path).exists():
        z = np.load(npz_path)
        # 3 h 的 npz 里：场是 t / T / C，逐时刻诊断量以 diag_ 前缀存放
        t_h = z["t"] / 3600.0
        bim = z["diag_Bi_m_surf"]
        ax.loglog(t_h, bim, "-", color="#d62728", lw=1.8,
                  label=r"$Bi_m=h_mR_0/D_s$")
        ax.loglog(t_h, z["diag_Lu_surf"], "-", color="#2ca02c", lw=1.8,
                  label=r"$Lu=D_s/\alpha_s$")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.text(t_h[-1], 1.0, " $Bi_m=1$", fontsize=7.5, va="bottom", ha="right")
        # 标出交叉点（内/外阻力主导的转换）
        if bim.min() < 1.0 < bim.max():   # 交叉点：内/外阻力主导的转换
            k = int(np.argmax(bim >= 1.0))
            ax.axvline(t_h[k], color="#d62728", lw=0.8, ls="--", alpha=0.7)
            ax.text(t_h[k], ax.get_ylim()[1], f" $Bi_m$ 越过 1\n t≈{t_h[k]:.1f} h",
                    fontsize=7.5, color="#d62728", va="top", ha="left")
        plain_log_ticks(ax, "x")
        plain_log_ticks(ax, "y")
        ax.set_xlabel("时间 / h")
        ax.set_ylabel("无量纲数")
        ax.set_title("(b) 过程演化：控制机制发生转换", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="center left")
        ax.grid(True, which="both", alpha=0.3, linestyle=":")
    else:
        ax.text(0.5, 0.5, "未找到 npz 数据\n（先运行 run_day2.py --stage core）",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_axis_off()

    fig.suptitle("图 无量纲数分析：内外阻力均不可忽略，且控制机制随过程转换",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    return save_fig(fig, NAME, LOC, outdir=outdir)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    for f in make(p):
        print("  ", f)
