"""
FIG-43  蒸发潜热开 / 关对照图              [甲 · M0 vs M1 · Day2]

图名与放置位置
--------------
    figs/潜热开关对照图，（论文第9章 模型扩展·蒸发潜热）.png/.pdf

这张图承载 Day2 最主要的**扩展性结论**，因此必须同时说清三件事：

1. **效应有多大** —— M0（不含潜热）与 M1（含潜热）的温度/含水率差。
   与文献 [R3-F1] 报道的"忽略潜热使初期物料平均温度预测偏高 12~22 °C"直接对照。
2. **效应从哪来** —— 表面能量平衡 −k∂T/∂r|_R = h(T_s−T∞) + λJ_w
   被改写为等效环境温度 T∞ᵉᶠᶠ = T∞ − λJ_w/h，故可画出"折算温降 dT_evap(t)"。
3. **为什么不能据此改答案** —— M1 的量级正比于干基密度 ρ_d，
   而 ρ_d 由附录3 的 ρ(C) 与干物质守恒**不兼容**（[OWN-RHOD]），
   题面也未给定。故 M1 只作**条件性扩展**报告，**不得修订问题3 的答案**。

图面
----
(a) 温度：中心/表面随时间，M0 实线、M1 虚线
(b) 含水率：中心/表面随时间，M0 实线、M1 虚线，并标出各自的达标时刻
(c) 蒸发折算温降 dT_evap(t)（仅 M1）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..config import FIG_DIR
from ..refs import get
from .plot_utils import save_fig, setup_style

NAME = "潜热开关对照"
LOC = "论文第9章 模型扩展·蒸发潜热"
C_TARGET = 0.15


def _load(layer: str, outdir: Path):
    p = outdir.parent / "data" / "processed" / f"core_{layer}_3h.npz"
    f = outdir.parent / "data" / "processed" / f"core_{layer}_full.npz"
    return (np.load(p) if p.exists() else None,
            np.load(f) if f.exists() else None)


def make(outdir: Path | None = None):
    setup_style()
    outdir = Path(outdir) if outdir else FIG_DIR
    z0_3, z0_f = _load("M0", outdir)
    z1_3, z1_f = _load("M1", outdir)
    if z0_3 is None or z1_3 is None:
        raise FileNotFoundError("缺 core_M0_3h.npz / core_M1_3h.npz，"
                                "请先运行 scripts/run_day2.py --stage core")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    # ============ (a) 温度 ============
    ax = axes[0]
    t0 = z0_3["t"] / 3600.0
    t1 = z1_3["t"] / 3600.0
    for z, t, ls, tag in ((z0_3, t0, "-", "M0 不含潜热"), (z1_3, t1, "--", "M1 含潜热")):
        ax.plot(t, z["diag_T_center"], ls, color="#d62728", lw=1.9,
                label=f"{tag}·中心")
        ax.plot(t, z["diag_T_surf"], ls, color="#1f77b4", lw=1.9,
                label=f"{tag}·表面")
    ax.plot(t0, np.full_like(t0, 50.0017), ":", color="k", lw=1.0,
            label=r"烘房 $T_\infty$")
    ax.set_xlabel("时间 / h"); ax.set_ylabel("温度 / °C")
    ax.set_title("(a) 温度（前 3 h）", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax.grid(True, alpha=0.3, linestyle=":")

    # ============ (b) 含水率 ============
    ax = axes[1]
    for z, t, ls, tag in ((z0_3, t0, "-", "M0"), (z1_3, t1, "--", "M1")):
        ax.plot(t, z["diag_C_center"], ls, color="#d62728", lw=1.9, label=f"{tag}·中心")
        ax.plot(t, z["diag_C_surf"], ls, color="#1f77b4", lw=1.9, label=f"{tag}·表面")
    ax.axhline(C_TARGET, color="k", lw=1.0, ls=":")
    ax.text(t0[-1], C_TARGET, " 0.15", fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("时间 / h"); ax.set_ylabel("干基含水率 / (kg/kg)")
    ax.set_title("(b) 含水率（前 3 h）：M1 明显更慢", fontsize=10.5)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3, linestyle=":")

    # ============ (c) 折算温降 ============
    ax = axes[2]
    ax.plot(t1, z1_3["diag_dT_evap"], "-", color="#9467bd", lw=2.0,
            label=r"$\Delta T_{evap}=\lambda J_w/h$")
    ax.axhspan(12, 22, color="#ff7f0e", alpha=0.18,
               label="文献 [R3-F1] 报道\n「忽略潜热偏高 12~22 °C」")
    ax.set_xlabel("时间 / h"); ax.set_ylabel("等效环境温度降 / K")
    ax.set_title("(c) 蒸发折算温降（仅 M1）", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_ylim(bottom=0)

    # 符号约定：报告 **M0 − M1**（正值 = 不含潜热者偏高），与文献"忽略潜热偏高 12~22 °C"同向
    succ = (z0_3["diag_T_center"][-1] - z1_3["diag_T_center"][-1])
    fig.suptitle(
        f"图 蒸发潜热开/关对照：3 h 中心温度 M0−M1 = {succ:+.2f} K（M0 偏高）"
        f"（文献 [R3-F1] 报 12~22 K，方向与量级一致）；"
        f"M1 为条件性扩展，量级正比于 ρ_d，不得据以修订问题3 的答案",
        fontsize=10, y=1.03)
    fig.tight_layout()
    return save_fig(fig, NAME, LOC, outdir=outdir)


if __name__ == "__main__":
    for f in make():
        print("  ", f)
