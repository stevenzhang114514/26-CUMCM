"""
FIG-15 / FIG-40  收敛性检验图（空间、时间**分列两幅**）   [甲 · M0 · Day2晚间]

图名与放置位置
--------------
    figs/收敛性检验图，（论文第8章 模型检验·数值验证）.png/.pdf

🔴 横轴必须是**步长 h**，不是 1/h、也不是网格数 N
------------------------------------------------
E ∝ h^p 时，以 log h 为横轴的斜率是 **+p**。
若横轴取 1/h 或 N，斜率才是 −p。
本项目统一以 h 为横轴，故图中标注的是 **+p**；
这一点在 Day1 的审查里被明确纠正过（清单 §CODE-19），画错会直接误导阅卷。

产出两幅子图
------------
(a) 空间收敛：均匀网格 N ∈ {40,80,160}，参照 N=320，时间固定 dt=1 s
(b) 时间收敛：生产网格 N=200/γ=1.5，BE（θ=1）与 CN（θ=0.5）并排，
    参照为最小步长的一半；图中同时标出各自的**期望斜率**（1 与 2）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..config import FIG_DIR
from .plot_utils import save_fig, setup_style

NAME = "收敛性检验"
LOC = "论文第8章 模型检验·数值验证"

LABELS = {"T_center": "中心温度 $T(0,t)$",
          "C_center": "中心含水率 $C(0,t)$",
          "C_max": "全域最大值 $C_{\\max}(t)$"}
COLORS = {"T_center": "#1f77b4", "C_center": "#d62728", "C_max": "#2ca02c"}


def _panel(ax, rows, keyfield, title, expected, ref_note=""):
    for k in LABELS:
        hs = np.array([r[keyfield] for r in rows], dtype=float)
        es = np.array([r[f"err_{k}"] for r in rows], dtype=float)
        m = (es > 0) & np.isfinite(es)
        if m.sum() < 2:
            continue
        ax.loglog(hs[m], es[m], "o-", color=COLORS[k], lw=1.6, ms=5,
                  label=LABELS[k])
    # 期望斜率参考线（平移到图中部）
    hs = np.array([r[keyfield] for r in rows], dtype=float)
    es_all = [r[f"err_{k}"] for r in rows for k in LABELS
              if r[f"err_{k}"] > 0]
    if es_all and len(hs) >= 2:
        y0 = np.median(es_all)
        x0 = float(np.exp(np.mean(np.log(hs))))
        xx = np.array([hs.min(), hs.max()])
        yy = y0 * (xx / x0) ** expected
        ax.loglog(xx, yy, "k--", lw=1.0, alpha=0.65,
                  label=f"期望斜率 $+{expected:g}$")
    ax.set_xlabel("步长 $h$（不是 $1/h$、也不是 $N$）")
    ax.set_ylabel("与参照解的偏差")
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="best")
    if ref_note:
        ax.text(0.02, 0.03, ref_note, transform=ax.transAxes, fontsize=7.5,
                va="bottom", ha="left", color="#444444")


def make(json_path, outdir: Path | None = None):
    setup_style()
    outdir = Path(outdir) if outdir else FIG_DIR
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))

    sp = d["spatial"]
    rows = [{"h": r["h"], **{k: v for k, v in r.items() if k.startswith("err_")}}
            for r in sp["rows"]]
    _panel(axes[0], rows, "h",
           "(a) 空间收敛（均匀网格，时间固定 $\\Delta t$=1 s）",
           2.0, f"参照：N={sp['N_ref']}；t={sp['t_end']:.0f} s")

    tp = d.get("temporal_be")
    if tp:
        rows = [{"h": r["h"], **{k: v for k, v in r.items()
                                 if k.startswith("err_")}} for r in tp["rows"]]
        _panel(axes[1], rows, "h",
               f"(b) 时间收敛（生产网格，θ={tp['theta']:g}）",
               tp["expected_order"],
               f"参照：Δt={tp['dt_ref']:g} s；"
               f"BE 期望 $+1$（CN 才期望 $+2$）")

    fig.suptitle("图 空间与时间误差分开细化（横轴为步长 $h$，故斜率为 $+p$）",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    return save_fig(fig, NAME, LOC, outdir=outdir)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    for f in make(p):
        print("  ", f)
