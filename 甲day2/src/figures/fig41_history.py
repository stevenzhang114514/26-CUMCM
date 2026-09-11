"""
FIG-41  自适应时间步长与非线性迭代历史  [甲 · M0 · Day1晚间]

放置位置：论文第8章 模型检验 · 数值验证小节

内容
----
(上) 时间步长 Δt 随 t 的变化（对数纵轴），被拒绝的步用红色标出
(下) 每个接受步的 Picard 迭代次数

要点
----
* 本图直接印证"**输出间隔 ≠ 计算步长**"：输出每 1 s 落点，
  而计算步长在远离初期瞬态后会自动放大
* 迭代次数在含水率陡变段（干燥前沿形成期）出现尖峰，
  对应 D(C) 跨数量级导致的方程刚性
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import setup_style, save_fig


def make(history, outdir=None, title_extra=""):
    setup_style()
    acc = [h for h in history if h.accepted]
    rej = [h for h in history if not h.accepted]

    t_acc = np.array([h.t for h in acc])
    dt_acc = np.array([h.dt for h in acc])
    it_acc = np.array([h.iters_C for h in acc])
    t_rej = np.array([h.t for h in rej]) if rej else np.array([])
    dt_rej = np.array([h.dt for h in rej]) if rej else np.array([])

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1.0]})

    # ---------- (上) 步长 ----------
    ax = axes[0]
    ax.semilogy(t_acc, dt_acc, "-", color="#1a5490", lw=1.1, label="接受步")
    if len(t_rej):
        ax.semilogy(t_rej, dt_rej, "x", color="#c0392b", ms=4.5, mew=1.1,
                    label="拒绝步")
    ax.set_ylabel(r"时间步长 $\Delta t$ / s")
    ax.set_title("(a) 自适应时间步长历史：输出每 1 s 精确落点，"
                 "计算步长由误差控制独立决定", fontsize=9.6)
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(1.0, color="0.45", lw=1.0, ls="--")
    ax.text(0.985, 0.90, "输出间隔 1 s（计算步长上限）", transform=ax.transAxes,
            va="top", ha="right", fontsize=8.6, color="0.30")
    ax.annotate("初期瞬态：\n步长自动细化", xy=(8, 3e-3), xytext=(0.16, 0.28),
                textcoords="axes fraction", fontsize=8.4, color="#7b2d26",
                arrowprops=dict(arrowstyle="->", color="#7b2d26", lw=1.0,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_ylim(bottom=min(dt_acc.min(), dt_rej.min() if len(dt_rej) else np.inf) * 0.7)

    # ---------- (下) 迭代次数 ----------
    ax2 = axes[1]
    ax2.plot(t_acc, it_acc, "-", color="#8e44ad", lw=1.0)
    ax2.fill_between(t_acc, 0, it_acc, color="#8e44ad", alpha=0.18)
    ax2.set_xlabel("时间 t / s")
    ax2.set_ylabel("Picard 迭代次数")
    ax2.set_title("(b) 每步 Picard 迭代次数（尖峰对应含水率陡变、$D(C)$ 刚性段）",
                  fontsize=9.6)
    ax2.set_ylim(0, max(it_acc.max() * 1.25, 3))

    n_acc, n_rej = len(acc), len(rej)
    info = (f"接受步 {n_acc}，拒绝步 {n_rej}，"
            f"平均 Δt = {dt_acc.mean():.3f} s，最大 Δt = {dt_acc.max():.2f} s，"
            f"迭代次数 均值 {it_acc.mean():.2f} / 最大 {it_acc.max()}")
    fig.text(0.5, -0.022, info + title_extra, ha="center", fontsize=9, color="0.3")

    fig.tight_layout()
    paths = save_fig(fig, "自适应时间步长与非线性迭代历史",
                     "论文第8章 模型检验·数值验证", outdir=outdir)
    return paths, {"n_accepted": n_acc, "n_rejected": n_rej,
                   "dt_mean": float(dt_acc.mean()), "dt_max": float(dt_acc.max()),
                   "iters_mean": float(it_acc.mean()), "iters_max": int(it_acc.max())}


if __name__ == "__main__":
    print("需由 run_day1.py 传入真实 history")
