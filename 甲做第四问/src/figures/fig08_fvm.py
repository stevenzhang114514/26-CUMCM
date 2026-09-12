"""
FIG-08  有限体积离散示意图            [甲 · M0 · Day1下午]

放置位置：论文第4章 问题1 · 数值方法小节

内容
----
(a) 药材横截面上的同心环形控制体划分 + **中心半控制体**
    （半径 Δr 的实心圆柱，其内侧面 r=0 处通量恒为 0，故无需处理 1/r 奇点）
(b) 径向一维剖面：单元、界面、界面通量与表面 Robin 条件

⚠️ 本图为**示意**，控制体数目并非按生产网格（N=200）绘制。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch, Rectangle

from .plot_utils import setup_style, save_fig

N_SCHEM = 5          # 示意画出的控制体个数
LABEL_CELLS = 4      # 标注 φ_i 的个数


def make(N_show=N_SCHEM, R0=0.02, outdir=None):
    setup_style()
    ds = R0 / 8.0                      # 示意单元宽度
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9),
                            gridspec_kw={"width_ratios": [1.0, 1.42], "wspace": 0.30})

    # ==================================================================
    # (a) 横截面
    # ==================================================================
    ax = axes[0]
    ax.set_aspect("equal")
    ax.grid(False)

    # 未画出的外圈
    ax.add_patch(Wedge((0, 0), R0, 0, 360, width=R0 - (N_show + 0.5) * ds,
                       facecolor="0.93", edgecolor="0.6", lw=0.8, zorder=1))
    # 环形控制体
    cmap = plt.cm.Blues(np.linspace(0.20, 0.80, N_show))
    for i in range(N_show, 0, -1):
        ax.add_patch(Wedge((0, 0), (i + 0.5) * ds, 0, 360, width=ds,
                           facecolor=cmap[i - 1], edgecolor="k", lw=1.0, zorder=2))
    # 中心半控制体
    ax.add_patch(Circle((0, 0), 0.5 * ds, facecolor="#f4a582",
                        edgecolor="k", lw=1.4, zorder=3))

    # 半径标注（水平箭头 + 下方文字，避开注释框）
    ax.annotate("", xy=(R0, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color="#c0392b"))
    ax.text(R0 * 0.66, -0.0034, r"$R_0=2\,\mathrm{cm}$",
            color="#c0392b", fontsize=9.5, ha="center", va="top")

    # 中心半控制体标注（置于左上，不遮挡半径标注）
    ax.annotate(r"中心半控制体 $[0,\Delta r]$" "\n"
                r"实心圆柱 $V_0=\pi\Delta r^{2}$" "\n"
                r"内侧面 $r=0$：$J\equiv 0$",
                xy=(0.5 * ds * 0.7, 0.5 * ds * 0.7),
                xytext=(-R0 * 1.20, R0 * 1.30),
                fontsize=8.4, color="#7b2d26", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#7b2d26", lw=1.1,
                                connectionstyle="arc3,rad=-0.25"))

    ax.set_xlim(-R0 * 1.20, R0 * 1.20)
    ax.set_ylim(-R0 * 1.30, R0 * 1.46)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(a) 横截面：环形控制体 + 中心半控制体", fontsize=9.2)

    # ==================================================================
    # (b) 径向剖面
    # ==================================================================
    ax2 = axes[1]
    ax2.grid(False)

    y0, hgt = 0.0, 0.52
    # 画出的控制体
    for i in range(N_show):
        fc = "#f4a582" if i == 0 else plt.cm.Blues(0.20 + 0.60 * i / max(N_show - 1, 1))
        ax2.add_patch(Rectangle((i * ds, y0), ds, hgt,
                                facecolor=fc, edgecolor="k", lw=1.0, zorder=3))
    # 其余部分
    ax2.add_patch(Rectangle((N_show * ds, y0), R0 - N_show * ds, hgt,
                            facecolor="0.93", edgecolor="0.6", lw=0.8, zorder=2))
    ax2.text((N_show * ds + R0) / 2, y0 + hgt / 2, r"$\cdots$",
             ha="center", va="center", fontsize=15, color="0.35")

    # 单元中心：1 基编号 φ₁…φ₄
    for i in range(1, LABEL_CELLS + 1):
        ax2.text((i - 0.5) * ds, y0 + hgt / 2, rf"$\varphi_{i}$",
                 ha="center", va="center", fontsize=11, zorder=4)

    # 界面位置（单元中心格式：面在 r_{i±1/2}）
    face_lbl = {1: r"$r_{3/2}$", 2: r"$r_{5/2}$", 4: r"$r_{N-1/2}$"}
    for j in range(1, N_show + 1):
        xf = j * ds
        ax2.plot([xf, xf], [y0 - 0.08, y0 + hgt + 0.08],
                 color="#c0392b", lw=1.4, zorder=4)
        if j in face_lbl:
            ax2.text(xf, y0 + hgt + 0.14, face_lbl[j],
                     ha="center", va="bottom", fontsize=8.6, color="#c0392b")

    # 对称面
    ax2.plot([0, 0], [y0 - 0.10, y0 + hgt + 0.10], color="k", lw=2.6, zorder=4)
    ax2.text(0, y0 - 0.17, r"$r=0$", ha="center", va="top", fontsize=9, color="0.2")
    ax2.text(0, y0 - 0.30, "对称", ha="center", va="top", fontsize=8.5, color="0.2")

    # 表面 Robin
    ax2.plot([R0, R0], [y0 - 0.10, y0 + hgt + 0.10], color="#27ae60",
             lw=2.6, zorder=4)
    ax2.text(R0, y0 - 0.17, r"$r=R_0$", ha="center", va="top",
             fontsize=9, color="#1e7a45")

    # 界面通量箭头（画在单元带上方，避开 φ 标注）
    for j in range(1, N_show):
        xf = j * ds
        ax2.add_patch(FancyArrowPatch((xf - 0.30 * ds, y0 + hgt + 0.30),
                                      (xf + 0.30 * ds, y0 + hgt + 0.30),
                                      arrowstyle="-|>", mutation_scale=10,
                                      color="#1a5490", lw=1.2, zorder=5))
    ax2.text(0.5 * ds * 3.0, y0 + hgt + 0.44, r"$J_{i+1/2}$",
             ha="center", va="bottom", fontsize=9.5, color="#1a5490")

    # 表面通量
    ax2.add_patch(FancyArrowPatch((R0 - 0.9 * ds, y0 + hgt + 0.30),
                                  (R0 + 0.30 * ds, y0 + hgt + 0.30),
                                  arrowstyle="-|>", mutation_scale=12,
                                  color="#27ae60", lw=1.6, zorder=5))
    ax2.text(R0 + 0.50 * ds, y0 + hgt + 0.30,
             r"$J_s=\dfrac{h(\varphi_s-\varphi_\infty)}{1+\mathrm{Bi}_\Delta}$",
             ha="left", va="center", fontsize=9.5, color="#1e7a45")

    ax2.text(0.5 * R0, y0 - 0.46,
             r"界面扩散系数取调和平均：$\Gamma_j=2\Gamma_{j-1}\Gamma_j/(\Gamma_{j-1}+\Gamma_j)$",
             ha="center", va="top", fontsize=8.8, color="0.3")

    ax2.set_xlim(-0.10 * R0, R0 * 1.85)
    ax2.set_ylim(y0 - 0.62, y0 + hgt + 0.74)
    ax2.set_yticks([])
    ax2.set_xlabel("径向坐标 r / m")
    ax2.set_title("(b) 径向离散：控制体、界面与边界条件", fontsize=9.2)
    for s in ("left", "right", "top"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("图 FIG-08  一维圆柱有限体积离散（中心半控制体规避 $1/r$ 奇点）",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    return save_fig(fig, "有限体积离散示意",
                    "论文第4章 问题1·数值方法", outdir=outdir)


if __name__ == "__main__":
    for p in make():
        print(p)
