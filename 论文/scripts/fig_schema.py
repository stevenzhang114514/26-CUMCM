"""
示意图三张（无数据依赖）：G1 几何/边界/路线 · G1b 技术路线 · FIG-08 有限体积离散
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle, Wedge)

from palette import (C, C_CENTER, C_MOIST, C_REF, C_SURF, C_TEMP, GRAY, GRAY_D,
                     INK, ascii_num_ticks, ramp, save, setup)


# ==========================================================================
def fig_G1():
    """几何 / 边界 / 烘干路线合图。"""
    fig = plt.figure(figsize=(14, 4.8))

    # ---------------- (a) 横截面 ----------------
    ax = fig.add_subplot(1, 3, 1)
    ax.grid(False)
    ax.add_patch(Circle((0, 0), 2.0, fill=False, color=INK, lw=1.6))
    for r in (0.5, 1.0, 1.5):
        ax.add_patch(Circle((0, 0), r, fill=False, color=GRAY, lw=0.8, ls="--"))
    ax.plot(0, 0, "o", color=C_TEMP, ms=6, zorder=5)
    ax.annotate("中心 $r=0$", (0, 0), xytext=(-0.35, -0.62), fontsize=10,
                color=C_TEMP,
                arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1.0))
    ax.annotate("", xy=(2.0, 0.18), xytext=(0, 0.18),
                arrowprops=dict(arrowstyle="<->", color=C_TEMP, lw=1.3))
    ax.text(1.0, 0.30, r"$R_0=2$ cm", color=C_TEMP, fontsize=10.5, ha="center")
    # 水分只从侧面走
    for y in np.linspace(-1.7, 1.7, 7):
        ax.annotate("", xy=(2.75, y), xytext=(2.05, y),
                    arrowprops=dict(arrowstyle="->", color=C_SURF, lw=1.4))
    ax.text(2.95, 0, "侧面\n$h,\\,h_m$", color=C_SURF, fontsize=10.5,
            ha="left", va="center")
    ax.set_xlim(-2.5, 4.2)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$r$ / cm")
    ax.set_ylabel("$r$ / cm")
    ax.set_title("(a) 横截面与侧面边界", fontsize=10.5)

    # ---------------- (b) 侧视图 ----------------
    ax = fig.add_subplot(1, 3, 2)
    ax.grid(False)
    ax.add_patch(Rectangle((-2.0, -3.0), 4.0, 6.0, fill=False, color=INK, lw=1.6))
    ax.plot([0, 0], [-3.0, 3.0], ls="--", color=GRAY_D, lw=0.9)
    ax.text(0.12, 3.15, "对称轴", fontsize=9.5, color=GRAY_D)
    for y in np.linspace(-2.5, 2.5, 6):
        ax.annotate("", xy=(-2.0, y), xytext=(-3.1, y),
                    arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1.4))
    ax.text(-3.3, 0, "热风\n$T_\\infty,\\ C_\\infty$", color=C_TEMP, fontsize=10.5,
            ha="right", va="center")
    ax.annotate("", xy=(2.0, -2.75), xytext=(2.0, 2.75),
                arrowprops=dict(arrowstyle="<->", color=C_TEMP, lw=1.3))
    ax.text(2.12, 0, "$L_0=25$ cm", color=C_TEMP, fontsize=10.5, rotation=90,
            va="center")
    ax.annotate("", xy=(0, 3.6), xytext=(2.0, 3.6),
                arrowprops=dict(arrowstyle="<->", color=C_CENTER, lw=1.3))
    ax.text(1.0, 3.72, "$R_0$", color=C_CENTER, fontsize=10.5, ha="center")
    ax.set_xlim(-5.6, 3.4)
    ax.set_ylim(-3.6, 4.2)
    ax.set_aspect("equal")
    ax.set_xlabel("$r$ / cm")
    ax.set_ylabel("$z$ / cm")
    ax.set_title("(b) 侧视图：长径比 $L_0/R_0$ = 12.5", fontsize=10.5)

    # ---------------- (c) 四问递进 + 分层 ----------------
    ax = fig.add_subplot(1, 3, 3)
    ax.grid(False)
    ax.axis("off")
    boxes = [
        ("问题 1\n预热平衡 0–1800 s\n附录2 常物性\n$D(C)$ 不含 $T$", C_TEMP),
        ("问题 2\n整个烘干过程\n附录3 变物性\n$D(C,T)$ 双向耦合", C_MOIST),
        ("问题 3\n达标时间 $t_*$\n$C_{max}(t)<0.15$", C_CENTER),
        ("问题 4\n尺寸收缩 $R(t)$\n材料坐标 $\\xi=r/R$", C_SURF),
    ]
    # 🔴 四个框必须自上而下排满且不能与底部 M0/M1/M2 说明框重叠。
    #    首版把步长写成 0.245、共 4 个框（0.92→0.185），第 4 个框底边 0.09
    #    正压在说明框上，渲染出来只剩 3 个框。改为步长 0.175、说明框下移。
    y0, step, bh = 0.955, 0.175, 0.145
    for i, (txt, col) in enumerate(boxes):
        yb = y0 - i * step
        ax.add_patch(FancyBboxPatch((0.08, yb - bh), 0.84, bh,
                                    boxstyle="round,pad=0.010,rounding_size=0.03",
                                    facecolor=col, edgecolor=INK, lw=1.0,
                                    alpha=0.32, zorder=2))
        ax.text(0.5, yb - bh / 2, txt, ha="center", va="center", fontsize=8.8,
                color=INK, zorder=3, linespacing=1.35)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(0.5, yb - bh - 0.026),
                        xytext=(0.5, yb - bh - 0.002),
                        arrowprops=dict(arrowstyle="-|>", color=GRAY_D, lw=1.4))
    ax.text(0.5, 0.015,
            "M0 题设基线（四问正式答案）\n"
            "M1 可核查扩展：潜热 / 辐射 / 端面 / 参数情景\n"
            "M2 工程推广：烘房系统 / 品质 / 结构（只给路线）",
            ha="center", va="bottom", fontsize=8.6, color=GRAY_D,
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRAY, lw=0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 1.0)
    ax.set_title("(c) 四问递进与模型分层", fontsize=10.5)

    fig.suptitle("图 G1　几何、边界与求解路线", fontsize=12.5, y=1.02)
    fig.tight_layout()
    return save(fig, "G1_geometry")


# ==========================================================================
def fig_G1b():
    """技术路线流程图。"""
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    ax.grid(False)
    ax.axis("off")

    def box(x, y, w, h, txt, col, fs=9.4, alpha=0.32):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.02",
                                    facecolor=col, edgecolor=INK, lw=1.05,
                                    alpha=alpha, zorder=2))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
                color=INK, zorder=3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY_D, lw=1.4))

    box(0.5, 0.955, 0.52, 0.065, "问题重述：四问 + 附件 1/2", C_TEMP)
    arrow(0.5, 0.922, 0.5, 0.892)
    box(0.5, 0.858, 0.68, 0.070,
        "变量基准与入口检查（$C$ vs $Y$、$\\rho$ 定义、$T$ 单位、外推方式）", C_TEMP)
    arrow(0.5, 0.823, 0.5, 0.793)

    # 双列：左模型右数据
    box(0.27, 0.748, 0.42, 0.078, "① 建模\n一维轴对称热湿耦合 PDE", C_MOIST)
    box(0.75, 0.748, 0.42, 0.078, "② 数据处理\n附件1 PCHIP + 平台外推\n附件2 半径 PCHIP", C_SURF)
    arrow(0.5, 0.709, 0.5, 0.681)
    box(0.5, 0.645, 0.72, 0.068,
        "③ 数值离散：半控制体 FVM + 调和平均界面 + 渐变网格", C_MOIST)
    arrow(0.5, 0.611, 0.5, 0.583)
    box(0.5, 0.547, 0.72, 0.068,
        "④ 时间推进：Backward Euler（L-稳定）+ 块 Gauss–Seidel Picard", C_MOIST)
    arrow(0.5, 0.513, 0.5, 0.485)

    box(0.145, 0.437, 0.255, 0.078, "问题1\n单向耦合\n先水后温", C_TEMP, 9.0)
    box(0.385, 0.437, 0.255, 0.078, "问题2\n双向耦合\n联立求解", C_TEMP, 9.0)
    box(0.625, 0.437, 0.245, 0.078, "问题3\n阈值事件\n$t_*=206960$ s", C_TEMP, 9.0)
    box(0.860, 0.437, 0.245, 0.078, "问题4\n材料坐标\n$\\xi=r/R(t)$", C_TEMP, 9.0)
    for x in (0.145, 0.385, 0.625, 0.860):
        arrow(x, 0.512, x, 0.479)

    arrow(0.5, 0.398, 0.5, 0.366)
    box(0.5, 0.325, 0.80, 0.072,
        "⑤ 模型检验：解析基准 · 收敛阶 · 守恒收支 · 退化一致性 · 灵敏度/稳健性",
        C_REF, 9.0)
    arrow(0.5, 0.289, 0.5, 0.257)
    box(0.5, 0.216, 0.80, 0.072,
        "⑥ 结果交付：result1–4.xlsx + 题设表 1–6 + 全部插图", C_REF, 9.0)
    arrow(0.5, 0.180, 0.5, 0.148)
    box(0.5, 0.108, 0.80, 0.062,
        "M1 条件性扩展（潜热/端面/结壳，不改答案） · M2 推广路线", GRAY, 9.0, 0.20)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.06, 1.0)
    ax.set_title("图 G1b　技术路线", fontsize=12.5)
    fig.tight_layout()
    return save(fig, "G1b_flowchart")


# ==========================================================================
def fig_FIG08():
    """有限体积离散示意：环状控制体 + 径向剖面。"""
    N_show, R0, ds = 5, 1.0, 1.0 / 6.0
    shade = ramp(N_show, i0=5, i1=1)          # 浅青 → 蓝紫，由内向外的深浅
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    ax = axes[0]
    ax.grid(False)
    ax.add_patch(Wedge((0, 0), R0, 0, 360, width=R0 - (N_show + 0.5) * ds,
                       facecolor="#F2F2F2", edgecolor=GRAY, lw=0.8, zorder=1))
    for i in range(N_show, 0, -1):
        ax.add_patch(Wedge((0, 0), (i + 0.5) * ds, 0, 360, width=ds,
                           facecolor=shade[i - 1], edgecolor=INK, lw=1.0,
                           zorder=2))
    ax.add_patch(Circle((0, 0), 0.5 * ds, facecolor=C_TEMP, edgecolor=INK,
                        lw=1.4, zorder=3, alpha=0.85))
    ax.annotate("", xy=(R0, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color=GRAY_D))
    ax.text(R0 * 0.62, -0.035, r"$R_0=2$ cm", color=GRAY_D, fontsize=9.5,
            ha="center", va="top")
    ax.annotate("中心半控制体 $[0,\\Delta r]$\n"
                "实心圆柱 $V_0=\\pi\\Delta r^{2}$\n"
                "内侧面 $r=0$：$J\\equiv 0$",
                xy=(0.5 * ds * 0.7, 0.5 * ds * 0.7),
                xytext=(-R0 * 1.20, R0 * 1.32), fontsize=8.6, color=INK,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=GRAY_D, lw=1.1,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xlim(-R0 * 1.20, R0 * 1.20)
    ax.set_ylim(-R0 * 1.32, R0 * 1.48)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(a) 横截面：环形控制体 + 中心半控制体", fontsize=10)

    # ---------------- (b) 径向剖面 ----------------
    ax2 = axes[1]
    ax2.grid(False)
    y0, hgt = 0.0, 0.52
    for i in range(N_show):
        fc = C_TEMP if i == 0 else shade[i]
        ax2.add_patch(Rectangle((i * ds, y0), ds, hgt, facecolor=fc,
                                edgecolor=INK, lw=1.0, zorder=3,
                                alpha=0.85 if i == 0 else 0.55))
    ax2.add_patch(Rectangle((N_show * ds, y0), R0 - N_show * ds, hgt,
                            facecolor="#F2F2F2", edgecolor=GRAY, lw=0.8, zorder=2))
    ax2.text((N_show * ds + R0) / 2, y0 + hgt / 2, r"$\cdots$", ha="center",
             va="center", fontsize=15, color=GRAY_D)
    for i in range(1, 5):
        ax2.text((i - 0.5) * ds, y0 + hgt / 2, rf"$\varphi_{i}$", ha="center",
                 va="center", fontsize=11, zorder=4, color=INK)
    for j in range(1, N_show + 1):
        xf = j * ds
        ax2.plot([xf, xf], [y0 - 0.08, y0 + hgt + 0.08], color=C_MOIST, lw=1.4,
                 zorder=4)
    ax2.plot([0, 0], [y0 - 0.10, y0 + hgt + 0.10], color=INK, lw=2.6, zorder=4)
    ax2.text(0, y0 - 0.17, r"$r=0$", ha="center", va="top", fontsize=9,
             color=INK)
    ax2.plot([R0, R0], [y0 - 0.10, y0 + hgt + 0.10], color=C_CENTER, lw=2.6,
             zorder=4)
    ax2.text(R0, y0 - 0.17, r"$r=R_0$", ha="center", va="top", fontsize=9,
             color=C_CENTER)
    for j in range(1, N_show):
        xf = j * ds
        ax2.add_patch(FancyArrowPatch((xf - 0.28 * ds, y0 + hgt + 0.30),
                                      (xf + 0.28 * ds, y0 + hgt + 0.30),
                                      arrowstyle="-|>", mutation_scale=10,
                                      color=C_MOIST, lw=1.2, zorder=5))
    ax2.text(1.5 * ds, y0 + hgt + 0.44, r"$J_{i+1/2}$", ha="center", va="bottom",
             fontsize=9.5, color=C_MOIST)
    ax2.add_patch(FancyArrowPatch((R0 - 0.9 * ds, y0 + hgt + 0.30),
                                  (R0 + 0.32 * ds, y0 + hgt + 0.30),
                                  arrowstyle="-|>", mutation_scale=12,
                                  color=C_CENTER, lw=1.6, zorder=5))
    ax2.text(R0 + 0.52 * ds, y0 + hgt + 0.30,
             r"$J_s=\dfrac{h(\varphi_s-\varphi_\infty)}{1+\mathrm{Bi}_\Delta}$",
             ha="left", va="center", fontsize=9.5, color=C_CENTER)
    ax2.text(0.5 * R0, y0 - 0.46,
             r"界面扩散系数取调和平均：$\Gamma_j=2\Gamma_{j-1}\Gamma_j/"
             r"(\Gamma_{j-1}+\Gamma_j)$",
             ha="center", va="top", fontsize=8.8, color=GRAY_D)
    ax2.set_xlim(-0.10 * R0, R0 * 1.85)
    ax2.set_ylim(y0 - 0.62, y0 + hgt + 0.74)
    ax2.set_yticks([])
    ax2.set_xlabel("径向坐标 $r$ / m")
    ax2.set_title("(b) 径向离散：控制体、界面与边界条件", fontsize=10)
    for s in ("left", "right", "top"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("图 FIG-08　一维圆柱有限体积离散（中心半控制体规避 $1/r$ 奇点）",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    return save(fig, "FIG-08_fvm")


ALL = {"G1": fig_G1, "G1b": fig_G1b, "FIG08": fig_FIG08}


def main():
    setup()
    for k, f in ALL.items():
        print(f"[{k}]")
        f()


if __name__ == "__main__":
    main()
