"""
G1 几何/边界/路线合图
画药材圆柱几何、边界条件、烘干路线示意。
输出：
    figs/G1_geometry.png
    figs/G1_geometry.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def draw_cylinder(ax):
    """画圆柱截面（俯视图）。"""
    # 外圆
    circle = plt.Circle((0, 0), 2.0, fill=False, color="black", lw=1.5)
    ax.add_patch(circle)
    # 中心
    ax.plot(0, 0, "ko", ms=5)
    ax.annotate("中心", (0, 0), xytext=(0.1, 0.1),
                fontsize=10)
    # 半径标注
    ax.annotate("", xy=(2.0, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color="blue"))
    ax.text(1.0, 0.15, "R = 2 cm", color="blue", fontsize=10)
    # 距离标注
    for r in [0.5, 1.0, 1.5]:
        c = plt.Circle((0, 0), r, fill=False, color="gray",
                       lw=0.8, ls="--")
        ax.add_patch(c)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("r / cm")
    ax.set_ylabel("r / cm")
    ax.set_title("药材圆柱截面（俯视图）")


def draw_side_view(ax):
    """画圆柱侧视图与边界条件。"""
    # 圆柱
    rect = Rectangle((-2.0, -3.0), 4.0, 6.0, fill=False,
                     color="black", lw=1.5)
    ax.add_patch(rect)
    # 热风
    for y in np.linspace(-2.5, 2.5, 6):
        ax.annotate("", xy=(-2.0, y), xytext=(-3.0, y),
                    arrowprops=dict(arrowstyle="->", color="red"))
    ax.text(-3.3, 0, "热风\n$T_\\infty, C_\\infty$",
            color="red", fontsize=10, ha="right", va="center")
    # 对称轴
    ax.plot([0, 0], [-3.0, 3.0], "k--", lw=0.8)
    ax.text(0.1, 3.1, "对称轴", fontsize=9)
    # 半径
    ax.annotate("", xy=(2.0, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color="blue"))
    ax.text(1.0, 0.2, "R", color="blue")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("r / cm")
    ax.set_ylabel("z / cm")
    ax.set_title("圆柱侧视图与热风边界")


def draw_route(ax):
    """画烘干路线示意，标注 M0/M1/M2 分层。"""
    ax.plot([0, 1], [0, 0], "k-", lw=1.5)
    ax.text(0.05, 0.1, "M0：题设基线", fontsize=10)
    ax.text(0.05, -0.15, "预热平衡 0 ~ 4898 s", fontsize=9, color="gray")

    ax.annotate("", xy=(1.2, 0), xytext=(1.0, 0),
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.plot([1.2, 2.2], [0, 0], "k-", lw=1.5)
    ax.text(1.25, 0.1, "M0：恒温干燥", fontsize=10)
    ax.text(1.25, -0.15, "4898 s ~ 烘干结束", fontsize=9, color="gray")

    ax.text(0.05, 0.35, "M1：潜热 / 辐射 / 端面 / 情景（可选）",
            fontsize=9, color="blue")
    ax.text(0.05, 0.5, "M2：烘房系统 / 品质 / 结构（只给路线）",
            fontsize=9, color="green")

    ax.set_xlim(0, 2.5)
    ax.set_ylim(-0.5, 0.7)
    ax.axis("off")
    ax.set_title("烘干路线（含 M0/M1/M2 分层）")


def main():
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    draw_cylinder(ax1)

    ax2 = fig.add_subplot(1, 3, 2)
    draw_side_view(ax2)

    ax3 = fig.add_subplot(1, 3, 3)
    draw_route(ax3)

    fig.suptitle("G1 几何 / 边界 / 烘干路线合图", fontsize=14)
    fig.tight_layout()

    out_png = FIGS_DIR / "G1_geometry.png"
    out_pdf = FIGS_DIR / "G1_geometry.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("图已保存:", out_png)
    print("图已保存:", out_pdf)


if __name__ == "__main__":
    main()