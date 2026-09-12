"""
G1b 技术路线流程图
输出：
    figs/G1b_flowchart.png
    figs/G1b_flowchart.pdf
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, color):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02",
                       linewidth=1.2,
                       edgecolor="black",
                       facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=10)


def arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="->", mutation_scale=15,
                        color="black")
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 第 1 层：数据准备
    box(ax, 0.05, 0.93, 0.9, 0.05,
        "数据准备：附件1（环境）· 附件2（半径）· 入口检查 6 项",
        "#E8F0FE")
    arrow(ax, 0.5, 0.93, 0.5, 0.90)

    # 第 2 层：数据预处理
    box(ax, 0.05, 0.86, 0.9, 0.04,
        "环境数据插值（忠实 + 受控平滑双版本）· 半径拟合（PCHIP）",
        "#FFF3E0")
    arrow(ax, 0.5, 0.86, 0.5, 0.83)

    # 第 3 层：数值内核
    box(ax, 0.05, 0.76, 0.9, 0.07,
        "数值内核\n"
        "有限体积法（半控制体）· 物性求值（附录2/3/4）\n"
        "隐式时间推进（Backward Euler）· Picard 非线性迭代",
        "#E8F5E9")
    arrow(ax, 0.5, 0.76, 0.5, 0.73)

    # 第 4 层：四问并列
    box(ax, 0.03, 0.63, 0.45, 0.08,
        "问题1\n常物性（附录2）· 先水后温",
        "#FFEBEE")
    box(ax, 0.52, 0.63, 0.45, 0.08,
        "问题2\n变物性（附录3）· T/C 双向耦合",
        "#FFEBEE")
    box(ax, 0.03, 0.52, 0.45, 0.08,
        "问题3\n阈值通过时刻（$C_{max}<0.15$）· 二分定位",
        "#FFEBEE")
    box(ax, 0.52, 0.52, 0.45, 0.08,
        "问题4\n材料坐标 · 均匀仿射收缩（附录4）",
        "#FFEBEE")

    arrow(ax, 0.26, 0.63, 0.26, 0.60)
    arrow(ax, 0.74, 0.63, 0.74, 0.60)
    arrow(ax, 0.5, 0.76, 0.5, 0.73)
    arrow(ax, 0.5, 0.63, 0.5, 0.52)

    arrow(ax, 0.5, 0.52, 0.5, 0.47)

    # 第 5 层：模型检验
    box(ax, 0.05, 0.33, 0.9, 0.12,
        "模型检验\n"
        "Bessel 解析基准（误差 < 0.001 ℃）· 空间/时间收敛（分开细化）\n"
        "守恒与通量收支 · 五项退化与一致性检验 · 物理合理性\n"
        "几何—密度诊断（条件性）· A/B/C 顺序差分 · 半径外推敏感性",
        "#E0F7FA")
    arrow(ax, 0.5, 0.33, 0.5, 0.28)

    # 第 6 层：情景与敏感性
    box(ax, 0.05, 0.20, 0.9, 0.07,
        "参数情景与敏感性\n"
        "D 倍率 · $h_m$ · 长时环境外推 · 网格/初始半径",
        "#F3E5F5")
    arrow(ax, 0.5, 0.20, 0.5, 0.15)

    # 第 7 层：输出
    box(ax, 0.05, 0.05, 0.9, 0.09,
        "输出\n"
        "result1–4.xlsx · 题设表1–6 · 图 G1–G12 / FIG-08~43\n"
        "论文与最小复现材料",
        "#ECEFF1")

    fig.suptitle("G1b 技术路线流程图", fontsize=14)
    fig.tight_layout()

    out_png = FIGS_DIR / "G1b_flowchart.png"
    out_pdf = FIGS_DIR / "G1b_flowchart.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("G1b 已保存:", out_png)
    print("G1b 已保存:", out_pdf)


if __name__ == "__main__":
    main()