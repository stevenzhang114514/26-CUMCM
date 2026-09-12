"""
G9 A/B/C 顺序差分：三版本 t_dry 对比
数据来自乙的汇总：
    A：固定 2 cm + 附录3 → 57.49 h
    B：固定 2 cm + 附录4 → > 120 h
    C：收缩 + 附录4 → 73.03 h
输出：
    figs/G9_abc.png
    figs/G9_abc.pdf
"""

from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    versions = ["A\n固定 + 附录3", "B\n固定 + 附录4", "C\n收缩 + 附录4"]
    t_dry = [57.49, 120.0, 73.03]   # B 用 120 表示 "> 120"
    colors = ["#4C72B0", "#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(versions, t_dry, color=colors, width=0.5)

    # 标注柱子数值
    for bar, v in zip(bars, t_dry):
        if v < 120:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 1.5, f"{v:.2f} h",
                    ha="center", fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 1.5, "> 120 h",
                    ha="center", fontsize=10)

    ax.set_ylabel("$t_{dry}$ / h")
    ax.set_title("G9 A/B/C 顺序差分：三版本 $t_{dry}$ 对比")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()

    out_png = FIGS_DIR / "G9_abc.png"
    out_pdf = FIGS_DIR / "G9_abc.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("G9 已保存:", out_png)
    print("G9 已保存:", out_pdf)


if __name__ == "__main__":
    main()