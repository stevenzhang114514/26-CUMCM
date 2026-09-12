"""
G10 关键参数与环境情景区间
数据来自甲 day2_scen.json：
    A_param：参数情景（D、h_m、h）
    B_env：环境外推情景（t_pre、T∞、C∞）
    D_d_quantile：D 一维分位区间
输出：
    figs/G10_scenario.png
    figs/G10_scenario.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 基线
BASE_H = 57.4917

# A_param：参数情景
A_LABELS = ["基线", "D×0.5", "D×2", "hm×0.5", "hm×2", "h×0.5", "h×2"]
A_VALUES = [57.4917, 109.0667, 32.5333, 64.9417, 54.6917, 57.5750, 57.4583]
A_COLORS = ["gray", "#C44E52", "#4C72B0", "#55A868", "#55A868", "#DD8452", "#DD8452"]

# B_env：环境外推情景
B_LABELS = ["基线", "tpre=9600", "T∞=49.8", "T∞=50.3", "C∞=0.0495", "C∞=0.0505"]
B_VALUES = [57.4917, 57.4917, 57.8667, 56.9500, 57.4833, 57.4989]
B_COLORS = ["gray", "#4C72B0", "#55A868", "#C44E52", "#DD8452", "#DD8452"]

# D 分位区间
D_Q = [0.05, 0.25, 0.50, 0.75, 0.95]
D_H = [77.2083, 64.7750, 57.4917, 51.1583, 43.4833]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 左图：参数情景
    x = np.arange(len(A_LABELS))
    bars = axes[0].bar(x, A_VALUES, color=A_COLORS, width=0.6)
    axes[0].axhline(BASE_H, color="black", ls="--", lw=1,
                    label=f"基线 = {BASE_H:.2f} h")
    for bar, v in zip(bars, A_VALUES):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(A_LABELS, rotation=30, ha="right")
    axes[0].set_ylabel("$t^*$ / h")
    axes[0].set_title("(a) 参数情景")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    # 中图：环境外推情景
    x = np.arange(len(B_LABELS))
    bars = axes[1].bar(x, B_VALUES, color=B_COLORS, width=0.6)
    axes[1].axhline(BASE_H, color="black", ls="--", lw=1,
                    label=f"基线 = {BASE_H:.2f} h")
    for bar, v in zip(bars, B_VALUES):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(B_LABELS, rotation=30, ha="right")
    axes[1].set_ylabel("$t^*$ / h")
    axes[1].set_title("(b) 环境外推情景")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    # 右图：D 分位区间
    axes[2].plot(D_Q, D_H, "-o", lw=1.5, ms=8, color="#4C72B0")
    axes[2].axhline(BASE_H, color="black", ls="--", lw=1,
                    label=f"基线 = {BASE_H:.2f} h")
    for q, h in zip(D_Q, D_H):
        axes[2].text(q, h + 1.5, f"{h:.1f}", ha="center", fontsize=9)
    axes[2].set_xlabel("D 倍率分位")
    axes[2].set_ylabel("$t^*$ / h")
    axes[2].set_title("(c) D 一维分位区间")
    axes[2].set_xticks(D_Q)
    axes[2].set_xticklabels(["5%", "25%", "50%", "75%", "95%"])
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("G10 关键参数与环境情景区间")
    fig.tight_layout()

    out_png = FIGS_DIR / "G10_scenario.png"
    out_pdf = FIGS_DIR / "G10_scenario.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("G10 已保存:", out_png)
    print("G10 已保存:", out_pdf)


if __name__ == "__main__":
    main()