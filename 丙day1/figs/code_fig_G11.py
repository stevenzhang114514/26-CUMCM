"""
G11 几何—密度一致性诊断 e_d(t) + 条件性上界比较
数据来自乙最新版诊断（10 个点）：
    time_h: [0.02, 8.13, 16.25, 24.35, 32.47, 40.58, 48.70, 56.80, 64.92, 73.03]
    M_d:    [0.083288, 0.062919, 0.067266, 0.071501, 0.073310,
             0.074637, 0.075546, 0.076280, 0.070252, 0.070759]
    e_d:    [0.0, -0.2446, -0.1924, -0.1415, -0.1198,
             -0.1039, -0.0929, -0.0841, -0.1565, -0.1504]
条件性下界：1.2112 cm
实测终态半径：1.1980 cm
输出：
    figs/G11_geo_density.png
    figs/G11_geo_density.pdf
"""

from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = ROOT / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 乙最新版诊断数据（10 个点）
TIME_H = [0.02, 8.13, 16.25, 24.35, 32.47, 40.58, 48.70, 56.80, 64.92, 73.03]
M_D = [0.083288, 0.062919, 0.067266, 0.071501, 0.073310,
       0.074637, 0.075546, 0.076280, 0.070252, 0.070759]
E_D = [0.0, -0.2446, -0.1924, -0.1415, -0.1198,
       -0.1039, -0.0929, -0.0841, -0.1565, -0.1504]

R_UPPER_BOUND = 1.2112
R_MEASURED_FINAL = 1.1980


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：e_d(t)
    axes[0].plot(TIME_H, E_D, "-o", lw=1.5, ms=6)
    axes[0].axhline(0, color="gray", ls="--")
    axes[0].set_xlabel("时间 / h")
    axes[0].set_ylabel("$e_d(t)$")
    axes[0].set_title("(a) 几何—密度一致性诊断量")
    axes[0].grid(alpha=0.3)

    # 右图：条件性下界 vs 实测终态半径
    labels = ["条件性下界\n$R_0\\sqrt{\\rho_{d,0}/760}$",
              "实测终态半径\n$R_f$"]
    values = [R_UPPER_BOUND, R_MEASURED_FINAL]
    colors = ["red", "green"]
    bars = axes[1].bar(labels, values, color=colors, width=0.4)
    for bar, v in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.002, f"{v:.4f} cm",
                     ha="center", fontsize=10)
    axes[1].set_ylabel("半径 / cm")
    axes[1].set_title("(b) 条件性下界 vs 实测终态半径")
    axes[1].set_ylim(1.19, 1.22)
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("G11 几何—密度一致性诊断")
    fig.tight_layout()

    out_png = FIGS_DIR / "G11_geo_density.png"
    out_pdf = FIGS_DIR / "G11_geo_density.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("G11 已保存:", out_png)
    print("G11 已保存:", out_pdf)


if __name__ == "__main__":
    main()