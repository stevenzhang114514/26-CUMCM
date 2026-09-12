"""
G3 半径数据、PCHIP 拟合曲线与残差
输入：
    data/附件2.xlsx
输出：
    figs/G3_radius.png
    figs/G3_radius.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGS_DIR = ROOT / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    # 读附件2
    path = DATA_DIR / "附件2.xlsx"
    if not path.exists():
        print(f"[缺失] {path}")
        return

    df = pd.read_excel(path)
    t = df.iloc[:, 0].to_numpy(dtype=float)
    R = df.iloc[:, 1].to_numpy(dtype=float)

    print("附件2 时间范围:", t[0], "~", t[-1], "s")
    print("附件2 半径范围:", R.min(), "~", R.max(), "cm")

    # PCHIP 拟合
    pchip = PchipInterpolator(t, R)

    # 密集采样
    t_dense = np.linspace(t[0], t[-1], 2000)
    R_dense = pchip(t_dense)

    # 残差
    R_fit_at_data = pchip(t)
    residual = R_fit_at_data - R

    # 自检
    print("\n=== 自检 ===")
    for tt in [0, 7200, 86400, 259200]:
        if tt <= t[-1]:
            print(f"R({tt}) = {float(pchip(tt)):.4f} cm")

    # 单调性
    dR = np.diff(R_dense)
    if np.all(dR <= 0):
        print("单调性检查：通过（严格单调递减）")
    else:
        print("单调性检查：不通过")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：原始数据 + 拟合
    axes[0].plot(t / 3600, R, "o", ms=4, mfc="none",
                 label="附件2 原始数据点")
    axes[0].plot(t_dense / 3600, R_dense, "-", lw=1.5,
                 label="PCHIP 拟合")
    axes[0].set_xlabel("时间 / h")
    axes[0].set_ylabel("半径 / cm")
    axes[0].set_title("附件2 半径拟合")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：残差（放大 1000 倍）
    axes[1].plot(t / 3600, residual * 1000, "-", lw=1.5, color="red")
    axes[1].axhline(0, color="gray", ls="--")
    axes[1].set_xlabel("时间 / h")
    axes[1].set_ylabel("残差 (mm)")
    axes[1].set_title("拟合残差（放大 1000 倍）")
    axes[1].grid(alpha=0.3)

    fig.suptitle("G3 半径数据、PCHIP 拟合与残差")
    fig.tight_layout()

    out_png = FIGS_DIR / "G3_radius.png"
    out_pdf = FIGS_DIR / "G3_radius.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("G3 已保存:", out_png)
    print("G3 已保存:", out_pdf)


if __name__ == "__main__":
    main()