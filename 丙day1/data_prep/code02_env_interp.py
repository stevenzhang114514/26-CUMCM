"""
CODE-02 环境插值
读取附件1.xlsx，生成 T_inf(t) 和 C_inf(t) 插值函数
- 支持 0~14400 s 内插值
- 14400 s 之后的外推规则留接口，等甲、乙确定
输出：
    data/env_interp.csv     插值后的密集数据（每 1 s）
    figs/G2_env.png          环境数据及插值图
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# 路径
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGS_DIR = ROOT / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_attachment1():
    """读附件1，返回时间、温度、水分浓度。"""
    path = DATA_DIR / "附件1.xlsx"
    df = pd.read_excel(path)
    t = df.iloc[:, 0].to_numpy(dtype=float)
    T = df.iloc[:, 1].to_numpy(dtype=float)
    C = df.iloc[:, 2].to_numpy(dtype=float)
    return t, T, C


def build_interp(t, T, C):
    """构建 PCHIP 插值函数。"""
    T_interp = PchipInterpolator(t, T, extrapolate=False)
    C_interp = PchipInterpolator(t, C, extrapolate=False)
    return T_interp, C_interp


def check_no_overshoot(t, T, C, T_interp, C_interp):
    """检查插值是否过冲、是否保正。"""
    t_dense = np.linspace(t[0], t[-1], 2000)
    T_dense = T_interp(t_dense)
    C_dense = C_interp(t_dense)

    T_min, T_max = T.min(), T.max()
    C_min, C_max = C.min(), C.max()

    print("温度插值范围:", T_dense.min(), "~", T_dense.max())
    print("温度原始范围:", T_min, "~", T_max)
    print("水分浓度插值范围:", C_dense.min(), "~", C_dense.max())
    print("水分浓度原始范围:", C_min, "~", C_max)

    if T_dense.min() < T_min - 1e-6 or T_dense.max() > T_max + 1e-6:
        print("[警告] 温度插值存在过冲")
    else:
        print("[OK] 温度插值无过冲")

    if C_dense.min() < C_min - 1e-9 or C_dense.max() > C_max + 1e-9:
        print("[警告] 水分浓度插值存在过冲")
    else:
        print("[OK] 水分浓度插值无过冲")

    if C_dense.min() < 0:
        print("[警告] 水分浓度插值出现负值")
    else:
        print("[OK] 水分浓度插值保正")

    return t_dense, T_dense, C_dense


def export_env_csv(t_dense, T_dense, C_dense):
    """导出插值后的密集数据到 CSV。"""
    df = pd.DataFrame({
        "time_s": t_dense,
        "T_inf_C": T_dense,
        "C_inf_kg_per_kg": C_dense,
    })
    out = DATA_DIR / "env_interp.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("插值数据已保存到:", out)
    return out


def plot_G2(t, T, C, t_dense, T_dense, C_dense):
    """画 G2：环境数据及插值。"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, T, "o", ms=3, label="附件1 原始温度")
    axes[0].plot(t_dense, T_dense, "-", lw=1.2, label="PCHIP 插值")
    axes[0].set_ylabel("温度 / ℃")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, C, "o", ms=3, label="附件1 原始水分浓度")
    axes[1].plot(t_dense, C_dense, "-", lw=1.2, label="PCHIP 插值")
    axes[1].set_xlabel("时间 / s")
    axes[1].set_ylabel("水分浓度 / (kg/kg)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("G2 烘房环境数据及插值")
    fig.tight_layout()

    out_png = FIGS_DIR / "G2_env.png"
    out_pdf = FIGS_DIR / "G2_env.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("图已保存:", out_png)
    print("图已保存:", out_pdf)


def controlled_smoothing(t, T, C):
    """受控平滑版本，仅作对照。
    依据：附件1 采样间隔 60 s；温度差分变号 116 次、
    水分浓度变号 119 次，波动遍布全程。
    采用 Savitzky-Golay 滤波，窗口 5 点（约 5 min），多项式阶 2。
    """
    window = 5
    poly = 2

    T_smooth = savgol_filter(T, window_length=window, polyorder=poly)
    C_smooth = savgol_filter(C, window_length=window, polyorder=poly)

    return T_smooth, C_smooth


def plot_FIG13(t, T, C, T_smooth, C_smooth):
    """FIG-13 环境插值 vs 受控平滑对照。"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, T, "o", ms=2, alpha=0.4, label="附件1 原始温度")
    axes[0].plot(t, T_smooth, "-", lw=1.2, label="受控平滑")
    axes[0].set_ylabel("温度 / ℃")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, C, "o", ms=2, alpha=0.4, label="附件1 原始水分浓度")
    axes[1].plot(t, C_smooth, "-", lw=1.2, label="受控平滑")
    axes[1].set_xlabel("时间 / s")
    axes[1].set_ylabel("水分浓度 / (kg/kg)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("FIG-13 环境插值 vs 受控平滑对照")
    fig.tight_layout()

    out_png = FIGS_DIR / "FIG-13_smoothing.png"
    out_pdf = FIGS_DIR / "FIG-13_smoothing.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("FIG-13 已保存:", out_png)
    print("FIG-13 已保存:", out_pdf)
def main():
    t, T, C = load_attachment1()
    print("附件1 时间范围:", t[0], "~", t[-1], "s")
    print("附件1 温度范围:", T.min(), "~", T.max(), "℃")
    print("附件1 水分浓度范围:", C.min(), "~", C.max(), "kg/kg")

    T_interp, C_interp = build_interp(t, T, C)

    t_dense, T_dense, C_dense = check_no_overshoot(
        t, T, C, T_interp, C_interp
    )

    export_env_csv(t_dense, T_dense, C_dense)
    plot_G2(t, T, C, t_dense, T_dense, C_dense)

    # 受控平滑对照
    T_smooth, C_smooth = controlled_smoothing(t, T, C)
    df_smooth = pd.DataFrame({
        "time_s": t,
        "T_inf_C": T_smooth,
        "C_inf_kg_per_kg": C_smooth,
    })
    out_smooth = DATA_DIR / "env_interp_smooth.csv"
    df_smooth.to_csv(out_smooth, index=False, encoding="utf-8-sig")
    print("受控平滑版已保存:", out_smooth)

    plot_FIG13(t, T, C, T_smooth, C_smooth)
    
    # 测试几个点
    print("\n测试插值:")
    for tt in [0, 100, 300, 600, 900, 1200, 1500, 1800, 3600, 7200, 14400]:
        if tt <= t[-1]:
            print(f"  t={tt}s  T={float(T_interp(tt)):.4f}℃  "
                  f"C={float(C_interp(tt)):.6f} kg/kg")


if __name__ == "__main__":
    main()