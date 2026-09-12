"""
数据处理章插图：G2 环境数据 · G2b 阶段识别 · FIG-13 插值 vs 平滑 · G3 半径
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import paperdata as P
from palette import (C, C_CENTER, C_MOIST, C_REF, C_SURF, C_TEMP, GRAY, GRAY_D,
                     INK, ascii_num_ticks, save, setup)

T_PRE = 14400.0        # 实测段结束时刻 s
T_PLATEAU_FROM = 9600.0
T_STAGE = 4898.0       # 预热/恒温分界（升温达最终升幅 95%）


# ==========================================================================
def fig_G2():
    """附件1 环境数据：温度与水分浓度实测序列 + 平台段。"""
    d = P.attach1()
    t = d["t"] / 3600.0
    m = d["t"] >= T_PLATEAU_FROM

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)

    for ax, y, lab, col, unit in (
            (axes[0], d["T"], "烘房温度 $T_\\infty$", C_TEMP, "℃"),
            (axes[1], d["C"], "环境水分浓度 $C_\\infty^{eq}$", C_MOIST, "kg/kg")):
        ax.plot(t, y, "-", color=col, lw=1.5, label=f"{lab}（241 点，60 s）")
        ax.plot(t[m], y[m], "o", ms=2.6, color=GRAY_D, alpha=0.55,
                label=f"平台段 $t\\geq${T_PLATEAU_FROM/3600:.1f} h（$n$={m.sum()}）")
        ax.axhline(float(y[m].mean()), color=C_CENTER, ls="--", lw=1.3,
                   label=f"平台均值 {y[m].mean():.4f} {unit}")
        ax.axvspan(T_PLATEAU_FROM / 3600, t[-1], color=C_REF, alpha=0.28,
                   zorder=0)
        ax.set_ylabel(f"{lab.split('（')[0]} / {unit}")
        ax.legend(fontsize=8.6, loc="lower right", ncol=2)
    axes[0].set_title("(a) 温度", fontsize=10.5, loc="left")
    axes[1].set_title("(b) 水分浓度", fontsize=10.5, loc="left")
    axes[1].set_xlabel("时间 / h")

    # 波动统计注记
    nT = int((np.diff(np.sign(np.diff(d["T"]))) != 0).sum())
    nC = int((np.diff(np.sign(np.diff(d["C"]))) != 0).sum())
    axes[0].text(0.015, 0.06,
                 f"温度差分变号 {nT} 次、水分 {nC} 次 —— 波动遍布全程，"
                 f"故采用忠实插值（PCHIP 过全部实测点）",
                 transform=axes[0].transAxes, fontsize=8.6, color=GRAY_D,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRAY,
                           lw=0.8, alpha=0.94))
    fig.suptitle("图 G2　附件1 环境数据及插值", fontsize=12, y=0.985)
    fig.tight_layout()
    return save(fig, "G2_env")


# ==========================================================================
def fig_G2b():
    """阶段识别：升温曲线 + 两个判据的切点。"""
    d = P.attach1()
    t = d["t"] / 3600.0
    T = d["T"]
    span = T[-1] - T[0]

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(t, T, "-", color=C_TEMP, lw=1.8, label="烘房温度实测")
    ax.axhline(T[-1], color=GRAY_D, ls=":", lw=1.0)
    ax.text(t[-1], T[-1], f"  末值 {T[-1]:.2f} ℃", fontsize=8.6, va="bottom",
            ha="right", color=GRAY_D)

    # 判据 1：达到最终升温幅度的 95%
    thr95 = T[0] + 0.95 * span
    ax.axhline(thr95, color=C_CENTER, ls="--", lw=1.2,
               label=f"判据1：升温达 95%（{thr95:.2f} ℃）→ $t$ = {T_STAGE:.0f} s")
    ax.plot([T_STAGE / 3600], [thr95], "o", color=C_CENTER, ms=7, zorder=5)
    ax.axvline(T_STAGE / 3600, color=C_CENTER, lw=0.9, ls="--", alpha=0.7)

    # 判据 2：升温速率 < 0.001 ℃/s
    dTdt = np.gradient(T, d["t"])
    k2 = int(np.argmax(dTdt < 1e-3))
    t2 = d["t"][k2] / 3600.0
    ax.plot([t2], [T[k2]], "s", color=C_MOIST, ms=6.5, zorder=5,
            label=f"判据2：$dT/dt<0.001$ ℃/s → $t$ = {d['t'][k2]:.0f} s")
    ax.axvline(t2, color=C_MOIST, lw=0.9, ls="--", alpha=0.7)

    ax.axvspan(0, T_STAGE / 3600, color=C_TEMP, alpha=0.10, zorder=0)
    ax.axvspan(T_STAGE / 3600, t[-1], color=C_SURF, alpha=0.12, zorder=0)
    ax.text(T_STAGE / 3600 / 2, T.max() * 0.995, "预热平衡段",
            ha="center", va="top", fontsize=9.5, color=C_TEMP)
    ax.text((T_STAGE / 3600 + t[-1]) / 2, T.max() * 0.995, "恒温干燥段",
            ha="center", va="top", fontsize=9.5, color=C_CENTER)

    ax.set_xlabel("时间 / h")
    ax.set_ylabel("烘房温度 / ℃")
    ax.set_title("图 G2b　预热/恒温阶段识别", fontsize=11.5)
    ax.legend(fontsize=8.8, loc="lower right")
    fig.tight_layout()
    return save(fig, "G2b_stage")


# ==========================================================================
def fig_FIG13():
    """附件1 忠实插值（PCHIP）vs 受控平滑（Savitzky–Golay）对照。"""
    d = P.attach1()
    t = d["t"] / 3600.0
    T = d["T"]

    from scipy.signal import savgol_filter
    w = 21
    Ts = savgol_filter(T, w, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    ax.plot(t, T, "o", ms=3.0, color=GRAY, alpha=0.65,
            label="附件1 实测（241 点）")
    ax.plot(t, T, "-", color=C_TEMP, lw=1.4, label="忠实插值 PCHIP（默认）")
    ax.plot(t, Ts, "-", color=C_MOIST, lw=1.8,
            label=f"受控平滑 Savitzky–Golay（窗 {w}，2 阶）")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("烘房温度 $T_\\infty$ / ℃")
    ax.set_title("(a) 两种处理方式", fontsize=10.5)
    ax.legend(fontsize=8.8, loc="lower right")

    ax = axes[1]
    ax.plot(t, T - Ts, "-", color=C_CENTER, lw=1.2)
    ax.axhline(0, color=INK, lw=0.8)
    sd = float(np.std(T - Ts))
    ax.fill_between(t, -sd, sd, color=C_CENTER, alpha=0.16,
                    label=f"$\\pm\\sigma$ = {sd:.4f} ℃")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("忠实插值 - 平滑 / ℃")
    ax.set_title("(b) 两者之差（即被平滑掉的部分）", fontsize=10.5)
    ax.legend(fontsize=9)

    fig.suptitle("图 FIG-13　环境插值 vs 受控平滑对照"
                 "（波动遍布全程，故默认不平滑）", fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "FIG-13_smoothing")


# ==========================================================================
def fig_G3():
    """附件2 半径数据、PCHIP 拟合与残差。"""
    a = P.attach2()
    t_h = a["t"] / 3600.0
    R = a["R_cm"]
    f = P.radius_interp()
    R_fit = f(a["t"]) * 100.0
    res = R - R_fit

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    ax.plot(t_h, R, "o", ms=3.2, color=GRAY, alpha=0.7,
            label=f"附件2 实测（{len(R)} 点）")
    tt = np.linspace(0, 72, 1200)
    ax.plot(tt, f(tt * 3600) * 100, "-", color=C_TEMP, lw=2.0,
            label="PCHIP 保单调拟合")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("半径 $R$ / cm")
    ax.set_title("(a) 半径随时间的收缩", fontsize=10.5)
    ax.legend(fontsize=9)
    ax.annotate(f"收缩 {(1 - R[-1] / R[0]) * 100:.1f}%\n"
                f"{R[0]:.3f} → {R[-1]:.3f} cm",
                xy=(t_h[-1], R[-1]), xytext=(-14, 22),
                textcoords="offset points", ha="right", fontsize=9,
                color=C_TEMP,
                arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1.1))

    ax = axes[1]
    ax.plot(t_h, res * 1000, "o-", ms=3.0, lw=0.9, color=C_MOIST)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("残差 $\\times10^{3}$ / cm")
    ax.set_title("(b) 拟合残差（放大 1000 倍）", fontsize=10.5)
    ax.text(0.03, 0.06,
            f"最大 $|$残差$|$ = {np.abs(res).max()*1000:.2f}$\\times10^{{-3}}$ cm\n"
            f"即拟合误差 $<10^{{-4}}$ cm 量级",
            transform=ax.transAxes, fontsize=8.8, color=GRAY_D,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRAY, lw=0.8))

    fig.suptitle("图 G3　附件2 半径数据、PCHIP 保单调拟合与残差",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "G3_radius")


ALL = {"G2": fig_G2, "G2b": fig_G2b, "FIG13": fig_FIG13, "G3": fig_G3}


def main():
    setup()
    for k, f in ALL.items():
        print(f"[{k}]", flush=True)
        f()


if __name__ == "__main__":
    main()
