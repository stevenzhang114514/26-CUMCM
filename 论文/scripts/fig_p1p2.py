"""
问题1 / 问题2 插图：G4 问题1 演化 · G5 问题2 前3h径向分布 ·
FIG-16 无量纲数分析 · FIG-18 扩散系数等值线
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, NullFormatter

import paperdata as P
from palette import (C, C_CENTER, C_MEAN, C_MOIST, C_REF, C_SURF, C_TEMP,
                     GRAY, GRAY_D, INK, ascii_log_ticks, ramp, save, setup)

# --------------------------------------------------------------------------
# 三套物性（与附录2/3/4 逐字一致；K = T+273.15）
# --------------------------------------------------------------------------
H_COEF, HM_COEF, R0 = 25.0, 8.0e-7, 0.02


def D_app2(C_):
    C_ = np.asarray(C_, float)
    out = np.zeros_like(C_)
    m = C_ > 0
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        out[m] = 7.0e-9 * np.exp(-0.89 / C_[m])
    return np.nan_to_num(out, nan=0.0, posinf=0.0)


def D_app3(C_, T):
    C_, T = np.asarray(C_, float), np.asarray(T, float)
    with np.errstate(over="ignore", divide="ignore"):
        return 2.4e-3 * np.exp(-0.45 / C_) * np.exp(-3850.0 / (T + 273.15))


def D_app4(C_, T):
    C_, T = np.asarray(C_, float), np.asarray(T, float)
    with np.errstate(over="ignore", divide="ignore"):
        return 4.2e-4 * np.exp(-0.30 / C_) * np.exp(-3850.0 / (T + 273.15))


def k_app3(C_):
    C_ = np.asarray(C_, float)
    return 0.36 + 0.50 * C_ / (1.0 + C_)


def rho_app3(C_):
    return 650.0 + 128.0 * np.asarray(C_, float)


def cp_app3(C_):
    C_ = np.asarray(C_, float)
    return 1200.0 + 2200.0 * C_ / (1.0 + C_)


# ==========================================================================
def fig_G4():
    """问题1：温度与含水率的径向演化（分子图，禁双纵轴）。"""
    d = P.result1()
    t, r = d["t"], d["r_cm"]
    T, Cc = d["T"], d["C"]
    hours = [100, 300, 600, 900, 1200, 1500, 1800]
    cols = ramp(len(hours), i0=5, i1=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax = axes[0]
    for h, col in zip(hours, cols):
        k = int(np.argmin(np.abs(t - h)))
        ax.plot(r, T[k], "-o", ms=3.0, lw=1.6, color=col,
                label=f"$t$ = {h} s")
    ax.set_xlabel("到药材中心的距离 $r$ / cm")
    ax.set_ylabel("温度 $T$ / ℃")
    ax.set_title("(a) 温度场：表面升温快、中心滞后", fontsize=10.5)
    ax.legend(fontsize=8.4, ncol=2, loc="lower right")
    ax.annotate(f"1800 s：中心 {T[-1,0]:.2f} ℃ / 表面 {T[-1,-1]:.2f} ℃\n"
                f"径向温差 {T[-1,-1]-T[-1,0]:.2f} ℃",
                xy=(r[-1], T[-1, -1]), xytext=(-10, -46),
                textcoords="offset points", ha="right", fontsize=9,
                color=C_TEMP,
                arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1.1))

    ax = axes[1]
    for h, col in zip(hours, cols):
        k = int(np.argmin(np.abs(t - h)))
        ax.plot(r, Cc[k], "-o", ms=3.0, lw=1.6, color=col, label=f"$t$ = {h} s")
    ax.axvline(1.3, color=GRAY_D, ls=":", lw=1.2)
    ax.text(1.3, 2.62, " 干燥前沿 $C=2.5$ 等值线（$r\\approx1.3$ cm）",
            fontsize=8.6, color=GRAY_D, va="top")
    ax.set_xlabel("到药材中心的距离 $r$ / cm")
    ax.set_ylabel("干基含水率 $C$ / (kg/kg)")
    ax.set_title("(b) 含水率场：干燥完全限制在表层", fontsize=10.5)
    ax.legend(fontsize=8.4, ncol=2, loc="lower left")
    ax.set_ylim(1.4, 2.72)

    fig.suptitle("图 G4　问题1：预热平衡阶段（0–1800 s）温度与含水率演化",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    return save(fig, "G4_p1_evolution")


# ==========================================================================
def fig_G5():
    """问题2：前 3 h 的径向分布。"""
    d = P.result2()
    t_s, r = d["t"], d["r_cm"]
    T, Cc = d["T"], d["C"]
    hours = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    cols = ramp(len(hours), i0=5, i1=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for ax, F, lab, unit, ttl in (
            (axes[0], T, "温度 $T$", "℃", "(a) 温度：3 h 内已接近烘房温度"),
            (axes[1], Cc, "干基含水率 $C$", "kg/kg",
             "(b) 含水率：显著下降但远未达终点")):
        for h, col in zip(hours, cols):
            k = int(np.argmin(np.abs(t_s - h * 3600)))
            ax.plot(r, F[k], "-o", ms=3.0, lw=1.6, color=col,
                    label=f"$t$ = {h:g} h")
        ax.set_xlabel("到药材中心的距离 $r$ / cm")
        ax.set_ylabel(f"{lab} / {unit}")
        ax.set_title(ttl, fontsize=10.5)
        ax.legend(fontsize=8.6, ncol=2)
    axes[0].annotate(f"3 h：中心 {T[-1,0]:.4f} ℃\n表面 {T[-1,-1]:.4f} ℃",
                     xy=(r[0], T[-1, 0]), xytext=(12, -30),
                     textcoords="offset points", fontsize=9, color=C_TEMP,
                     arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1.1))
    axes[1].annotate(f"3 h：中心 {Cc[-1,0]:.4f}\n表面 {Cc[-1,-1]:.4f}",
                     xy=(r[0], Cc[-1, 0]), xytext=(12, -34),
                     textcoords="offset points", fontsize=9, color=C_MOIST,
                     arrowprops=dict(arrowstyle="->", color=C_MOIST, lw=1.1))

    fig.suptitle("图 G5　问题2：整个烘干过程前 3 h 的径向分布"
                 "（附录3 变物性 + 双向耦合）", fontsize=12, y=1.0)
    fig.tight_layout()
    return save(fig, "G5_p2_radial")


# ==========================================================================
def fig_FIG16():
    """无量纲数：Bi_m 在过程中跨越 1（本项目的关键机理）。"""
    Cg = np.logspace(np.log10(0.15), np.log10(2.55), 300)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))

    # ---------------- (a) Bi 随含水率 ----------------
    ax = axes[0]
    for T_, lab, col in ((50.0, "$T$ = 50 ℃（恒温段）", C_TEMP),
                         (28.0, "$T$ = 28 ℃（初期）", C_MOIST)):
        Bi_m = HM_COEF * R0 / np.asarray(D_app3(Cg, T_))
        ax.plot(Cg, Bi_m, "-", lw=2.2, color=col, label=f"$Bi_m$，{lab}")
    Bi_T = H_COEF * R0 / k_app3(Cg)
    ax.plot(Cg, Bi_T, "--", lw=2.0, color=C_CENTER, label="$Bi_T$（附录3 的 $k(C)$）")
    ax.axhline(1.0, color=INK, lw=1.1, ls=":")
    ax.text(2.45, 1.06, "$Bi=1$：内外阻力相当", fontsize=8.8, ha="right",
            color=INK)
    ax.fill_between(Cg, 0.1, 1.0, color=C_REF, alpha=0.35, zorder=0)
    ax.text(1.4, 0.30, "表面阻力控制\n($Bi<1$)", fontsize=8.6, color=GRAY_D,
            ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("干基含水率 $C$ / (kg/kg)")
    ax.set_ylabel("毕渥数")
    ax.set_title("(a) $Bi_m$ 由 1.19 升到 47.7，过程中跨越 1", fontsize=10.5)
    ascii_log_ticks(ax, "x", ticks=(0.15, 0.3, 0.6, 1.2, 2.55),
                    labels=["0.15", "0.3", "0.6", "1.2", "2.55"])
    ascii_log_ticks(ax, "y", ticks=(0.1, 0.3, 1, 3, 10, 30, 100),
                    labels=["0.1", "0.3", "1", "3", "10", "30", "100"])
    ax.legend(fontsize=8.6, loc="upper left")

    # ---------------- (b) 特征时间 ----------------
    ax = axes[1]
    alpha = k_app3(Cg) / (rho_app3(Cg) * cp_app3(Cg))
    tau_T = R0 ** 2 / alpha / 3600.0
    for T_, lab, col in ((50.0, "$T$ = 50 ℃", C_TEMP), (28.0, "$T$ = 28 ℃", C_MOIST)):
        tau_m = R0 ** 2 / np.asarray(D_app3(Cg, T_)) / 3600.0
        ax.plot(Cg, tau_m, "-", lw=2.2, color=col, label=f"传质 $\\tau_m$，{lab}")
    ax.plot(Cg, tau_T, "--", lw=2.0, color=C_CENTER, label="传热 $\\tau_T$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("干基含水率 $C$ / (kg/kg)")
    ax.set_ylabel("特征时间 $R_0^2/\\alpha$ 或 $R_0^2/D$ / h")
    ax.set_title("(b) 传质特征时间比传热大 1~2 个数量级", fontsize=10.5)
    ascii_log_ticks(ax, "x", ticks=(0.15, 0.3, 0.6, 1.2, 2.55),
                    labels=["0.15", "0.3", "0.6", "1.2", "2.55"])
    ascii_log_ticks(ax, "y", ticks=(0.1, 1, 10, 100),
                    labels=["0.1", "1", "10", "100"])
    ax.legend(fontsize=8.6, loc="upper right")

    fig.suptitle("图 FIG-16　无量纲数分析：为什么「烘干 2~3 天」而「预热只要 40 分钟」",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "FIG-16_dimensionless")


# ==========================================================================
def fig_FIG18():
    """三套附录的扩散系数 D(C,T) 等值线对照。"""
    Cg = np.logspace(np.log10(0.05), np.log10(2.55), 400)
    Tg = np.linspace(28.0, 50.0, 300)
    CC, TT = np.meshgrid(Cg, Tg)

    sets = [("附录2（问题1）", D_app2(CC) * np.ones_like(TT), C_TEMP),
            ("附录3（问题2/3）", D_app3(CC, TT), C_MOIST),
            ("附录4（问题4）", D_app4(CC, TT), C_CENTER)]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4))
    for ax, (name, D, col) in zip(axes, sets):
        m = np.isfinite(D) & (D > 0)
        pc = ax.pcolormesh(CC, TT, np.where(m, D, np.nan),
                           norm=LogNorm(vmin=1e-16, vmax=1e-3),
                           cmap="BuPu", shading="auto")
        cb = fig.colorbar(pc, ax=ax, pad=0.02)
        cb.set_label("$D$ / (m$^2$/s)", fontsize=9)
        cb.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _p: f"{v:g}"))
        cb.ax.yaxis.set_minor_formatter(NullFormatter())
        # 本文工况轨迹（C 由 2.55 降到 0.15，T 由 28 升到 50）
        Ctr = np.linspace(2.55, 0.15, 200)
        Ttr = np.linspace(28.0, 50.0, 200)
        ax.plot(Ctr, Ttr, "-", color=INK, lw=1.8)
        ax.plot([2.55], [28.0], "o", color=INK, ms=6)
        ax.plot([0.15], [50.0], "s", color=INK, ms=6)
        ax.text(1.35, 34.5, "本文工况轨迹", fontsize=8.8, color=INK, rotation=-30)
        ax.set_xscale("log")
        ax.set_xlabel("$C$ / (kg/kg)")
        ax.set_title(name, fontsize=10.5)
        ascii_log_ticks(ax, "x", ticks=(0.05, 0.1, 0.3, 0.6, 1.2, 2.55),
                        labels=["0.05", "0.1", "0.3", "0.6", "1.2", "2.55"])
    axes[0].set_ylabel("温度 $T$ / ℃")

    fig.suptitle("图 FIG-18　附录2/3/4 的扩散系数 $D(C,T)$ 等值线与本文工况轨迹"
                 "（注意三者的色标范围相同）", fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "FIG-18_D_contour")


ALL = {"G4": fig_G4, "G5": fig_G5, "FIG16": fig_FIG16, "FIG18": fig_FIG18}


def main():
    setup()
    for k, f in ALL.items():
        print(f"[{k}]", flush=True)
        f()


if __name__ == "__main__":
    main()
