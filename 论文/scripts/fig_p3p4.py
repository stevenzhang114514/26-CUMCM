"""
问题3 / 问题4 插图：G6 阈值与终态 · G12 四个特征含水率 ·
G7 问题4 物理坐标与移动边界 · G11 几何-密度一致性诊断
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import paperdata as P
from palette import (C, C_CENTER, C_MEAN, C_MOIST, C_REF, C_SURF, C_TEMP,
                     GRAY, GRAY_D, INK, ramp, save, setup)

T_TARGET = 0.15


def _first_crossing(t, y, target=T_TARGET):
    below = y < target
    if not np.any(below):
        return float("nan")
    k = int(np.argmax(below))
    if k == 0:
        return float(t[0])
    t0, t1, y0, y1 = t[k - 1], t[k], y[k - 1], y[k]
    return float(t0 + (y0 - target) / (y0 - y1) * (t1 - t0))


# ==========================================================================
def fig_G6():
    """问题3：C_max(t) 与阈值 + 终态径向分布。"""
    d = P.result3()
    t_h = d["t"] / 3600.0
    C = d["C"]
    Cmax = np.nanmax(C, axis=1)
    r = d["r_cm"]
    t_star = P.TSTAR_H

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    ax.plot(t_h, Cmax, "-", color=C_TEMP, lw=2.0, label="$C_{\\max}(t)$")
    ax.axhline(T_TARGET, color=C_CENTER, ls="--", lw=1.5,
               label=f"阈值 {T_TARGET} kg/kg（题面）")
    ax.axvline(t_star, color=GRAY_D, ls=":", lw=1.4)
    ax.plot([t_star], [T_TARGET], "o", color=C_CENTER, ms=9, zorder=5)
    ax.annotate(f"$t_*$ = {t_star:.2f} h\n= {t_star/24:.4f} 天",
                xy=(t_star, T_TARGET), xytext=(-24, 46),
                textcoords="offset points", ha="right", fontsize=10,
                color=C_CENTER,
                arrowprops=dict(arrowstyle="->", color=C_CENTER, lw=1.2))
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("全域最大含水率 $C_{\\max}$ / (kg/kg)")
    ax.set_title("(a) 全域最大含水率与阈值", fontsize=10.5)
    ax.legend(fontsize=9.5)
    ax.set_ylim(0, 2.8)

    ax = axes[1]
    ax.plot(r, C[-1], "-o", ms=3.4, lw=2.0, color=C_MOIST, label="终态 $t_*$")
    k60 = int(np.argmin(np.abs(d["t"] - 21600)))
    ax.plot(r, C[k60], "--", lw=1.6, color=GRAY_D, label="$t$ = 6 h")
    ax.axhline(T_TARGET, color=C_CENTER, ls="--", lw=1.3)
    ax.set_xlabel("到药材中心的距离 $r$ / cm")
    ax.set_ylabel("干基含水率 $C$ / (kg/kg)")
    ax.set_title("(b) 终态径向分布：中心是最后达标的位置", fontsize=10.5)
    ax.legend(fontsize=9.5)
    ax.annotate("中心恰好压线 0.15", xy=(r[0], C[-1, 0]), xytext=(28, 26),
                textcoords="offset points", fontsize=9, color=C_MOIST,
                arrowprops=dict(arrowstyle="->", color=C_MOIST, lw=1.1))

    fig.suptitle("图 G6　问题3：阈值通过时刻与终态含水率分布", fontsize=12, y=1.0)
    fig.tight_layout()
    return save(fig, "G6_p3_threshold")


# ==========================================================================
def fig_G12():
    """中心 / 表面 / 全域最大值 / 面积平均 四条曲线 —— 判据合法性。"""
    d = P.result3()
    t_h = d["t"] / 3600.0
    r_cm = d["r_cm"]
    C = d["C"]
    Cmax = np.nanmax(C, axis=1)
    Ccen = C[:, 0]
    Csurf = C[:, -1]

    # 🔴 面积平均必须在**计算网格**上算，不能在 0.1 cm 的输出网格上算。
    #    首版用输出网格（21 点等距）做梯形权重，得 t_avg = 35.34 h；
    #    而权威值（`甲day3/docs/g12_stats.json`，用 FVM 控制体体积加权）
    #    是 **36.04 h**，差了 0.7 h —— 因为输出网格分辨不了近表面的陡梯度，
    #    且最外一格的面积占比很大。这里改用 npz 里 FVM 细网格上的 C_cells
    #    与 grid.V 计算，与权威值一致。
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "甲day3"))
    from src.numerics.fvm_cyl import Grid as _Grid
    z = P.core_npz("M0", full=True)
    g = _Grid(N=z["C_cells"].shape[1], R0=0.02, grading=1.5)
    V = g.V
    Ccells = z["C_cells"]
    n = min(Ccells.shape[0], z["t_diag"].shape[0])
    Cavg = (Ccells[:n] * V).sum(axis=1) / V.sum()
    t_avg_h = z["t_diag"][:n] / 3600.0

    t_max = P.TSTAR_H
    t_avg = _first_crossing(t_avg_h, Cavg)
    t_srf = _first_crossing(t_h, Csurf)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    ax.plot(t_h, Ccen, "-", lw=2.0, color=C_CENTER, label="中心 $C(0,t)$")
    ax.plot(t_h, Cmax, "--", lw=1.7, color=INK,
            label="全域最大值 $C_{\\max}(t)$（判据用）")
    ax.plot(t_h, Csurf, "-", lw=1.9, color=C_SURF, label="表面 $C(R_0,t)$")
    ax.plot(t_h, Cavg, "-", lw=1.9, color=C_MEAN, label="面积平均 $\\bar C(t)$")
    ax.axhline(T_TARGET, color=GRAY_D, ls=":", lw=1.1)
    for tt, lab, col in ((t_max, f"$C_{{\\max}}$ {t_max:.2f} h", INK),
                         (t_avg, f"平均 {t_avg:.2f} h", C_MEAN),
                         (t_srf, f"表面 {t_srf:.2f} h", C_SURF)):
        if np.isfinite(tt):
            ax.plot([tt], [T_TARGET], "o", color=col, ms=7, zorder=5)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("干基含水率 $C$ / (kg/kg)")
    ax.set_title("(a) 四条特征曲线：$C_{\\max}$ 与中心几乎重合", fontsize=10.5)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 2.72)

    ax = axes[1]
    vals = [t_max, t_avg, t_srf]
    labs = ["$C_{\\max}$\n（正确）", "面积平均\n（错误）", "表面\n（错误）"]
    cols = [INK, C_MEAN, C_SURF]
    bars = ax.bar(labs, vals, color=cols, width=0.52, edgecolor=INK, lw=1.0)
    for b, v, base in zip(bars, vals, [t_max] * 3):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.2f} h",
                ha="center", fontsize=10)
        if base > v:
            ax.text(b.get_x() + b.get_width() / 2, v / 2,
                    f"低估\n{(base-v)/base*100:.1f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    ax.set_ylabel("达标时间 / h")
    ax.set_title("(b) 判据用错会低估多少", fontsize=10.5)
    ax.grid(axis="x", visible=False)

    fig.suptitle("图 G12　中心 / 表面 / 全域最大值 / 面积平均"
                 "——题面「各处」决定必须用 $C_{\\max}$", fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "G12_max_center_surface")


# ==========================================================================
def fig_G7():
    """问题4：物理坐标下的含水率演化 + 移动边界轨迹。"""
    d = P.result4()                       # 交付版 60 s × 0.1 cm
    rad = P.radius_interp()
    t_h = d["t"] / 3600.0
    r_cm = d["r_cm"]
    C = d["C"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---------- (a) 物理坐标下的含水率演化（等值线）----------
    ax = axes[0]
    Rtraj = rad(d["t"]) * 100.0                       # 各时刻半径 cm
    # 域外的点掩掉，不要把外插值画进去
    Cm = np.where(r_cm[None, :] <= Rtraj[:, None] + 1e-9, C, np.nan)
    lv = np.linspace(0.05, 2.55, 14)
    cf = ax.contourf(r_cm, t_h, Cm, levels=lv, cmap="BuPu", extend="both")
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("干基含水率 $C$ / (kg/kg)", fontsize=9)
    ax.plot(Rtraj, t_h, "-", color=INK, lw=2.4, label="药材表面 $R(t)$")
    ax.set_xlabel("到药材中心的距离 $r$ / cm")
    ax.set_ylabel("时间 / h")
    ax.set_title("(a) 物理坐标下的含水率演化", fontsize=10.5)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(0, 2.0)

    # ---------- (b) 移动边界轨迹 ----------
    ax = axes[1]
    a2 = P.attach2()
    ax.plot(a2["t"] / 3600.0, a2["R_cm"], "o", ms=3.2, color=GRAY, alpha=0.65,
            label="附件2 实测")
    tt = np.linspace(0, 72, 800)
    ax.plot(tt, rad(tt * 3600) * 100, "-", color=C_TEMP, lw=2.2,
            label="PCHIP 拟合 $R(t)$")
    ax.axvline(P.TDRY_H, color=C_CENTER, ls="--", lw=1.6)
    ax.plot([P.TDRY_H], [rad(P.TDRY_H * 3600) * 100], "o", color=C_CENTER,
            ms=9, zorder=5)
    ax.annotate(f"$t_{{dry}}$ = {P.TDRY_H:.4f} h\n"
                f"$R$ = {rad(P.TDRY_H*3600)*100:.4f} cm",
                xy=(P.TDRY_H, rad(P.TDRY_H * 3600) * 100),
                xytext=(-18, 34), textcoords="offset points", ha="right",
                fontsize=9.5, color=C_CENTER,
                arrowprops=dict(arrowstyle="->", color=C_CENTER, lw=1.2))
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("半径 $R$ / cm")
    ax.set_title("(b) 移动边界：$R$ 由 2.000 收缩到 1.198 cm", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("图 G7　问题4：材料坐标下的收缩体干燥"
                 "（$\\xi=r/R(t)$，无 $\\dot R$ 对流项）", fontsize=12, y=1.0)
    fig.tight_layout()
    return save(fig, "G7_p4_moving_boundary")


# ==========================================================================
def fig_G11():
    """几何—密度一致性诊断 e_d(t) 与条件性下界。"""
    try:
        d = P.p4_paper_table6()
    except FileNotFoundError:
        print("    [跳过] 缺 p4_paper_table6.json（先跑 甲做第四问/scripts/p4_paper.py）")
        return []
    tr = d["e_d_trace"]
    t_h = np.array(tr["t_h"], float)
    ed = np.array(tr["e_d"], float)
    Cmax = np.array(tr["C_max"], float)
    Rc = np.array(tr["R_cm"], float)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.7),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    ax.plot(t_h, ed, "-o", ms=4.5, lw=2.0, color=C_TEMP,
            label="$e_d(t)$（干物质量相对偏离）")
    ax.axhline(0.0, color=INK, lw=1.1)
    ax.axhline(d["e_d_min"], color=GRAY_D, ls=":", lw=1.0)
    ax.text(t_h[-1], d["e_d_min"], f"  最小 {d['e_d_min']:+.4f}", fontsize=8.6,
            va="bottom", ha="right", color=GRAY_D)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("$e_d(t)$")
    ax.set_title("(a) 诊断量：整体偏离 0", fontsize=10.5)
    ax.legend(fontsize=9.5)

    ax = axes[1]
    ax.plot(Rc, Cmax, "-o", ms=4.5, lw=2.0, color=C_MOIST,
            label="$C_{\\max}$ 随半径的变化")
    ax.axhline(T_TARGET, color=C_CENTER, ls="--", lw=1.4,
               label=f"阈值 {T_TARGET}")
    ax.axvline(d["R_bound_cm"], color=GRAY_D, ls=":", lw=1.5)
    ax.text(d["R_bound_cm"], 2.35,
            f" 条件性下界\n $R_f\\geq$ {d['R_bound_cm']:.4f} cm",
            fontsize=8.8, color=GRAY_D, va="top")
    ax.set_xlabel("当前半径 $R$ / cm")
    ax.set_ylabel("$C_{\\max}$ / (kg/kg)")
    ax.set_title("(b) 条件性下界 vs 实测终态半径", fontsize=10.5)
    ax.legend(fontsize=9.5)
    ax.annotate(f"实测终态 {d['R_final_cm']:.4f} cm\n"
                f"比下界小 {d['R_bound_gap_cm']:.4f} cm",
                xy=(d["R_final_cm"], Cmax[-1]), xytext=(14, 40),
                textcoords="offset points", fontsize=9, color=C_TEMP,
                arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1.1))

    fig.suptitle("图 G11　几何—密度一致性诊断（条件性，不下「数据有误」的结论）",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "G11_geo_density")


ALL = {"G6": fig_G6, "G12": fig_G12, "G7": fig_G7, "G11": fig_G11}


def main():
    setup()
    for k, f in ALL.items():
        print(f"[{k}]", flush=True)
        f()


if __name__ == "__main__":
    main()
