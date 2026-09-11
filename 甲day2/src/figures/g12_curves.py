"""
G12  中心 / 表面 / 全域最大值三条含水率曲线      [甲 · M0 · Day2 · P0]

图名与放置位置
--------------
    figs/含水率三曲线对比图，（论文第6章 问题3·达标判据）.png/.pdf

这张图只用来说明**一件事**，但它是问题3 的判据合法性问题：
题面要求"药材**各处**的水分浓度应低于 0.15"，
所以判据必须是 **$C_{max}(t)=\\max_r C(r,t)$**，
而**不能**用平均值、也不能用表面值。

做法：把四条曲线画在同一张图上 ——
    中心 $C(0,t)$、表面 $C(R_0,t)$、**全域最大值 $C_{max}(t)$**、面积平均 $\\bar C(t)$，
并标出各自与 0.15 的交点。
若误用平均值，达标时间会被显著低估 —— 图中直接把这个差量标出来。

⚠️ 本工况 $C$ 沿 $r$ 单调递减，故 $C_{max}=C(0,t)$（中心）。
但脚本**仍然对全域取 max 并核对 argmax 位置**，
不把"最大值在中心"当作可以默认的前提（清单 §CODE-15 第 3 条）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..config import FIG_DIR, R0
from .plot_utils import save_fig, setup_style

NAME = "含水率三曲线对比"
LOC = "论文第6章 问题3·达标判据"
C_TARGET = 0.15


def area_average(C_cells, grid):
    """面积加权平均（= 截面上的水量平均）。C 定义在干基上，故权重是控制体面积。"""
    V = grid.V
    return float((np.asarray(C_cells) * V).sum() / V.sum())


def first_crossing(t, y, target=C_TARGET):
    """首次跌破 target 的时刻（线性插值，仅用于作图标注）。"""
    below = y < target
    if not np.any(below):
        return float("nan")
    k = int(np.argmax(below))
    if k == 0:
        return float(t[0])
    t0, t1 = t[k - 1], t[k]
    y0, y1 = y[k - 1], y[k]
    return float(t0 + (y0 - target) / (y0 - y1) * (t1 - t0))


def make(npz_path=None, outdir: Path | None = None, layer="M0"):
    setup_style()
    outdir = Path(outdir) if outdir else FIG_DIR
    if npz_path is None:
        npz_path = FIG_DIR.parent / "data" / "processed" / f"core_{layer}_full.npz"
    z = np.load(npz_path)

    # ⚠️ npz 里各数组长度可能差 1：单元场 (T_cells/C_cells) 是"粗扫输出 + 续算末点"，
    #    而逐时刻诊断量只到粗扫的最后一个输出时刻。作图层统一截到共同长度，
    #    并在图注里说明曲线止于 t* 前一个 60 s 网格点。
    Ccells = z["C_cells"]
    n = min(Ccells.shape[0], z["t_diag"].shape[0], z["C_max"].shape[0])
    t = z["t_diag"][:n] / 3600.0
    Cc = z["C_center"][:n]; Cs = z["C_surf"][:n]; Cm = z["C_max"][:n]
    Ccells = Ccells[:n]

    # 面积平均（用与计算网格一致的权重）
    from ..numerics.fvm_cyl import Grid
    from ..models.problem3 import c_max_of
    grid = Grid(N=Ccells.shape[1], R0=R0, grading=1.5)
    Cavg = np.array([area_average(Ccells[k], grid) for k in range(Ccells.shape[0])])

    # 全域 max 的位置核对：必须用与判据**同一个**函数 c_max_of
    # （它把 r=0 的二次外推点值与单元最大值一起比较）。
    # ⚠️ 直接看 np.argmax(C_cells, axis=1)==0 会误报：全剖面接近平坦时
    #    （如初期 C≡2.55）浮点噪声会让 argmax 落在任意单元上。
    argmaxs = np.array([c_max_of(Ccells[k], grid)[1] for k in range(Ccells.shape[0])])
    argmax_is_center = bool(np.all(argmaxs <= 1e-12))

    # **达标时刻以 day2_core.json 为唯一权威来源**，不从这里重新推：
    # 曲线只到 t* 前一个 60 s 网格点，末点 C_max 仍略高于 0.15，
    # 若在图里自行插值会得到 NaN 或偏小的值。
    t_star_auth = None
    jp = FIG_DIR.parent / "docs" / "day2_core.json"
    if jp.exists():
        import json
        _j = json.loads(jp.read_text(encoding="utf-8"))
        _k = f"tstar_{layer}"
        if _k in _j and _j[_k].get("found"):
            t_star_auth = float(_j[_k]["t_star_s"]) / 3600.0

    t_max = t_star_auth if t_star_auth is not None else first_crossing(t, Cm)
    t_avg = first_crossing(t, Cavg)
    t_srf = first_crossing(t, Cs)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ================= (a) 四条曲线 =================
    ax = axes[0]
    ax.plot(t, Cc, "-", color="#d62728", lw=2.0, label=r"中心 $C(0,t)$")
    ax.plot(t, Cm, "--", color="#111111", lw=1.6,
            label=r"全域最大值 $C_{\max}(t)$（判据用）")
    ax.plot(t, Cs, "-", color="#1f77b4", lw=1.8, label=r"表面 $C(R_0,t)$")
    ax.plot(t, Cavg, "-", color="#7f7f7f", lw=1.8, label=r"面积平均 $\bar C(t)$")

    ax.axhline(C_TARGET, color="k", lw=1.0, ls=":")
    ax.text(t[-1], C_TARGET, " 0.15 kg/kg（题面要求）", fontsize=8.5,
            va="bottom", ha="right")

    for tt, lab, col, dy in ((t_max, f"$C_{{\\max}}$ 达标 {t_max:.2f} h",
                              "#111111", 6),
                             (t_avg, f"平均达标 {t_avg:.2f} h", "#7f7f7f", -14),
                             (t_srf, f"表面 {t_srf:.2f} h", "#1f77b4", 6)):
        if np.isfinite(tt):
            ax.axvline(tt, color=col, lw=0.9, ls="--", alpha=0.7)
            ax.plot([tt], [C_TARGET], "o", color=col, ms=6, zorder=5)
            ax.annotate(lab, xy=(tt, C_TARGET),
                        xytext=(tt + (t[-1] - t[0]) * 0.03, C_TARGET + 0.35 + dy * 0.03),
                        fontsize=8.5, color=col,
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

    ax.set_xlabel("时间 / h")
    ax.set_ylabel("干基含水率 $C$ / (kg/kg)")
    ax.set_title(f"(a) 四个特征含水率的演化（{layer}）", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_ylim(0, max(2.7, float(Cc.max()) * 1.05))

    # ================= (b) 误用平均值的后果 =================
    ax = axes[1]
    if np.isfinite(t_max) and np.isfinite(t_avg):
        delta = t_max - t_avg
        rel = delta / t_max * 100.0
        bars = ax.bar(["用 $C_{\\max}$\n（正确）", "用 $\\bar C$\n（错误）"],
                      [t_max, t_avg], color=["#111111", "#c0c0c0"],
                      width=0.5, edgecolor="k", linewidth=0.8)
        for b, v in zip(bars, [t_max, t_avg]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f} h",
                    ha="center", fontsize=10, fontweight="bold")
        ax.annotate("", xy=(1, t_avg), xytext=(0, t_max),
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.6))
        ax.text(0.5, (t_max + t_avg) / 2, f"低估 {delta:.2f} h\n（{rel:.1f}%）",
                ha="center", va="center", fontsize=10, color="#d62728",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#d62728", alpha=0.95))
        ax.set_ylabel("达标时间 / h")
        ax.set_title("(b) 判据用错会低估多少", fontsize=10.5)
    else:
        ax.text(0.5, 0.5, "数据不足", ha="center", transform=ax.transAxes)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    fig.suptitle(
        "图 含水率三曲线对比：判据必须用全域最大值 $C_{\\max}$，不能用平均值或表面值",
        fontsize=10.5, y=1.02)
    fig.tight_layout()
    paths = save_fig(fig, NAME, LOC, outdir=outdir)

    stats = {"t_max_h": t_max, "t_max_source": ("day2_core.json" if t_star_auth is not None else "曲线插值"),
             "t_avg_h": t_avg, "t_surf_h": t_srf,
             "underestimate_h": (t_max - t_avg) if np.isfinite(t_max) and
             np.isfinite(t_avg) else None,
             "underestimate_pct": (t_max - t_avg) / t_max * 100.0
             if np.isfinite(t_max) and np.isfinite(t_avg) else None,
             "argmax_always_at_center": argmax_is_center,
             "note": "argmax 已逐时刻核对，未默认'最大值在中心'" if argmax_is_center
                     else "⚠️ 存在 argmax 不在中心的时刻，判据须真正取全域 max"}
    return paths, stats


if __name__ == "__main__":
    import sys, json
    p = sys.argv[1] if len(sys.argv) > 1 else None
    paths, st = make(p)
    for f in paths:
        print("  ", f)
    print(json.dumps(st, ensure_ascii=False, indent=2))
