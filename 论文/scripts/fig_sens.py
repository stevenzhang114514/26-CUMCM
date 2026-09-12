"""
灵敏度与稳健性插图：G9 A/B/C 顺序差分 · G10 参数情景与区间 ·
G13 参数灵敏度龙卷风 · G14 数据扰动稳健性
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

import paperdata as P
from palette import (C, C_CENTER, C_MEAN, C_MOIST, C_REF, C_SURF, C_TEMP,
                     GRAY, GRAY_D, INK, ascii_log_ticks, ascii_num_ticks,
                     ramp, save, setup)

T_TARGET = 0.15


# ==========================================================================
def fig_G9():
    """A/B/C 顺序差分：三版本 t_dry 对比。"""
    tA, tC = 57.4889, 51.0906
    try:
        fb = P.p4_paper_fixb()
        tB = fb["t_dry_h"] if fb.get("reached") else None
        B_note = (f"{tB:.2f} h" if tB else
                  f"$>{fb['horizon_h']:g}$ h\n（未达标）")
    except FileNotFoundError:
        tB, B_note = None, "待算"

    labs = ["A\n固定 $R_0$ + 附录3\n（= 问题3）",
            "B\n固定 $R_0$ + 附录4",
            "C\n收缩 $R(t)$ + 附录4\n（= 问题4）"]
    vals = [tA, tB if tB else 120.0, tC]
    cols = [C_TEMP, C_CENTER, C_MOIST]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    bars = ax.bar(labs, vals, color=cols, width=0.52, edgecolor=INK, lw=1.1)
    for b, v, lab in zip(bars, vals, labs):
        txt = f"{v:.2f} h" if lab.startswith(("A", "C")) else B_note
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, txt, ha="center",
                fontsize=10.5, linespacing=1.3)
    ax.set_ylabel("$t_{dry}$ / h")
    ax.set_title("图 G9　A/B/C 顺序差分：分离「物性公式」与「几何收缩」",
                 fontsize=11.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(vals) * 1.22)

    # 差分标注
    ax.annotate("", xy=(0, tA), xytext=(1, vals[1]),
                arrowprops=dict(arrowstyle="<->", color=GRAY_D, lw=1.3, ls="--"))
    ax.text(0.5, (tA + vals[1]) / 2, "物性 A→B\n（变慢）", ha="center",
            va="center", fontsize=9, color=GRAY_D,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.8))
    ax.annotate("", xy=(1, vals[1]), xytext=(2, tC),
                arrowprops=dict(arrowstyle="<->", color=GRAY_D, lw=1.3, ls="--"))
    ax.text(1.5, (vals[1] + tC) / 2, "几何 B→C\n（收缩加速）", ha="center",
            va="center", fontsize=9, color=GRAY_D,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.8))

    fig.tight_layout()
    return save(fig, "G9_abc")


# ==========================================================================
def fig_G10():
    """参数情景 + 环境外推情景 + D 一维分位区间。"""
    try:
        sc = P.day2_scen()
    except FileNotFoundError:
        print("    [跳过] 缺 day2_scen.json")
        return []
    A = sc["A_param"]["rows"]
    base = next(r["t_star_h"] for r in A if r["key"] == "base")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))

    # ---------- (a) 参数倍率情景 ----------
    ax = axes[0]
    rows = [r for r in A if r["key"] != "base" and r["reachable"]]
    keys = [r["key"] for r in rows]
    vals = [r["t_star_h"] for r in rows]
    order = np.argsort(vals)
    keys = [keys[i] for i in order]
    vals = [vals[i] for i in order]
    cols = [C_MOIST if v > base else C_TEMP for v in vals]
    ax.barh(range(len(vals)), vals, color=cols, edgecolor=INK, lw=0.9,
            height=0.62)
    ax.axvline(base, color=INK, ls="--", lw=1.4)
    ax.text(base, -0.85, f"基准 {base:.2f} h", fontsize=9, ha="center",
            color=INK)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=9)
    ax.set_xlabel("$t^*$ / h")
    ax.set_title("(a) 参数倍率情景（×0.5 / ×2）", fontsize=10.5)
    ax.grid(axis="y", visible=False)

    # ---------- (b) 环境外推情景 ----------
    ax = axes[1]
    B = sc["B_env"]["rows"]
    keys_b = [r["key"] for r in B]
    vals_b = [r["t_star_h"] for r in B]
    ax.bar(range(len(vals_b)), vals_b,
           color=[C_REF if k.endswith("9600") else C_SURF for k in keys_b],
           edgecolor=INK, lw=0.9, width=0.6)
    ax.set_xticks(range(len(keys_b)))
    ax.set_xticklabels([k.replace("_", "\n") for k in keys_b], fontsize=8.2)
    lo, hi = min(vals_b), max(vals_b)
    ax.axhspan(lo, hi, color=C_SURF, alpha=0.16, zorder=0)
    ax.set_ylim(lo - 0.12, hi + 0.12)
    ax.set_ylabel("$t^*$ / h")
    ax.set_title(f"(b) 环境外推情景：区间 [{lo:.2f}, {hi:.2f}] h", fontsize=10.5)
    ax.grid(axis="x", visible=False)

    # ---------- (c) D 一维分位区间 ----------
    ax = axes[2]
    Q = sc["D_d_quantile"]["rows"]
    qk = [r["key"] for r in Q]
    qv = [r["t_star_h"] for r in Q]
    ax.plot(range(len(qv)), qv, "o-", ms=7, lw=2.0, color=C_MOIST)
    ax.fill_between(range(len(qv)), qv, min(qv), color=C_MOIST, alpha=0.14)
    ax.set_xticks(range(len(qk)))
    ax.set_xticklabels([k.split("_")[-1] for k in qk], fontsize=9)
    ax.set_xlabel("分位点")
    ax.set_ylabel("$t^*$ / h")
    ax.set_title(f"(c) $D$ 一维分位区间（5%~95%："
                 f"[{min(qv):.2f}, {max(qv):.2f}] h）", fontsize=10.5)

    fig.suptitle("图 G10　参数情景与区间（情景 / 参数扰动分位；**不含统计置信区间**）"
                 .replace("**", ""), fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "G10_scenario")


# ==========================================================================
def fig_G13():
    """参数灵敏度：龙卷风图 + 响应曲线 + 灵敏度系数随档位。"""
    d = P.sens_param()
    t0 = d["base"]["t_star_h"]
    coeffs = {a["param"]: a for a in d["coeffs"]}
    rank20 = d["rank"][20] if 20 in d["rank"] else d["rank"]["20"]
    params = [r["param"] for r in rank20]
    SYM = {"D": "$D$", "h_m": "$h_m$", "h": "$h$"}
    COL = {"D": C_TEMP, "h_m": C_MOIST, "h": C_CENTER}
    MARK = {"D": "o", "h_m": "s", "h": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.7),
                             gridspec_kw={"width_ratios": [1.2, 1.05, 0.9]})

    # ---------- (a) 嵌套龙卷风 ----------
    ax = axes[0]
    shade = {20: C_TEMP, 10: C_MOIST, 5: C_REF}
    height = {20: 0.78, 10: 0.50, 5: 0.24}
    ypos = np.arange(len(params))[::-1]
    for dd in (20, 10, 5):
        dn = [coeffs[p][f"dt_dn_{dd}_h"] for p in params]
        up = [coeffs[p][f"dt_up_{dd}_h"] for p in params]
        ax.barh(ypos, dn, height=height[dd], color=shade[dd], edgecolor=INK,
                lw=0.6, label=f"$\\pm${dd}%", zorder=2 + (20 - dd))
        ax.barh(ypos, up, height=height[dd], color=shade[dd], edgecolor=INK,
                lw=0.6, zorder=2 + (20 - dd))
    ax.axvline(0, color=INK, lw=1.2, zorder=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SYM[p] for p in params], fontsize=13)
    ax.set_xlabel(f"$\\Delta t^*$ / h（基准 {t0:.2f} h）")
    ax.set_title("(a) 龙卷风图：$\\pm$5/10/20%", fontsize=10.5)
    ax.legend(title="扰动幅度", fontsize=8.6, title_fontsize=8.6,
              loc="lower right")
    ax.grid(axis="y", visible=False)
    for p, y in zip(params, ypos):
        ax.annotate(f"{coeffs[p]['dt_dn_20_h']:+.2f}", (coeffs[p]['dt_dn_20_h'], y),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8.4)
        ax.annotate(f"{coeffs[p]['dt_up_20_h']:+.2f}", (coeffs[p]['dt_up_20_h'], y),
                    xytext=(-4, 0), textcoords="offset points", va="center",
                    ha="right", fontsize=8.4)
    ax.margins(x=0.24)

    # ---------- (b) 响应曲线 ----------
    ax = axes[1]
    for p in params:
        rows = sorted([r for r in d["table"] if r["param"] == p],
                      key=lambda r: r["delta"])
        xs = np.array([r["delta"] for r in rows]) * 100
        ys = np.array([r["rel_Y"] for r in rows]) * 100
        ax.plot(xs, ys, MARK[p] + "-", lw=2.0, ms=6, color=COL[p], label=SYM[p])
        s5 = coeffs[p]["S_cd_5"]
        xr = np.array([-50.0, 50.0])
        ax.plot(xr, s5 * xr, ":", lw=1.3, color=COL[p], alpha=0.85)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("扰动幅度 $\\delta$ / %")
    ax.set_ylabel("相对响应 $\\Delta t^*/t^*$ / %")
    ax.set_title("(b) 响应曲线（点线 = $\\pm$5% 处线性外推）", fontsize=10.5)
    ax.legend(fontsize=10)

    # ---------- (c) S 随档位 ----------
    ax = axes[2]
    ds = [5, 10, 20]
    w = 0.26
    for k, p in enumerate(params):
        v = [coeffs[p][f"S_cd_{dd}"] for dd in ds]
        ax.bar(np.arange(3) + (k - 1) * w, v, w, color=COL[p], label=SYM[p],
               edgecolor="white", lw=0.6)
    for k, p in enumerate(params):
        r50 = next((r for r in d["table"] if r["param"] == p
                    and abs(r["delta"] + 0.5) < 1e-12), None)
        if r50:
            ax.plot([3 + (k - 1) * w], [r50["S"]], "o", ms=7, mec=COL[p],
                    mfc="white", mew=1.8)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(list(range(4)))
    ax.set_xticklabels(["5%", "10%", "20%", "50%\n(单侧,减)"], fontsize=9)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.45)
    ax.set_ylim(-3.0, 0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("$S=(\\Delta Y/Y)/(\\Delta P/P)$")
    ax.set_title("(c) $S$ 随档位（纵轴 symlog）", fontsize=10.5)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="x", visible=False)

    fig.suptitle("图 G13　参数灵敏度：$|S_D|$ 在 $\\pm$5% 处为 0.886、"
                 "在 $-$50% 处升至 1.794 —— 报「灵敏度系数」必须注明档位",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "G13_tornado")


# ==========================================================================
def fig_G14():
    """数据扰动稳健性：分布 + 误差量级对比。"""
    d = P.robust_data()
    t0 = d["base"]["t_star_h"]
    NAME = {"del10": "A 随机删 10%\n(24/241 点)",
            "quant": "B1 量化噪声\n(附件末位精度)",
            "sensor": "B2 传感器噪声\n(假定 $\\sigma_T$=0.1℃)"}
    COL = {"del10": C_TEMP, "quant": C_MOIST, "sensor": C_CENTER}

    groups, rescued = {}, {}
    for r in d["runs"]:
        if not r.get("found") or r["kind"] not in NAME:
            continue
        (rescued if r.get("dt_min") not in (None, "prod") else groups
         ).setdefault(r["kind"], []).append(r["t_star_h"])
    keys = [k for k in ("del10", "quant", "sensor") if k in groups]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    ax = axes[0]
    rng = np.random.default_rng(7)
    for i, k in enumerate(keys):
        v = np.array(groups[k]) * 3600.0
        x = i + rng.uniform(-0.13, 0.13, size=v.size)
        ax.scatter(x, v, s=26, color=COL[k], alpha=0.75, zorder=3,
                   edgecolors="white", linewidths=0.6,
                   label="生产步长下限" if i == 0 else None)
        bp = ax.boxplot([v], positions=[i], widths=0.44, showfliers=False,
                        patch_artist=True, zorder=2,
                        medianprops=dict(color=INK, lw=1.6))
        bp["boxes"][0].set(facecolor="none", edgecolor=COL[k], linewidth=1.6)
        if k in rescued:
            rv = np.array(rescued[k]) * 3600.0
            xr = i + rng.uniform(-0.13, 0.13, size=rv.size)
            ax.scatter(xr, rv, s=54, facecolors="none", edgecolors=COL[k],
                       linewidths=1.7, zorder=4,
                       label="放宽步长下限抢救\n（3/20）" if k == "del10" else None)
        ax.annotate(f"$\\sigma$={v.std(ddof=1):.1f} s\n极差 {np.ptp(v):.1f} s",
                    (i, v.max()), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8.5, color=COL[k])
    ax.axhline(t0 * 3600.0, color=INK, ls="--", lw=1.2,
               label=f"基准 {t0*3600:.1f} s")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([NAME[k] for k in keys], fontsize=9)
    ax.set_ylabel("$t^*$ / s")
    ax.set_title("(a) 数据扰动下的 $t^*$ 分布", fontsize=10.5)
    ax.legend(fontsize=8.2, loc="upper left", framealpha=0.93)
    ax.grid(axis="x", visible=False)
    yl = [v for k in keys for v in groups[k]]
    ax.set_ylim(min(yl) * 3600 - 22, max(yl) * 3600 + 34)

    ax = axes[1]
    items = [("事件定位（二分）", 5e-4, GRAY),
             ("量化噪声（2$\\sigma$）", 2 * float(np.std(groups["quant"], ddof=1)) * 3600, C_MOIST),
             ("时间离散（$t^*$）", 1.86, GRAY),
             ("空间离散（$t^*$，主导）", 17.4, GRAY_D),
             ("删 10% 数据（2$\\sigma$）", 2 * float(np.std(groups["del10"], ddof=1)) * 3600, C_TEMP),
             ("传感器噪声（2$\\sigma$）", 2 * float(np.std(groups["sensor"], ddof=1)) * 3600, C_CENTER),
             ("平台窗口取法（4 档极差）", 1090.0, C_SURF)]
    labels = [a for a, _, _ in items][::-1]
    vals = [b for _, b, _ in items][::-1]
    cols = [c for _, _, c in items][::-1]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="white", lw=0.8)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.3g} s", (v, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 6e3)
    ascii_log_ticks(ax, "x", ticks=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1000),
                    labels=["0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    ax.set_xlabel("量级 / s（对数轴）")
    ax.set_title("(b) 各项误差/散布的量级对比", fontsize=10.5)
    ax.grid(axis="y", visible=False)

    fig.suptitle("图 G14　数据扰动稳健性：影响全部经由「平台均值」单一渠道传递"
                 "（$r=-0.9997$）；**口径主导，不是数据主导**".replace("**", ""),
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    return save(fig, "G14_robust")


ALL = {"G9": fig_G9, "G10": fig_G10, "G13": fig_G13, "G14": fig_G14}


def main():
    setup()
    for k, f in ALL.items():
        print(f"[{k}]", flush=True)
        f()


if __name__ == "__main__":
    main()
