"""
审稿意见补做 ① — 参数灵敏度（±5% / ±10% / ±20%）+ 灵敏度系数 + 龙卷风图
                                                          [甲 · 验证脚本]

审稿人的原话要求
----------------
    "关键参数 ±5%/±10%/±20%，计算灵敏度系数，龙卷风图或折线图可视化"

本脚本补的正是这三件：
    ① 三档扰动（±5%、±10%、±20%）—— Day2 只做了 ×0.5 / ×2（即 ±50% / ±100%）
    ② 灵敏度系数  S = (ΔY/Y)/(ΔP/P)，  Y = t*（问题3 烘干时间）
    ③ 龙卷风图（tornado），并附 S(δ) 曲线以暴露**非线性**

为什么必须补三档小扰动（而不是拿 ×0.5/×2 充数）
-----------------------------------------------
S 的定义是**局部**（对数）导数，只有当响应在扰动区间内近似幂律时才与扰动幅度无关。
本文的 D(C,T) = 2.4e-3·exp(-0.45/C)·exp(-3850/T_K) 含 exp(-0.45/C)：
    D 变小 → 干燥变慢 → C 在更长时间里维持在高位 → exp(-0.45/C) 反而**更大**
    ⇒ **自限效应**，响应不是幂律，S 随 δ 变化。
实测（见 docs/day3_sens_param.json）证实了这一点：|S_D| 从 δ=5% 的 ~1.1
一路涨到 δ=100% 的 1.79。**用 ±100% 的 S 代表"灵敏度系数"会系统性高估。**

方法（与 Day3 冻结口径一致）
----------------------------
    网格  N=200, γ=1.5（生产网格）
    容差  NUMERICS 生产容差（**不放松**：见下）
    求根  problem3.locate_threshold（事件早停 + 固定锚点二分），与交付值同源
    并行  ProcessPoolExecutor（纯 CPU、单线程 numpy，16 核可跑 8 路）

🔴 **不放松容差**的理由：Day3 实测 ×1→×100 容差会让 t* 移动 16.7 s，
   而 ±5% 扰动对 t* 的移动约 1×10⁴ s（大 3 个量级），看似无妨，
   但本脚本要报的**是同一批数之间的差**，容差若各档不同就成了不可归因的混杂。
   故全部 19 次运行（基准 1 + 扰动 18）用**完全同一套**数值配置。

用法
----
    py scripts/run_sens_param.py --stage param     # 跑 19 次（约 10–25 min，8 路并行）
    py scripts/run_sens_param.py --stage fig       # 只出图（读 json）
    py scripts/run_sens_param.py --stage all
    py scripts/run_sens_param.py --stage param --workers 6
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------
VERSION = "A-2026-M0-r4"
T_PRE = 14400.0
N_GRID, GRADING = 200, 1.5
MAX_HORIZON = 6.0 * 86400.0
DT_OUT = 60.0

# 三档扰动幅度（审稿人指定）
LEVELS = (0.05, 0.10, 0.20)
# ★ 额外补跑的大扰动档（−50% / +50% / +100%）
#   为什么补：审稿人给的 ±5/10/20% 是**局部**档；Day2 已有的 ×0.5 / ×2 是**大**扰动档，
#   但那一批用的是 dt_out=1800 的另一套配置。把 −50%/+50%/+100% 放到**同一批**里跑，
#   响应曲线 S(δ) 才有一条自洽的基准线，也才能把 Day2 的值当回归来核对。
EXTRA_LEVELS = (0.50, 1.00)
# 参数 → Problem2Setup 的倍率字段名
PARAMS = {"D": "x_D", "h_m": "x_hm", "h": "x_h"}
# 论文正文里用的符号
SYMBOL = {"D": r"$D$", "h_m": r"$h_m$", "h": r"$h$"}

DOC = ROOT / "docs"
FIGS = ROOT / "figs"
JSON_NAME = "day3_sens_param.json"

P = lambda *a: print(*a, flush=True)


# ==========================================================================
# 单个配置运行（**必须模块级**：Windows 下 spawn 要能 pickle）
# ==========================================================================
def _run_one(job: dict) -> dict:
    """job = {key, param, delta, mult, x_D, x_hm, x_h}"""
    # 子进程是新解释器，不继承父进程 stdout.reconfigure → 每个子进程再设一次
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    from src.config import NUMERICS
    from src.data_prep.env_interp import load_env
    from src.data_prep.env_extrap import build_env
    from src.models.boundary import BoundaryConfig
    from src.models.problem2 import Problem2Setup
    from src.models.problem3 import locate_threshold
    from src.numerics.fvm_cyl import Grid

    env = build_env(load_env(), mode="faithful", t_pre=T_PRE)
    grid = Grid(N=N_GRID, R0=0.02, grading=GRADING)
    bc = BoundaryConfig(latent=False)
    setup = Problem2Setup(
        grid=grid, env=env, bc=bc, t_end=MAX_HORIZON, label=job["key"],
        x_D=float(job["x_D"]), x_hm=float(job["x_hm"]), x_h=float(job["x_h"]))

    t0 = time.time()
    tr, _, _ = locate_threshold(setup, dt_out=DT_OUT, max_horizon=MAX_HORIZON)
    wall = time.time() - t0

    rec = dict(job)
    rec.update({
        "found": bool(tr.found),
        "t_star_s": float(tr.t_star),
        "t_star_h": float(tr.t_star) / 3600.0 if tr.found else float("nan"),
        "C_max_at_tstar": float(tr.C_max_at),
        "r_argmax_m": float(tr.r_argmax),
        "n_bisect": int(tr.n_bisect),
        "bracket_s": [float(x) for x in tr.t_bracket],
        "bracket_src": tr.bracket_src,
        "atol_T": NUMERICS.atol_T, "atol_C": NUMERICS.atol_C,
        "N": N_GRID, "grading": GRADING, "wall_s": wall,
        "version": VERSION, "python": platform.python_version(),
    })
    P(f"    [{job['key']:>22s}] t* = {rec['t_star_h']:9.6f} h   "
      f"C_max(t*) = {rec['C_max_at_tstar']:.8f}   "
      f"r_argmax = {rec['r_argmax_m']*1000:.3f} mm   ({wall:.0f}s)")
    return rec


# ==========================================================================
# 作业表
# ==========================================================================
def build_jobs(with_extra=True):
    """基准 1 次 + 每个参数 6 小档 + （可选）3 大档。"""
    deltas = [s * d for d in LEVELS for s in (+1, -1)]
    if with_extra:
        deltas += [s * d for d in EXTRA_LEVELS for s in (+1, -1)]
        deltas.remove(-1.00)             # δ=−100%（D→0）无意义，去掉
    jobs = [dict(key="base", param="base", delta=0.0, mult=1.0,
                 x_D=1.0, x_hm=1.0, x_h=1.0)]
    for pname, field in PARAMS.items():
        for d in deltas:
            mult = 1.0 + d
            kw = dict(x_D=1.0, x_hm=1.0, x_h=1.0)
            kw[field] = mult
            # 键格式必须与首批一致，否则 json 里的已完成结果无法复用
            tag = f"{pname}_{'p' if d > 0 else 'm'}{abs(int(round(d*100))):02d}"
            jobs.append(dict(key=tag, param=pname, delta=float(d),
                             mult=float(mult), **kw))
    return jobs


def stage_param(workers: int):
    P("=" * 92)
    P(f"  审稿补做 ① 参数灵敏度  ±5% / ±10% / ±20%（+ 大扰动档对照）   版本 {VERSION}")
    P("=" * 92)
    from src.config import NUMERICS
    P(f"  网格 N={N_GRID}/γ={GRADING}   容差 atol_T={NUMERICS.atol_T:g} "
      f"atol_C={NUMERICS.atol_C:g}（全部运行同一套）")
    P(f"  求根 locate_threshold(dt_out={DT_OUT:g}, horizon={MAX_HORIZON/86400:.0f} d)")

    # ★ 增量执行：已有 json 里跑过的 key 直接复用（**必须同配置**才认）
    done = {}
    p_json = DOC / JSON_NAME
    if p_json.exists():
        old = json.loads(p_json.read_text(encoding="utf-8"))
        if (old.get("meta", {}).get("N") == N_GRID
                and old.get("meta", {}).get("grading") == GRADING):
            done = {r["key"]: r for r in old.get("runs", [])
                    if r.get("atol_T") == NUMERICS.atol_T and r.get("found")}
    jobs = build_jobs()
    todo = [j for j in jobs if j["key"] not in done]
    P(f"  作业共 {len(jobs)}，已完成可复用 {len(jobs)-len(todo)}，"
      f"本次实跑 {len(todo)}   并行度 {workers}")
    P("")

    recs, t0 = list(done.values()), time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, j): j["key"] for j in todo}
            for f in as_completed(futs):
                recs.append(f.result())
        P(f"\n  本批完成，耗时 {time.time()-t0:.0f} s")
    order = {j["key"]: i for i, j in enumerate(jobs)}
    recs.sort(key=lambda r: order.get(r["key"], 999))

    out = summarize(recs)
    out["meta"] = {
        "version": VERSION, "N": N_GRID, "grading": GRADING,
        "levels": list(LEVELS), "extra_levels": list(EXTRA_LEVELS),
        "params": list(PARAMS),
        "method": "problem3.locate_threshold（事件早停 + 固定锚点二分）",
        "note": "全部运行使用完全相同的时间/空间数值配置，差异只来自参数扰动",
    }
    dump(JSON_NAME, out)
    print_summary(out)
    return out


# ==========================================================================
# 汇总：灵敏度系数
# ==========================================================================
def summarize(recs):
    base = next(r for r in recs if r["key"] == "base")
    t0 = base["t_star_h"]
    tab, coeffs = [], []
    for r in recs:
        if r["key"] == "base":
            continue
        d, t = r["delta"], r["t_star_h"]
        dY = (t - t0) / t0                       # 相对响应 ΔY/Y
        dP = d                                   # 相对扰动 ΔP/P
        S = dY / dP                              # 灵敏度系数
        tab.append({
            "key": r["key"], "param": r["param"], "delta": d,
            "mult": r["mult"], "t_star_h": t, "dt_star_h": t - t0,
            "dt_star_s": (t - t0) * 3600.0,
            "rel_Y": dY, "S": S, "wall_s": r["wall_s"],
            "found": r["found"],
        })
    # 按参数聚合：中心差分系数 + 两档平均
    for pname in PARAMS:
        rows = [x for x in tab if x["param"] == pname]
        agg = {"param": pname, "rows": rows}
        for d in LEVELS:
            up = next(x for x in rows if abs(x["delta"] - d) < 1e-12)
            dn = next(x for x in rows if abs(x["delta"] + d) < 1e-12)
            # 中心差分：(Y(P+δ) − Y(P−δ)) / (2·Y0·δ)
            agg[f"S_cd_{int(d*100)}"] = (up["rel_Y"] - dn["rel_Y"]) / (2.0 * d)
            agg[f"dt_up_{int(d*100)}_h"] = up["dt_star_h"]
            agg[f"dt_dn_{int(d*100)}_h"] = dn["dt_star_h"]
            agg[f"absdt_{int(d*100)}_h"] = 0.5 * (
                abs(up["dt_star_h"]) + abs(dn["dt_star_h"]))
        coeffs.append(agg)
    coeffs.sort(key=lambda a: -a["absdt_20_h"])
    # 同一档位下的排序（龙卷风图的排序依据）
    rank = {int(d * 100): sorted(
        [{"param": a["param"], "halfspan_h": a[f"absdt_{int(d*100)}_h"]}
         for a in coeffs], key=lambda x: -x["halfspan_h"]) for d in LEVELS}
    return {"base": base, "runs": recs, "table": tab,
            "coeffs": coeffs, "rank": rank}


def rank20(out):
    """json 往返会把 int 键变成 str，这里两种都认。"""
    r = out["rank"]
    return r[20] if 20 in r else r["20"]


def print_summary(out):
    t0 = out["base"]["t_star_h"]
    P("")
    P("=" * 92)
    P("  灵敏度系数  S = (ΔY/Y)/(ΔP/P)，  Y = t*（问题3 达标时间）")
    P("=" * 92)
    P(f"  基准 t* = {t0:.6f} h = {t0*3600:.1f} s")
    P("")
    hdr = f"  {'参数':<6s}{'档位':>6s}{'倍率':>9s}{'t* / h':>12s}{'Δt* / h':>11s}" \
          f"{'Δt* / s':>11s}{'ΔY/Y':>10s}{'S':>9s}"
    P(hdr)
    P("  " + "-" * 88)
    for r in out["table"]:
        P(f"  {r['param']:<6s}{r['delta']*100:>+5.0f}%{r['mult']:>9.3f}"
          f"{r['t_star_h']:>12.6f}{r['dt_star_h']:>+11.4f}"
          f"{r['dt_star_s']:>+11.1f}{r['rel_Y']:>+10.5f}{r['S']:>+9.4f}")
    P("")
    P("  ── 中心差分灵敏度系数 S_cd = [Y(P+δ)−Y(P−δ)] / (2·Y0·δ) ──")
    P(f"  {'参数':<8s}{'δ=5%':>12s}{'δ=10%':>12s}{'δ=20%':>12s}"
      f"{'半幅|Δt*| @20%':>16s}")
    for a in out["coeffs"]:
        P(f"  {a['param']:<8s}{a['S_cd_5']:>+12.4f}{a['S_cd_10']:>+12.4f}"
          f"{a['S_cd_20']:>+12.4f}{a['absdt_20_h']:>+16.4f} h")
    P("")
    P("  ── 龙卷风排序（按 ±20% 半幅）──")
    for i, r in enumerate(rank20(out), 1):
        P(f"    {i}. {r['param']:<6s} ±{r['halfspan_h']:.4f} h")
    P("")
    P("  ⚠️ S 随 δ 变化即**响应非线性**的证据；报「灵敏度系数」必须注明档位。")


# ==========================================================================
# 图：龙卷风图 + S(δ) 曲线
# ==========================================================================
COLOR = {"D": "#d62728", "h_m": "#1f77b4", "h": "#2ca02c"}
MARK = {"D": "o", "h_m": "s", "h": "^"}


def stage_fig():
    p = DOC / JSON_NAME
    out = json.loads(p.read_text(encoding="utf-8"))
    from src.figures.plot_utils import save_fig, setup_style
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    setup_style()
    t0 = out["base"]["t_star_h"]
    tab = out["table"]
    coeffs = {a["param"]: a for a in out["coeffs"]}
    params = [r["param"] for r in rank20(out)]              # 已按 20% 排序

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8),
                             gridspec_kw={"width_ratios": [1.20, 1.05, 0.85]})

    # ---------------- (a) 嵌套龙卷风图（审稿人指定 ±5/10/20%）----------------
    # 画法：三档用**同一起点、不同条高**的横向条叠加（20% 最高、5% 最矮），
    # 形成"嵌套"外观，一眼同时读出**量级**与**档位**。
    ax = axes[0]
    colors = {5: "#c6dbef", 10: "#6baed6", 20: "#08519c"}
    heights = {20: 0.78, 10: 0.50, 5: 0.24}
    ypos = np.arange(len(params))[::-1]
    for d in (20, 10, 5):
        dn = np.array([coeffs[p][f"dt_dn_{d}_h"] for p in params])   # P−δ：t* 变大
        up = np.array([coeffs[p][f"dt_up_{d}_h"] for p in params])   # P+δ：t* 变小
        ax.barh(ypos, dn, height=heights[d], color=colors[d],
                edgecolor="#08306b", linewidth=0.6, zorder=2 + (20 - d),
                label=f"$\\pm${d}%")
        ax.barh(ypos, up, height=heights[d], color=colors[d],
                edgecolor="#08306b", linewidth=0.6, zorder=2 + (20 - d))
    ax.axvline(0.0, color="k", lw=1.2, zorder=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SYMBOL[p] for p in params], fontsize=13)
    ax.set_xlabel(r"$\Delta t^*$ / h （相对基准 $t^*$ = "
                  f"{t0:.2f} h）", fontsize=11)
    ax.set_title("(a) 龙卷风图：$\\pm$5%/10%/20% 对达标时间的影响",
                 fontsize=11.5)
    ax.legend(title="扰动幅度", fontsize=9, title_fontsize=9, loc="lower right")
    ax.grid(axis="y", visible=False)
    for p, y in zip(params, ypos):
        dn = coeffs[p]["dt_dn_20_h"]
        up = coeffs[p]["dt_up_20_h"]
        ax.annotate(f"{dn:+.2f} h", (dn, y), xytext=(4 if dn > 0 else -4, 0),
                    textcoords="offset points", ha="left" if dn > 0 else "right",
                    va="center", fontsize=8.5, color="#08306b")
        ax.annotate(f"{up:+.2f} h", (up, y), xytext=(4 if up > 0 else -4, 0),
                    textcoords="offset points", ha="left" if up > 0 else "right",
                    va="center", fontsize=8.5, color="#08306b")
    ax.margins(x=0.22)

    # ---------------- (b) 响应曲线 ΔY/Y vs δ（含大扰动档）----------------
    # 直线参考取 ±5% 处的中心差分斜率 —— 曲线偏离该直线的程度就是非线性的度量。
    ax = axes[1]
    for p in params:
        rows = sorted([r for r in tab if r["param"] == p], key=lambda r: r["delta"])
        xs = np.array([r["delta"] for r in rows])
        ys = np.array([r["rel_Y"] for r in rows])
        ax.plot(xs * 100.0, ys * 100.0, MARK[p] + "-", lw=2.0, ms=6,
                color=COLOR[p], label=SYMBOL[p])
        s5 = coeffs[p]["S_cd_5"]
        xr = np.array([-50.0, 50.0])
        ax.plot(xr, s5 * xr, ":", lw=1.3, color=COLOR[p], alpha=0.85)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"扰动幅度 $\delta$ / %", fontsize=11)
    ax.set_ylabel(r"相对响应 $\Delta t^*/t^*$ / %", fontsize=11)
    ax.set_title("(b) 响应曲线：实线为实测，点线为 $\\pm$5% 处线性外推",
                 fontsize=11.5)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # ---------------- (c) 灵敏度系数 S 随档位 ------------------
    ax = axes[2]
    ds = [5, 10, 20]
    w = 0.26
    for k, p in enumerate(params):
        vals = [coeffs[p][f"S_cd_{d}"] for d in ds]
        ax.bar(np.arange(3) + (k - 1) * w, vals, w, color=COLOR[p],
               label=SYMBOL[p], edgecolor="white", linewidth=0.6)
        for xi, v in zip(np.arange(3) + (k - 1) * w, vals):
            ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, -12 if v < 0 else 3),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color="white" if v < 0 else "#333333")
    # ±50% 档的单侧系数（大扰动档），单独用空心点叠加
    for k, p in enumerate(params):
        r50 = next((r for r in tab if r["param"] == p
                    and abs(r["delta"] + 0.5) < 1e-12), None)
        if r50:
            ax.plot([3 + (k - 1) * w], [r50["S"]], "o", ms=7, mec=COLOR[p],
                    mfc="white", mew=1.8)
    ax.axhline(0.0, color="k", lw=0.9)
    ax.set_xticks(list(range(4)))
    # 🔴 不要在这里手打 U+2212（−）：SimHei 缺该字形，会画成方框。
    ax.set_xticklabels(["5%", "10%", "20%", "50%\n(单侧, 减)"], fontsize=9.5)
    # 三个参数的 S 差 3 个量级（−0.0014 ~ −1.79），线性轴会把小两个压成 0。
    # 用 symlog：|S| < 0.01 段线性，之外对数，两端口径都读得出来。
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.45)
    ax.set_ylim(-3.0, 0.3)
    # symlog 的默认刻度用 mathtext（含 U+2212），而 SimHei 无该字形 → 手工给 ASCII 标签
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(r"灵敏度系数 $S=(\Delta Y/Y)/(\Delta P/P)$", fontsize=11)
    ax.set_title("(c) $S$ 随扰动档位变化（纵轴 symlog）", fontsize=11.5)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="both", alpha=0.25)

    fig.tight_layout()
    paths = save_fig(fig, "参数灵敏度龙卷风", "论文第6章 问题3·灵敏度分析")
    plt.close(fig)
    P(f"  → 已出图 {[str(x.name) for x in paths]}")
    return paths


# ==========================================================================
def dump(name, obj):
    DOC.mkdir(parents=True, exist_ok=True)
    def jd(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    p = DOC / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jd),
                 encoding="utf-8")
    P(f"\n  → {p}")
    return p


def stage_report():
    """从 json 重印汇总表（崩溃/只看结果时用，不重跑任何计算）。"""
    out = json.loads((DOC / JSON_NAME).read_text(encoding="utf-8"))
    print_summary(out)
    return out


def main():
    # 🔴 Windows 控制台默认 GBK，打印 U+2212(减号)/⇒/⚠ 会抛 UnicodeEncodeError。
    #    首次运行时确实因此在**所有计算都跑完之后**崩在了打印上（json 已正常写出）。
    #    统一把 stdout 改成 UTF-8 + replace，日志也顺带变成可读的 UTF-8。
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["param", "fig", "report", "all"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage in ("param", "all"):
        stage_param(a.workers)
    if a.stage in ("report", "all"):
        stage_report()
    if a.stage in ("fig", "all"):
        stage_fig()


if __name__ == "__main__":
    main()
