"""
问题4 主运行脚本                      [甲 · M0 · 接替乙的问题4]

产出
----
    results/result4.xlsx            正式交付（Sheet1，60 s × 0.1 cm，末列「药材表面」）
    results/result4_运行信息.xlsx   配置与版本标识
    docs/表6_问题4.md                题设表6
    docs/p4_core.json               数值证据
    docs/p4_grid.json               网格收敛（与乙的 N=20 对照）
    docs/p4_rsens.json              半径外推敏感性（CODE-05）

用法
----
    python scripts/run_p4.py --stage core     # 主结果（关键路径）
    python scripts/run_p4.py --stage grid     # 网格收敛（证明乙的 N=20 不够）
    python scripts/run_p4.py --stage rsens    # 半径外推敏感性
    python scripts/run_p4.py --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DOC_DIR, NUMERICS, R_OUT_CM, T_TARGET  # noqa: E402
from src.data_prep.env_extrap import build_env               # noqa: E402
from src.data_prep.env_interp import load_env                # noqa: E402
from src.models.problem4 import GridXiN, load_radius, solve_p4  # noqa: E402

RESULT_DIR = ROOT / "results"
TDRY_TARGET = 0.15


def P(*a):
    print(*a, flush=True)


def dump(name, obj):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    p = DOC_DIR / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    P(f"    → docs/{name}")


def make_env():
    return build_env(load_env(), mode="faithful", t_pre=14400.0)


# ==========================================================================
def find_tdry(env, rad, grid, t_max=6 * 86400.0, dt_out=300.0, num=None,
              R_mode="pchip", verbose=False):
    """二分/扫描定位 C_max 首次跌破 0.15 的时刻。"""
    t_out = np.arange(dt_out, t_max + dt_out, dt_out)
    r = solve_p4(env, rad, grid, t_max, t_out, num=num, R_mode=R_mode,
                 verbose=verbose)
    Cm = r["C_max"]
    below = Cm < TDRY_TARGET
    if not np.any(below):
        return float("nan"), r
    k = int(np.argmax(below))
    return float(r["times"][k]), r


# ==========================================================================
def stage_core(args):
    P("=" * 88)
    P("  问题4 主体 —— 材料坐标 + 附件2 收缩 + 附录4 物性")
    P("=" * 88)
    env, rad = make_env(), load_radius()
    P(f"  {rad.summary()}")
    grid = GridXiN(N=args.N, grading=args.gamma)
    P(f"  {grid.summary()}")
    num = NUMERICS
    out = {"网格": grid.summary(), "半径": rad.summary(),
           "物性": "附录4", "环境": "附件1 全程 PCHIP，14400 s 后取平台均值"}

    t0 = time.time()
    tdry, r = find_tdry(env, rad, grid, num=num, verbose=True)
    P(f"\n  ★ 烘干时长 t_dry = {tdry:.1f} s = {tdry/3600:.4f} h = "
      f"{tdry/86400:.4f} 天   （{time.time()-t0:.0f}s）")
    out["t_dry_s"] = tdry
    out["t_dry_h"] = tdry / 3600.0
    out["t_dry_day"] = tdry / 86400.0

    # ---- 末时刻场 + 输出采样 ----
    R_f = rad(tdry)
    P(f"  终态半径 R(t_dry) = {R_f*100:.4f} cm")
    out["R_final_cm"] = R_f * 100

    # 重新积分到 tdry 的**整秒**，并按 60 s 输出（正式交付）
    tdry_int = int(np.ceil(tdry))
    t_out = np.arange(60, tdry_int + 1, 60, dtype=float)
    if t_out[-1] < tdry_int:
        t_out = np.append(t_out, float(tdry_int))
    P(f"  重跑正式输出网格：{len(t_out)} 个时刻（60 s 间隔，末点 {t_out[-1]:.0f} s）")
    t0 = time.time()
    res = solve_p4(env, rad, grid, float(tdry_int), t_out, num=num)
    P(f"     完成（{time.time()-t0:.0f}s，{len(res['times'])} 个时刻）")

    # ---- 采样到物理坐标 ----
    r_out_cm = np.array(R_OUT_CM, float)                  # 0, 0.1, ..., 2.0
    r_out_cm = np.array(R_OUT_CM[:20], float)   # result4 距离轴只到 1.9 cm
    Cmat = np.full((len(res["times"]), len(r_out_cm)), np.nan)
    Csurf = np.empty(len(res["times"]))
    xic = grid.xi          # 节点式网格：场量就存在节点上（ξ_0…ξ_{N-1}，末节点=1）
    for i, tk in enumerate(res["times"]):
        Rk = rad(tk)
        xi_q = (r_out_cm * 1e-2) / Rk
        m = xi_q <= 1.0 + 1e-12
        Cmat[i, m] = np.interp(xi_q[m], xic, res["C"][i])
        Csurf[i] = res["C"][i][-1]
    out["末行有效列数"] = int(np.sum(~np.isnan(Cmat[-1])))
    out["C_center_final"] = float(Cmat[-1, 0])
    out["C_surf_final"] = float(Csurf[-1])
    out["C_max_final"] = float(res["C"][-1].max())

    # ---- 写 result4.xlsx ----
    from src.io.excel_writer import write_result4
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "层次": "M0（题设基线）", "问题": "问题4（考虑尺寸变化）",
        "模型": "材料坐标 ξ=r/R(t) 半控制体 FVM；**无 Ṙ 对流项**（材料导数形式）",
        "物性": "附录4：ρ=760+90C，cp=1850+2150C/(1+C)，k=0.12+0.20C/(1+C)，"
                "D=4.2e-4·exp(-0.30/C)·exp(-3850/T_K)",
        "半径": f"附件2 PCHIP（{len(rad.t)} 点，0—{rad.t_max:.0f} s）",
        "网格": grid.summary(),
        "时间格式": "θ=1.0（Backward Euler，L-稳定）+ 块 Gauss-Seidel Picard",
        "判据": "C_max(t)=max_ξ C < 0.15 kg/kg（未舍入判定）",
        "烘干时长 t_dry": f"{tdry:.4f} s = {tdry/3600:.4f} h = {tdry/86400:.4f} 天",
        "终态半径": f"{R_f*100:.4f} cm",
        "版本标识": "A-2026-M0-P4-r1",
    }
    p = RESULT_DIR / "result4.xlsx"
    write_result4(p, res["times"].astype(int), Cmat, Csurf, meta=meta)
    P(f"    → results/result4.xlsx（{len(res['times'])} 行 × {len(r_out_cm)+2} 列）")

    from src.io.excel_writer import validate_result4
    rep = validate_result4(p, expect_t=res["times"].astype(int))
    P(f"    回读自检：{'通过' if rep.get('ok') else '★未通过★'}  {rep}")
    out["输出自检"] = bool(rep.get("ok"))

    # ---- 表6 ----
    rows = []
    for hh in range(6, int(tdry / 3600) + 1, 6):
        k = int(np.searchsorted(res["times"], hh * 3600))
        if k >= len(res["times"]):
            break
        rows.append((hh, Cmat[k], Csurf[k], rad(res["times"][k]) * 100))
    md = ["# 表6　药材烘干过程的水分浓度（问题4）", "",
          f"> t_dry = {tdry/3600:.4f} h = {tdry/86400:.4f} 天；"
          f"网格 {grid.summary()}", "",
          "| 时间/h | 0 | 0.5 | 1 | 1.5 | 2 | 药材表面 | 半径/cm |",
          "|---|---|---|---|---|---|---|---|"]
    for hh, cm, cs, Rc in rows:
        cells = []
        for j, rcm in enumerate(r_out_cm):
            v = cm[j]
            cells.append("—" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}")
        # 只保留 0/0.5/1/1.5/2 五列
        sel = [0, 5, 10, 15, 20]
        md.append(f"| {hh} | " + " | ".join(f"`{cells[i]}`" for i in sel)
                  + f" | {cs:.4f} | {Rc:.4f} |")
    md.append(f"| **{tdry/3600:.4f}（烘干结束）** | ")
    P("")
    for line in md[-len(rows) - 1:]:
        P("  " + line)
    (DOC_DIR / "表6_问题4.md").write_text("\n".join(md), encoding="utf-8")
    P("    → docs/表6_问题4.md")

    dump("p4_core.json", out)
    return out


# ==========================================================================
def stage_grid(args):
    P("=" * 88)
    P("  网格收敛 —— 证明「均匀 N=20」不够（乙用 N=20，本实现用渐变）")
    P("=" * 88)
    env, rad = make_env(), load_radius()
    num = dataclasses.replace(NUMERICS, dt_max=20.0) if False else NUMERICS
    out = {"行": []}
    P(f"  {'配置':>22} | {'t_dry / h':>11} | {'相对 N=160':>11} | {'耗时':>7}")
    P("  " + "-" * 62)
    ref = None
    cases = [(20, 1.0), (40, 1.0), (80, 1.0), (160, 1.0),
             (20, 1.5), (40, 1.5), (80, 2.0), (200, 1.5)]
    for N, g in cases:
        gr = GridXiN(N=N, grading=g)
        t0 = time.time()
        td, _ = find_tdry(env, rad, gr, num=num, dt_out=600.0)
        el = time.time() - t0
        if N == 160 and g == 1.0:
            ref = td
        rel = (td - ref) / ref * 100 if ref else float("nan")
        P(f"  N={N:4d} γ={g:<4} 均匀/渐变 | {td/3600:11.4f} | {rel:+10.2f}% | {el:6.0f}s")
        out["行"].append({"N": N, "grading": g, "t_dry_s": td, "t_dry_h": td/3600,
                          "rel_pct": rel, "wall_s": el})
    dump("p4_grid.json", out)
    return out


# ==========================================================================
def stage_rsens(args):
    P("=" * 88)
    P("  半径外推敏感性（CODE-05）—— t_dry 超出附件2 覆盖范围时")
    P("=" * 88)
    env, rad = make_env(), load_radius()
    grid = GridXiN(N=args.N, grading=args.gamma)
    P(f"  附件2 覆盖到 {rad.t_max/3600:.1f} h；超出后需外推")
    out = {"附件2_覆盖_h": rad.t_max / 3600, "行": []}
    P(f"  {'外推方式':>14} | {'t_dry / h':>11}")
    P("  " + "-" * 30)
    for mode in ("pchip", "const_end", "linear"):
        td, _ = find_tdry(env, rad, grid, R_mode=mode, dt_out=600.0)
        P(f"  {mode:>14} | {td/3600:11.4f}")
        out["行"].append({"mode": mode, "t_dry_s": td, "t_dry_h": td/3600})
    ts = [r["t_dry_h"] for r in out["行"]]
    out["最大相对差_pct"] = (max(ts) - min(ts)) / min(ts) * 100
    P(f"\n  三种外推的最大相对差 = {out['最大相对差_pct']:.2f}%")
    dump("p4_rsens.json", out)
    return out


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="core",
                    choices=["core", "grid", "rsens", "all"])
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=1.5)
    args = ap.parse_args()
    t0 = time.time()
    P(f"问题4 运行时：Python {sys.version.split()[0]}，numpy {np.__version__}，"
      f"网格 N={args.N}/γ={args.gamma}")
    if args.stage in ("core", "all"):
        stage_core(args)
    if args.stage in ("grid", "all"):
        stage_grid(args)
    if args.stage in ("rsens", "all"):
        stage_rsens(args)
    P(f"\n总耗时 {time.time()-t0:.1f} s")


if __name__ == "__main__":
    import dataclasses
    main()
