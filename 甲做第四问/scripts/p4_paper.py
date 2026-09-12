"""
问题4 论文口径补充计算                [甲 · 验证脚本]

补三样论文正文/插图需要、但现有产物里口径不对或缺失的东西：

    stage table6  表6（**Δt=1 s 收敛口径**）+ 终态场 + G11 的 e_d(t) 序列
                  现有 docs/表6_问题4.md 是 60 s 输出网格上的 51.1667 h，
                  而论文声明的收敛值是 Δt=1 s 的 **51.0905 h**，两者不一致。
    stage fixB    B 方案：固定半径 R₀ + 附录4 → t_dry
                  A/B/C 顺序差分里的 B 档现在填的是乙声称的「>120 h」，
                  需要用本项目自己的代码在**同一套数值口径**下重算。
    stage all

为什么 B 方案要重算
------------------
A/B/C 的作用是"顺序差分"：A（附录3+固定）→ B（附录4+固定）→ C（附录4+收缩）。
三者必须在**同一网格、同一时间步、同一判据**下比较，差出来的才是物性/几何的贡献。
乙的「>120 h」来自 N=20 的欠分辨网格，与本项目的 N=200/γ=1.5 不可比，
直接放进同一张表会让"物性变化的贡献"被网格误差污染。

用法
----
    py scripts/p4_paper.py --stage table6
    py scripts/p4_paper.py --stage fixB
    py scripts/p4_paper.py --stage all
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DOC_DIR, NUMERICS, T_TARGET          # noqa: E402
from src.data_prep.env_extrap import build_env              # noqa: E402
from src.data_prep.env_interp import load_env               # noqa: E402
from src.models.problem4 import GridXiN, load_radius, solve_p4  # noqa: E402
from src.numerics.properties import rho_app4                # noqa: E402

P = lambda *a: print(*a, flush=True)

N_GRID, GAMMA = 200, 1.5
DT_CONV = 1.0                      # 论文声明的收敛时间步
R0 = 0.02                          # m，初始半径（B 方案固定用）
COL_CM = (0.0, 0.5, 1.0, 1.5)      # 表6 的四个位置列


def make_env():
    return build_env(load_env(), mode="faithful", t_pre=14400.0)


def conv_num():
    return dataclasses.replace(NUMERICS, dt_max=DT_CONV, dt_init=DT_CONV)


def dump(name, obj):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    p = DOC_DIR / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    P(f"    → docs/{name}")
    return p


# ==========================================================================
def ed_trace(grid, C_row, R):
    """
    几何—密度一致性诊断量 e_d(t) 与"条件性下界"所需量。

        M_d(t) = 2πL∫₀^{R(t)} [ρ(C)/(1+C)] r dr
        材料坐标下 r=ξR, dr=R dξ ⇒ M_d = 2πL·R²·Σ_i w_i·ρ(C_i)/(1+C_i)

    其中 w_i = ∫_{cell i} ξ dξ 正是 GridXiN 的节点权重（0.5*(f[i+1]²−f[i]²)），
    已含 ξ 与 dξ 两重因子，故无需再乘。
    """
    rho_d = np.asarray(rho_app4(C_row), float) / (1.0 + np.asarray(C_row, float))
    return float(R ** 2 * np.sum(grid.w * rho_d))


def stage_table6(args):
    P("=" * 88)
    P("  表6（Δt=1 s 收敛口径）+ 终态场 + e_d(t)")
    P("=" * 88)
    env, rad = make_env(), load_radius()
    grid = GridXiN(N=args.N, grading=args.gamma)
    num = conv_num()
    P(f"  {grid.summary()}")
    P(f"  时间：Δt_max = {DT_CONV:g} s（论文口径），θ=1")
    P(f"  {rad.summary()}\n")

    # 输出网格：6 h 的整数倍 + 终点留白，末点由 stop_below 追加为**达标时刻本身**
    t_out = np.arange(6 * 3600.0, 72 * 3600.0, 6 * 3600.0)
    t0 = time.time()
    res = solve_p4(env, rad, grid, 72 * 3600.0, t_out, num=num,
                   stop_below=T_TARGET, verbose=True)
    wall = time.time() - t0
    ts = res["times"]
    t_dry = float(ts[-1])
    P(f"\n  ★ Δt=1 s 收敛口径 t_dry = {t_dry:.4f} s = {t_dry/3600:.6f} h "
      f"= {t_dry/86400:.4f} 天   （{wall:.0f}s，{len(ts)} 个输出时刻）")
    P(f"    对照：probe 法（p4_time.json）= 51.090506 h；"
      f"60 s 网格旧值 = 51.1667 h")

    # ---- 参考干物质量 M_d(0)：**解析给出，不取第一条输出** ----
    # 🔴 这里踩过一次：最初把 Md0 取成 `res["C"][0]` 算出的值，
    #    而输出网格的首点是 **t=6 h**（不是 t=0）。6 h 时半径已收缩、
    #    表层已失水，M_d(6h) < M_d(0)，于是此后所有 e_d 全变成**正**的
    #    （实测 +0.00~+0.24），与"干物质守恒"的符号直觉相反。
    #    初值均匀（C≡2.55），故 M_d(0) 有闭式：
    #        M_d(0) = 2πL·R₀²·ρ_d(C₀)·∫₀¹ξdξ = 2πL·R₀²·ρ_d(C₀)/2
    #    🔴 系数里**必须带 R₀²** —— ed_trace 返回的是 R²·Σ(w·ρ_d)，
    #       而 Σw = ∫₀¹ξdξ = 1/2。首版把 Md0 写成 0.5·ρ_d0 漏了 R₀²，
    #       与 ed_trace 的约定不一致，量级差 4×10⁻⁴，assert 直接拦下。
    rho_d0_start = float(rho_app4(2.55)) / 3.55
    Md0 = R0 ** 2 * 0.5 * rho_d0_start           # 2πL 为公共因子，作比值时约掉
    assert abs(ed_trace(grid, np.full(grid.N, 2.55), R0) - Md0) < 1e-12 * Md0, \
        "M_d(0) 的数值积分与解析式不符"

    # ---- 采样到物理坐标 ----
    xic = grid.xi
    Cmat = np.full((len(ts), len(COL_CM)), np.nan)
    Rof = np.empty(len(ts))
    ed = np.empty(len(ts))
    for i, tk in enumerate(ts):
        Rk = rad(tk)
        Rof[i] = Rk
        xq = np.array(COL_CM, float) * 1e-2 / Rk
        m = xq <= 1.0 + 1e-12
        Cmat[i, m] = np.interp(xq[m], xic, res["C"][i])
        ed[i] = ed_trace(grid, res["C"][i], Rk) / Md0 - 1.0
    Csurf = res["C"][:, -1]
    Cmax = res["C_max"]

    P(f"\n  e_d(t) 范围 = [{ed.min():+.4f}, {ed.max():+.4f}]")
    P(f"  终态半径 R(t_dry) = {Rof[-1]*100:.4f} cm，C_max = {Cmax[-1]:.8f}")

    # ---- 条件性下界 ----
    rho_d0 = float(rho_app4(2.55)) / 3.55
    R_bound = R0 * np.sqrt(rho_d0 / 760.0)
    P(f"  ρ_d,0（附录4，C=2.55）= {rho_d0:.4f} kg/m³")
    P(f"  条件性下界 R_f ≥ R₀√(ρ_d,0/760) = {R_bound*100:.4f} cm；"
      f"实测 {Rof[-1]*100:.4f} cm，差 {(R_bound-Rof[-1])*100:+.4f} cm")

    # ---- 表6 ----
    rows = []
    for i, tk in enumerate(ts):
        rows.append((tk / 3600.0, Cmat[i], Csurf[i], Rof[i] * 100, Cmax[i]))
    md = ["# 表6　药材烘干过程的水分浓度（问题4）", "",
          f"> **Δt = {DT_CONV:g} s（论文声明的收敛口径）**；"
          f"t_dry = {t_dry/3600:.6f} h = {t_dry/86400:.4f} 天",
          f"> 网格 {grid.summary()}",
          f"> 终态半径 {Rof[-1]*100:.4f} cm；判据 C_max(t) < {T_TARGET}", "",
          "| 时间/h | 0 cm | 0.5 cm | 1.0 cm | 1.5 cm | 药材表面 | 半径/cm |",
          "|---|---|---|---|---|---|---|"]
    for hh, cm, cs, Rc, _ in rows:
        cells = ["—" if np.isnan(v) else f"{v:.4f}" for v in cm]
        tag = f"**{hh:.4f}（烘干结束）**" if abs(hh * 3600 - t_dry) < 1.0 else f"{hh:g}"
        md.append(f"| {tag} | " + " | ".join(cells)
                  + f" | {cs:.4f} | {Rc:.4f} |")
    (DOC_DIR / "表6_问题4.md").write_text("\n".join(md), encoding="utf-8")
    P(f"    → docs/表6_问题4.md（{len(rows)} 行）")

    out = {
        "时间步": DT_CONV, "网格": grid.summary(),
        "t_dry_s": t_dry, "t_dry_h": t_dry / 3600.0, "t_dry_day": t_dry / 86400.0,
        "R_final_cm": float(Rof[-1] * 100), "C_max_final": float(Cmax[-1]),
        "e_d_min": float(ed.min()), "e_d_max": float(ed.max()),
        "rho_d0": rho_d0, "R_bound_cm": float(R_bound * 100),
        "R_bound_gap_cm": float((R_bound - Rof[-1]) * 100),
        "表6行": [[float(h), *[None if np.isnan(v) else float(v) for v in cm],
                  float(cs), float(Rc)]
                 for h, cm, cs, Rc, _ in rows],
        "e_d_trace": {"t_h": [float(x) for x in ts / 3600.0],
                      "e_d": [float(x) for x in ed],
                      "C_max": [float(x) for x in Cmax],
                      "R_cm": [float(x * 100) for x in Rof]},
        "对照": {"probe法": 51.090506, "60s网格旧值": 51.1667,
                 "参考答案": 51.0823},
        "wall_s": wall, "版本标识": "A-2026-M0-P4-r2",
    }
    dump("p4_paper_table6.json", out)
    return out


# ==========================================================================
def stage_fixB(args):
    P("=" * 88)
    P("  B 方案：固定半径 R₀ + 附录4 物性（A/B/C 顺序差分的中间档）")
    P("=" * 88)
    env, rad = make_env(), load_radius()
    grid = GridXiN(N=args.N, grading=args.gamma)
    num = conv_num()
    P(f"  R 固定为 {R0*100:g} cm（不收缩）；其余同问题4\n")

    # 🔴 arange 的右端是**开区间**：写成 `np.arange(a, 120*3600, 6*3600)`
    #    时最后一个输出点是 114 h，**不含 120 h**。首次运行因此只记录到 114 h，
    #    却把结论写成"120 h 内未达标"——真实含义是"114 h 时仍未达标"。
    #    加 `+1` 把 120 h 纳入输出网格。
    HORIZON_H = 120.0
    t_out = np.arange(6 * 3600.0, HORIZON_H * 3600.0 + 1.0, 6 * 3600.0)
    t0 = time.time()
    res = solve_p4(env, rad, grid, HORIZON_H * 3600.0, t_out, num=num,
                   R_override=R0, stop_below=T_TARGET)
    wall = time.time() - t0
    ts = res["times"]
    tB = float(ts[-1])
    reached = bool(res["C_max"][-1] < T_TARGET)
    P(f"\n  ★ B 方案末记录时刻 {tB/3600:.1f} h，C_max = {res['C_max'][-1]:.6f}"
      f"   （{wall:.0f}s）")
    P(f"    达标 = {reached}"
      + ("" if reached else f"  ⇒ **{HORIZON_H:g} h 内未达标**"))
    if reached:
        P(f"    t_dry = {tB:.1f} s = {tB/3600:.4f} h")
    P(f"  对照：A = 57.4889 h（附录3+固定，问题3）   C = {51.090506:.4f} h"
      f"（附录4+收缩，问题4）")

    out = {"R_override_cm": R0 * 100, "时间步": DT_CONV, "网格": grid.summary(),
           "t_last_s": tB, "t_last_h": tB / 3600.0,
           "t_dry_s": (tB if reached else None),
           "t_dry_h": (tB / 3600.0 if reached else None),
           "reached": reached,
           "C_max_at_tlast": float(res["C_max"][-1]),
           "A_h": 57.4889, "C_h": 51.090506, "wall_s": wall,
           "horizon_h": HORIZON_H,
           "版本标识": "A-2026-M0-P4-r2"}
    dump("p4_paper_fixB.json", out)
    return out


# ==========================================================================
def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["table6", "fixB", "all"])
    ap.add_argument("--N", type=int, default=N_GRID)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    a = ap.parse_args()
    if a.stage in ("table6", "all"):
        stage_table6(a)
    if a.stage in ("fixB", "all"):
        stage_fixB(a)


if __name__ == "__main__":
    main()
