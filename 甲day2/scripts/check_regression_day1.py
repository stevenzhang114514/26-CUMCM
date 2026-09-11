"""
Day2 回归检验：甲day2 的内核必须**逐位复现**甲day1 的 result1.xlsx

为什么必须做
------------
Day2 对 fvm_cyl.assemble 做了两项改动：
    ① 新增 cap_cell 参数（变物性传热用）
    ② 把逐单元的 Python 循环**向量化**（长时积分的性能前提）
这两项都可能悄悄改变数值。在往上叠问题2/3/4 之前，
必须先证明 **cap_cell=None 时的结果与 Day1 完全一致**。

三级检验（由弱到强）
--------------------
    L1  算子矩阵逐元素比对：向量化版 vs 原循环版（本文件内重建）
    L2  问题1 重算：甲day2 内核 vs 甲day1 的 result1.xlsx（逐格比对）
    L3  新增 cap_cell 的一致性：常数 cap 应等价于把 Γ 换成 Γ/cap

输出：docs/regression_day1.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NUMERICS, PROPS_APP2, H_COEF, HM_COEF, R_OUT_CM, DOC_DIR
from src.numerics.fvm_cyl import (Grid, assemble, face_diffusivity,
                                  surface_value, surface_flux)
from src.numerics.properties import D_app2
from src.data_prep.env_interp import load_env, EnvInterpolator
from src.io.resample import output_radii_m, sample_field
from src.models.problem1 import Problem1Setup, solve_problem1

DAY1_XLSX = ROOT.parent / "甲day1" / "results" / "M0" / "result1.xlsx"


# ==========================================================================
# L1：向量化装配 vs 原循环装配
# ==========================================================================
def assemble_loop_reference(Gamma_cell, h_bc, phi_inf, grid: Grid,
                            cap_cell=None):
    """Day1 原版的逐单元循环写法，作为向量化版的对照实现。"""
    N, R0 = grid.N, grid.R0
    rf, rc = grid.rf, grid.rc
    h = grid.h
    V = grid.V
    A = grid.A_face
    Gf = face_diffusivity(Gamma_cell)
    Vd = V if cap_cell is None else V * np.asarray(cap_cell, dtype=float)

    diag = np.zeros(N); lower = np.zeros(N); upper = np.zeros(N); b = np.zeros(N)
    c0 = A[1] * Gf[1] / (h[0] * Vd[0])
    diag[0] = -c0; upper[0] = +c0
    for i in range(1, N):
        a_in = A[i] * Gf[i] / (h[i - 1] * Vd[i])
        diag[i] -= a_in; lower[i] = a_in
        if i < N - 1:
            a_out = A[i + 1] * Gf[i + 1] / (h[i] * Vd[i])
            diag[i] -= a_out; upper[i] = a_out
        else:
            d = R0 - rc[i]
            if Gf[N] > 0.0 and d > 0.0:
                Bi_d = h_bc * d / Gf[N]
                beta = A[N] * h_bc / ((1.0 + Bi_d) * Vd[i])
            else:
                Bi_d = np.inf
                beta = A[N] * h_bc / Vd[i]
            diag[i] -= beta; b[i] = beta * phi_inf
    return diag, lower, upper, b, beta, Bi_d, Gf


def level1() -> dict:
    rng = np.random.default_rng(20260911)
    out = {"cases": [], "max_abs_diff": 0.0}
    for N, g, cap in [(40, 1.0, None), (200, 1.5, None), (137, 2.2, None),
                      (200, 1.5, "rand")]:
        grid = Grid(N=N, R0=0.02, grading=g)
        # 用跨越多个数量级的 Γ 制造最不利情形（与真实 D 同结构）
        base = 10.0 ** rng.uniform(-16, -7, size=N)
        cap_arr = None
        if cap == "rand":
            cap_arr = 10.0 ** rng.uniform(6.0, 7.0, size=N)
        o = assemble(base, 2.3e-5, 51.7, grid, cap_cell=cap_arr)
        ref = assemble_loop_reference(base, 2.3e-5, 51.7, grid, cap_cell=cap_arr)
        d = max(float(np.max(np.abs(o.diag - ref[0]))),
                float(np.max(np.abs(o.lower - ref[1]))),
                float(np.max(np.abs(o.upper - ref[2]))),
                float(np.max(np.abs(o.b - ref[3]))))
        # 用相对尺度归一（diag 量级可达 1e3）
        scale = float(np.max(np.abs(ref[0])))
        out["cases"].append({"N": N, "grading": g, "cap": str(cap),
                             "max_abs_diff": d, "rel": d / scale,
                             "bitwise_equal": d == 0.0})
        out["max_abs_diff"] = max(out["max_abs_diff"], d / scale)
    out["ok"] = out["max_abs_diff"] == 0.0
    return out


# ==========================================================================
# L3：cap_cell 的数学一致性
# ==========================================================================
def level3() -> dict:
    """
    常数 cap 时，带 cap_cell 的装配应等价于把 Γ 换成 Γ/cap：
        (ρc_p 常数)  ρc_p ∂T/∂t = ∇·(k∇T)  ⇔  ∂T/∂t = ∇·((k/ρc_p)∇T)
    这是 cap_cell 实现的**自洽性**检验（同时验证 h_bc 的换算口径）。
    """
    grid = Grid(N=80, R0=0.02, grading=1.2)
    rng = np.random.default_rng(7)
    k = 0.2 + 0.5 * rng.random(grid.N)
    cap = np.full(grid.N, 2.71e6)
    h = 25.0

    o_cap = assemble(k, h, 50.0, grid, cap_cell=cap)
    # Γ = α = k/(ρcp)；h_bc = h/(ρcp)。cap 为常数，故 h_bc 是标量。
    o_alpha = assemble(k / cap, h / float(cap[0]), 50.0, grid)
    d = max(float(np.max(np.abs(o_cap.diag - o_alpha.diag))),
            float(np.max(np.abs(o_cap.lower - o_alpha.lower))),
            float(np.max(np.abs(o_cap.upper - o_alpha.upper))),
            float(np.max(np.abs(o_cap.b - o_alpha.b))))
    scale = max(float(np.max(np.abs(o_alpha.diag))), 1e-300)
    return {"max_abs_diff": d, "rel": d / scale, "ok": d / scale < 1e-13}


# ==========================================================================
# L2：问题1 重算 vs 甲day1 result1.xlsx
# ==========================================================================
def level2() -> dict:
    import openpyxl
    if not DAY1_XLSX.exists():
        return {"ok": False, "detail": f"未找到 {DAY1_XLSX}"}

    data = load_env()
    env = EnvInterpolator(data, mode="faithful")
    num = NUMERICS
    grid = Grid(N=num.N, R0=0.02, grading=num.grading)
    setup = Problem1Setup(grid=grid, env=env, theta=num.theta,
                          t_end=1800.0)
    res, extra = solve_problem1(setup, t_output=np.arange(1.0, 1801.0))

    r_out = output_radii_m()
    T_out = sample_field(res.T, grid, extra["surface_T"], r_out)
    C_out = sample_field(res.C, grid, extra["surface_C"], r_out)

    wb = openpyxl.load_workbook(DAY1_XLSX, data_only=True)
    diffs = {}
    for sname, mine in (("温度", T_out), ("水分浓度", C_out)):
        ws = wb[sname]
        n_row = min(ws.max_row - 1, mine.shape[0])
        ref = np.array([[ws.cell(2 + i, 2 + j).value
                         for j in range(len(R_OUT_CM))] for i in range(n_row)],
                       dtype=float)
        d = np.abs(mine[:n_row] - ref)
        # 甲day1 写盘时已按 4 位小数舍入，故差异应与舍入量子 5e-5 同量级
        diffs[sname] = {"max_abs": float(d.max()),
                        "n_over_half_ulp": int((d > 5.0e-5).sum()),
                        "n_cells": int(d.size),
                        "mean_abs": float(d.mean())}
    ok = all(v["n_over_half_ulp"] == 0 for v in diffs.values())
    return {"ok": ok, "diffs": diffs,
            "note": "对照文件已舍入到 4 位小数，故判据是差异不超过 ±5e-5（半个舍入量子）"}


# ==========================================================================
def main():
    print("=" * 74)
    print("Day2 回归检验 —— 甲day2 内核 vs 甲day1")
    print("=" * 74)
    rep = {}
    for name, fn in (("L1_算子逐元素", level1),
                     ("L3_cap_cell自洽", level3),
                     ("L2_问题1重算", level2)):
        r = fn()
        rep[name] = r
        print(f"\n[{name}]  {'通过' if r['ok'] else '**未通过**'}")
        for k, v in r.items():
            if k != "ok":
                print(f"    {k}: {v}")
    rep["总体通过"] = all(v["ok"] for v in rep.values())
    print(f"\n总体: {'全部通过' if rep['总体通过'] else '**存在失败项**'}")

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    p = DOC_DIR / "regression_day1.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {p}")
    return 0 if rep["总体通过"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
