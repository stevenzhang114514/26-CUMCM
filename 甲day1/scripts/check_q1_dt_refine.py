"""
问题1 差异归因（三）：Δt → 0 时两种边界处理是否收敛到同一解   [诊断 · 非交付]

已知（见 check_q1_bc_level.py）：
    边界取 t^{n+1}（本项目）  T(0,1800) = 33.5765   T(R0,1800) = 36.7862
    边界取 t^n  （公开解）    T(0,1800) = 33.5754   T(R0,1800) = 36.7852
    公开答案 33.5754 / 36.7856 与之吻合到 4 位小数。

若这 1.1e-3 的差**纯粹是 O(Δt) 截断误差**（而非模型差异），
则把 Δt 逐步减半时：
    ① 两者的差应**按 Δt 一次方缩小**（比值 → 2）；
    ② 两者应**收敛到同一个值**。

这正是"同一模型、不同离散"的判据；若差是模型差异，缩小 Δt 不会让它们靠拢。

固定步长（不来自适应器），保证两种处理走的是**同一串时间网格**。

用法：python scripts/check_q1_dt_refine.py
"""

from __future__ import annotations

import copy
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import H_COEF, HM_COEF, NUMERICS, PROPS_APP2
from src.data_prep.env_interp import EnvInterpolator, load_env
from src.numerics.fvm_cyl import Grid, assemble, implicit_step, surface_value
from src.numerics.nonlinear import picard_solve
from src.numerics.properties import D_app2

T_END = 1800.0
P = lambda *a: print(*a, flush=True)


def run(env, grid, bc_level, dt):
    """严格等步长推进到 T_END。返回 (T_center, T_surf)。"""
    num = NUMERICS
    alpha = PROPS_APP2.alpha
    alpha_cell = np.full(grid.N, alpha)
    h_bc = H_COEF / (PROPS_APP2.rho * PROPS_APP2.cp)
    op_t = assemble(alpha_cell, h_bc, 0.0, grid)
    tol = dict(atol_u=num.picard_atol_u, rtol_u=num.picard_rtol_u,
               atol_r=num.picard_atol_r, rtol_r=num.picard_rtol_r)

    T = np.full(grid.N, 28.0)
    C = np.full(grid.N, 2.55)
    n = int(round(T_END / dt))
    assert abs(n * dt - T_END) < 1e-9 * T_END, f"dt={dt} 不能整除 {T_END}"

    t = 0.0
    T_inf_bc = 0.0
    for _ in range(n):
        tn = t + dt
        t_bc = tn if bc_level == "new" else t
        C, _, _ = picard_solve(
            lambda Cx: assemble(np.atleast_1d(D_app2(Cx)).astype(float),
                                HM_COEF, float(env.C_inf(tn)), grid),
            C, dt, 1.0, tol, num.picard_max_iter)
        op = copy.deepcopy(op_t)
        T_inf_bc = float(env.T_inf(t_bc))
        op.b[:-1] = 0.0
        op.b[-1] = op.beta * T_inf_bc
        T = implicit_step(op, T, dt, theta=1.0)
        t = tn
    return float(T[0]), float(surface_value(T, alpha_cell, h_bc, T_inf_bc, grid))


def main():
    env = EnvInterpolator(load_env(), mode="faithful")
    grid = Grid(N=200, R0=0.02, grading=1.5)

    P("=" * 92)
    P("  Δt → 0 收敛性：边界取 t^(n+1)（本项目） vs 边界取 t^n（公开解）")
    P("=" * 92)
    P(f"  {'Δt / s':>8} | {'T(0) 新':>10} {'T(0) 旧':>10} {'差':>10} {'比':>6} "
      f"| {'T(R0) 新':>10} {'T(R0) 旧':>10} {'差':>10} {'比':>6}")
    P("  " + "-" * 88)

    prev = None
    for dt in (2.0, 1.0, 0.5, 0.25, 0.125):
        t0 = time.time()
        cn, sn = run(env, grid, "new", dt)
        co, so = run(env, grid, "old", dt)
        d_c, d_s = cn - co, sn - so
        r_c = (prev[0] / d_c) if prev and d_c != 0 else float("nan")
        r_s = (prev[1] / d_s) if prev and d_s != 0 else float("nan")
        P(f"  {dt:8.3f} | {cn:10.5f} {co:10.5f} {d_c:10.2e} {r_c:6.2f} "
          f"| {sn:10.5f} {so:10.5f} {d_s:10.2e} {r_s:6.2f}   ({time.time()-t0:.0f}s)")
        prev = (d_c, d_s)
    P("=" * 92)
    P("  判读：差随 Δt 减半而减半（比≈2）且两列同步靠拢 ⇒ 纯 O(Δt) 截断差，")
    P("        不是模型差。外推到 Δt→0 即两法共同的真解。")


if __name__ == "__main__":
    main()
