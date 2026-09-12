"""
问题1 差异归因（二）：边界条件取值时刻 / 环境分段常值     [诊断 · 非交付]

[A]/[B] 已排除"网格"与"插值形状"是主因：
      网格 N=40→320 与 γ=1.5 的全部组合，T(0,1800) 只在 33.5764~33.5768 间动（幅 4e-4）。
      公开解 33.5754 低于本项目的收敛值 33.5764 约 1.1e-3，属**系统性**偏移，非离散噪声。

本脚本检验两个能产生 ~1e-3 系统性偏移的**时间离散**选择：

  H1 边界取 t^{n+1}（本项目，与 Backward Euler 一致）
     vs 边界取 t^n  （显式边界，很多手写代码的自然写法）
     → 等效于把整条环境曲线平移半个步长。t≈1800 s 时 dT∞/dt≈6.3e-3 °C/s，
       平均步长 0.67 s ⇒ 驱动温差移动约 2e-3 °C，经表面热阻衰减后 ~1e-3 °C。

  H2 环境按**分段常值**（只用附件1 的 60 s 采样点，不插值）
     vs PCHIP 连续插值。锯齿状驱动同样造成 ~1e-3 量级的温度偏移。

用法：python scripts/check_q1_bc_level.py
"""

from __future__ import annotations

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
from src.numerics.time_stepper import AdaptiveStepper

T_END = 1800.0
PUBLIC = (33.5754, 36.7856)


def P(*a):
    print(*a, flush=True)


class StepEnv:
    """分段常值环境：用附件1 的 60 s 采样点做零阶保持（左值）。"""

    def __init__(self, data):
        self.t, self.T, self.C = data.t, data.T, data.C

    def _zoh(self, arr, t):
        i = np.searchsorted(self.t, t, side="right") - 1
        i = int(np.clip(i, 0, len(self.t) - 1))
        return arr[i]

    def T_inf(self, t):
        return self._zoh(self.T, t)

    def C_inf(self, t):
        return self._zoh(self.C, t)


def solve(env, grid, bc_level="new", theta=1.0, label=""):
    """
    bc_level: "new" → 边界用 t^{n+1}（与 BE 一致，本项目）
              "old" → 边界用 t^n  （显式边界）
    """
    num = NUMERICS
    alpha = PROPS_APP2.alpha
    alpha_cell = np.full(grid.N, alpha)
    h_bc = H_COEF / (PROPS_APP2.rho * PROPS_APP2.cp)
    op_t = assemble(alpha_cell, h_bc, 0.0, grid)

    tol = dict(atol_u=num.picard_atol_u, rtol_u=num.picard_rtol_u,
               atol_r=num.picard_atol_r, rtol_r=num.picard_rtol_r)

    surf = {"T": None}

    def step_fn(t, dt, state):
        tn = t + dt
        t_bc = tn if bc_level == "new" else t
        C_inf_new = float(env.C_inf(tn))
        Cnew, it, _ = picard_solve(
            lambda C: assemble(np.atleast_1d(D_app2(C)).astype(float),
                               HM_COEF, C_inf_new, grid),
            state["C"], dt, theta, tol, num.picard_max_iter)
        T_inf_bc = float(env.T_inf(t_bc))
        import copy
        op = copy.deepcopy(op_t)
        op.b[:-1] = 0.0
        op.b[-1] = op.beta * T_inf_bc
        Tnew = implicit_step(op, state["T"], dt, theta=theta)
        surf["T"] = (Tnew, T_inf_bc)
        return {"T": Tnew, "C": Cnew}, it

    st = AdaptiveStepper(num)
    res = st.integrate(0.0, {"T": np.full(grid.N, 28.0), "C": np.full(grid.N, 2.55)},
                       T_END, np.array([T_END]), step_fn, p_order=1.0)
    Tn, Tinf = surf["T"]
    Ts = surface_value(Tn, alpha_cell, h_bc, Tinf, grid)
    return float(res.T[-1][0]), float(Ts), res.n_accepted


def main():
    data = load_env()
    env_pchip = EnvInterpolator(data, mode="faithful")
    env_step = StepEnv(data)
    grid = Grid(N=200, R0=0.02, grading=1.5)

    P("=" * 96)
    P(f"  公开解 T(0)={PUBLIC[0]}  T(R0)={PUBLIC[1]}   |   本项目收敛值 T(0)=33.5764  T(R0)=36.7862")
    P("=" * 96)

    cases = [
        ("H0 边界 t^(n+1) + PCHIP（本项目生产）", env_pchip, "new", 1.0),
        ("H1 边界 t^n   + PCHIP（显式边界）", env_pchip, "old", 1.0),
        ("H2 边界 t^(n+1) + 分段常值环境", env_step, "new", 1.0),
        ("H1' 边界 t^n  + 分段常值环境", env_step, "old", 1.0),
        ("H3 Crank-Nicolson θ=0.5 + PCHIP", env_pchip, "new", 0.5),
    ]
    for label, env, lvl, th in cases:
        t0 = time.time()
        Tc, Ts, n = solve(env, grid, bc_level=lvl, theta=th)
        P(f"  {label:34s} θ={th:<4g} T(0)={Tc:9.4f} T(R0)={Ts:9.4f} "
          f"d={Tc-PUBLIC[0]:+.4f}/{Ts-PUBLIC[1]:+.4f}  steps={n:5d} {time.time()-t0:5.1f}s")
    P("=" * 96)


if __name__ == "__main__":
    main()
