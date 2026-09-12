"""
问题1 差异归因（四）：自适应容差 → 四位数末位的可信度      [诊断 · 非交付]

背景
----
Δt→0 外推给出问题1 的真解（N=200/γ=1.5）：
        T(0,1800)  = 33.5759
        T(R0,1800) = 36.7858
本项目 result1.xlsx 报 33.5763 / 36.7861 —— 偏高 4e-4 / 3e-4。
即：**当前生产容差 atol_T = 1e-5 的全局时间离散误差，正好落在第 4 位小数上。**

结论的用处
----------
题面要求"所有结果保留四位小数"。若时间离散误差 ~4e-4，
第 4 位小数就是不可信的。本脚本量化：容差收紧后能否兑现真解。

Backward Euler 的局部误差 ∝ Δt²，步长加倍法控制 |u_full−u_half| ≤ atol
⇒ Δt ∝ √atol，步数 ∝ 1/√atol。容差再收紧 100 倍 ⇒ 步数约 ×10。

用法：python scripts/check_q1_tolerance.py
"""

from __future__ import annotations

import dataclasses
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import NUMERICS
from src.data_prep.env_interp import EnvInterpolator, load_env
from src.numerics.fvm_cyl import Grid

T_END = 1800.0
TRUTH = (33.5759, 36.7858)        # Δt→0 外推（Richardson，两法一致）
PROD = (33.5763, 36.7861)         # 当前交付 result1.xlsx
P = lambda *a: print(*a, flush=True)


def main():
    import src.models.problem1 as p1

    env = EnvInterpolator(load_env(), mode="faithful")
    grid = Grid(N=200, R0=0.02, grading=1.5)

    P("=" * 96)
    P(f"  Δt→0 外推真解: T(0)={TRUTH[0]}  T(R0)={TRUTH[1]}")
    P(f"  当前交付值   : T(0)={PROD[0]}  T(R0)={PROD[1]}   (偏高 4e-4 / 3e-4)")
    P("=" * 96)
    P(f"  {'atol_T':>10} | {'T(0)':>10} {'偏差':>9} | {'T(R0)':>10} {'偏差':>9} "
      f"| {'步数':>7} {'耗时':>7}")
    P("  " + "-" * 78)

    for atol in (1e-5, 1e-6, 1e-7, 1e-8):
        num = dataclasses.replace(NUMERICS, atol_T=atol, rtol_T=atol,
                                  atol_C=atol * 1e-3, rtol_C=atol * 1e-3)
        p1.NUMERICS = num
        setup = p1.Problem1Setup(grid=grid, env=env, theta=1.0, t_end=T_END)
        t0 = time.time()
        res, extra = p1.solve_problem1(setup, t_output=np.array([T_END]))
        wall = time.time() - t0
        Tc, Ts = float(res.T[-1][0]), float(extra["surface_T"][-1])
        P(f"  {atol:10.0e} | {Tc:10.5f} {Tc-TRUTH[0]:+9.2e} | {Ts:10.5f} "
          f"{Ts-TRUTH[1]:+9.2e} | {res.n_accepted:7d} {wall:6.0f}s")
    p1.NUMERICS = NUMERICS
    P("=" * 96)


if __name__ == "__main__":
    main()
