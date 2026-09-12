"""
问题1 外部答案对照 / 差异归因        [甲 · 诊断脚本 · 非交付]

背景
----
网上公开解答给出 t=1800 s：
    T(r=0)   = 33.5754 °C
    T(r=2cm) = 36.7856 °C
本项目 result1.xlsx 给出：
    T(r=0)   = 33.5763 °C
    T(r=2cm) = 36.7861 °C

差 9e-4 / 5e-4 °C —— 落在 4 位小数的末位上。

本脚本回答两个问题
------------------
Q1  本项目的问题1 是否用了题面（附录2 / 附件1）之外的变量或公式？
Q2  那 9e-4 的差从哪来？是模型不同，还是离散/插值选择不同？

做法：把**每一个可自由选择的实现细节**单独拨动一次，
      看 T(0,1800) 与 T(R0,1800) 各自移动多少。
      若所有合理选择的移动幅度都能覆盖 9e-4，则属同一模型的数值差异，
      不是"用了不同的公式"。

用法
----
    python scripts/check_q1_vs_public.py            # 精简集（默认，约 5 分钟）
    python scripts/check_q1_vs_public.py --full     # 含均匀 N=400/640（很慢）
"""

from __future__ import annotations

import argparse
import dataclasses
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
from src.numerics.fvm_cyl import Grid
from scipy.interpolate import interp1d

T_END = 1800.0
PUBLIC = {"T_center": 33.5754, "T_surf": 36.7856}


def P(*a):
    print(*a, flush=True)


# ==========================================================================
class LinearEnv:
    """最朴素的线性插值 —— 很多公开解答用的就是这一档。"""

    def __init__(self, data):
        self._T = interp1d(data.t, data.T, kind="linear")
        self._C = interp1d(data.t, data.C, kind="linear")

    def _clip(self, t):
        return np.clip(t, 0.0, 1800.0)

    def T_inf(self, t):
        return self._T(self._clip(t))

    def C_inf(self, t):
        return self._C(self._clip(t))


# ==========================================================================
def run(env, N=200, grading=1.5, theta=1.0, dt_max=None, dt_fixed=None,
        label=""):
    """
    dt_fixed 给定时：**严格等步长**（模拟公开解答常见做法），
    通过把自适应容差放到极松、dt_init=dt_max=dt_fixed 实现。
    """
    import src.models.problem1 as p1

    num = NUMERICS
    if dt_fixed is not None:
        num = dataclasses.replace(NUMERICS, dt_init=dt_fixed, dt_max=dt_fixed,
                                  dt_min=dt_fixed,
                                  atol_T=1e9, rtol_T=1e9,
                                  atol_C=1e9, rtol_C=1e9)
    elif dt_max is not None:
        num = dataclasses.replace(NUMERICS, dt_max=dt_max)
    p1.NUMERICS = num

    grid = Grid(N=N, R0=0.02, grading=grading)
    setup = p1.Problem1Setup(grid=grid, env=env, theta=theta, t_end=T_END)
    t0 = time.time()
    res, extra = p1.solve_problem1(setup, t_output=np.array([T_END]))
    wall = time.time() - t0

    Tc = float(res.T[-1][0])
    Ts = float(extra["surface_T"][-1])
    P(f"  {label:36s} N={N:4d} g={grading:<4g} th={theta:<4g} "
      f"T(0)={Tc:9.4f} T(R0)={Ts:9.4f} "
      f"d={Tc-PUBLIC['T_center']:+.4f}/{Ts-PUBLIC['T_surf']:+.4f} "
      f"steps={res.n_accepted:6d} {wall:6.1f}s")
    p1.NUMERICS = NUMERICS
    return Tc, Ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    P("=" * 104)
    P("  问题1 外部答案对照 —— 逐个拨动实现细节")
    P(f"  网上公开解:  T(0,1800)={PUBLIC['T_center']}   T(R0,1800)={PUBLIC['T_surf']}")
    P("=" * 104)

    data = load_env()
    env_pchip = EnvInterpolator(data, mode="faithful")
    env_smooth = EnvInterpolator(data, mode="smoothed")
    env_lin = LinearEnv(data)

    P("\n[0] 题面给定的物性（本项目问题1 使用的全部参数）")
    P(f"    rho={PROPS_APP2.rho} kg/m3  cp={PROPS_APP2.cp} J/(kg.K)  "
      f"k={PROPS_APP2.k} W/(m.K)  h={H_COEF} W/(m2.K)  hm={HM_COEF} m/s")
    P(f"    D = 7e-9*exp(-0.89/C)   T0=28 C, C0=2.55 kg/kg,  R0=2 cm")
    P(f"    以上全部来自 附录2 / 题面 / 附件1，无外来参数。")

    P("\n[A] 环境边界插值方式（唯一可自由选择的物理输入处理）")
    run(env_pchip,  label="A1 PCHIP 保形（本项目生产）")
    run(env_lin,    label="A2 分段线性（公开解答常见）")
    run(env_smooth, label="A3 Savitzky-Golay 平滑 + PCHIP")

    P("\n[B] 空间网格")
    for N, g in ((40, 1.0), (80, 1.0), (160, 1.0), (320, 1.0),
                 (100, 1.5), (200, 1.5), (400, 1.5)):
        run(env_pchip, N=N, grading=g, label=f"B N={N} g={g} 均匀/渐变")
    if args.full:
        for N, g in ((400, 1.0), (640, 1.0)):
            run(env_pchip, N=N, grading=g, label=f"B N={N} g={g} 均匀")

    P("\n[C] 时间步长")
    for dtm in (1.0, 5.0, 60.0):
        run(env_pchip, dt_max=dtm, label=f"C dt_max={dtm:g}s 自适应")
    for dtf in (1.0, 10.0):
        run(env_pchip, dt_fixed=dtf, label=f"C dt={dtf:g}s 严格等步长")
    run(env_pchip, theta=0.5, label="C Crank-Nicolson th=0.5")

    P("=" * 104)


if __name__ == "__main__":
    main()
