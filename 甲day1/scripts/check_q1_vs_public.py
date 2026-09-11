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
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import H_COEF, HM_COEF, PROPS_APP2
from src.data_prep.env_interp import EnvInterpolator, load_env
from src.models.problem1 import Problem1Setup, solve_problem1
from src.numerics.fvm_cyl import Grid
from scipy.interpolate import interp1d

T_END = 1800.0
PUBLIC = {"T_center": 33.5754, "T_surf": 36.7856}


# ==========================================================================
# 线性插值的环境（对照 PCHIP）
# ==========================================================================
class LinearEnv:
    """最朴素的线性插值 + 端点截断 —— 很多公开解答用的就是这一档。"""

    def __init__(self, data):
        self._T = interp1d(data.t, data.T, kind="linear")
        self._C = interp1d(data.t, data.C, kind="linear")

    def T_inf(self, t):
        return float(self._T(np.clip(t, 0.0, 1800.0))) if np.isscalar(t) \
            else self._T(np.clip(t, 0.0, 1800.0))

    def C_inf(self, t):
        return float(self._C(np.clip(t, 0.0, 1800.0))) if np.isscalar(t) \
            else self._C(np.clip(t, 0.0, 1800.0))


# ==========================================================================
def run(env, N=200, grading=1.5, theta=1.0, dt_max=None, label=""):
    import dataclasses
    from src.config import NUMERICS
    num = NUMERICS
    if dt_max is not None:
        num = dataclasses.replace(NUMERICS, dt_max=dt_max)
        import src.models.problem1 as p1
        p1.NUMERICS = num
    else:
        import src.models.problem1 as p1
        p1.NUMERICS = NUMERICS
    grid = Grid(N=N, R0=0.02, grading=grading)
    setup = Problem1Setup(grid=grid, env=env, theta=theta, t_end=T_END)
    res, extra = solve_problem1(setup, t_output=np.array([T_END]))
    Tc = float(res.T[-1][0])
    Ts = float(extra["surface_T"][-1])
    print(f"  {label:38s} N={N:4d} γ={grading:<4g} θ={theta:<4g} "
          f"T(0)={Tc:9.4f}  T(R0)={Ts:9.4f}  "
          f"Δ={Tc-PUBLIC['T_center']:+.4f}/{Ts-PUBLIC['T_surf']:+.4f}  "
          f"步数={res.n_accepted}")
    return Tc, Ts


def main():
    print("=" * 100)
    print("  问题1 外部答案对照 —— 逐个拨动实现细节")
    print(f"  网上公开解:  T(0,1800)={PUBLIC['T_center']}   T(R0,1800)={PUBLIC['T_surf']}")
    print("=" * 100)

    data = load_env()
    env_pchip = EnvInterpolator(data, mode="faithful")
    env_smooth = EnvInterpolator(data, mode="smoothed")
    env_lin = LinearEnv(data)

    print("\n【0】题面给定的物性（本项目使用的全部参数）")
    print(f"  ρ={PROPS_APP2.rho} kg/m³  cp={PROPS_APP2.cp} J/(kg·K)  "
          f"k={PROPS_APP2.k} W/(m·K)  h={H_COEF} W/(m²·K)  h_m={HM_COEF} m/s")
    print(f"  D = 7e-9·exp(-0.89/C)   初值 T=28 °C, C=2.55 kg/kg   R0=2 cm")
    print("  以上**全部**来自附录2 / 题面，无外来参数。")

    print("\n【A】环境边界插值方式（唯一可能引入差异的自由选择）")
    run(env_pchip,  label="A1 PCHIP 保形（本项目生产）")
    run(env_lin,    label="A2 分段线性（常见公开做法）")
    run(env_smooth, label="A3 Savitzky-Golay 平滑 + PCHIP")

    print("\n【B】网格离散")
    for N, g in ((50, 1.0), (100, 1.0), (200, 1.0), (400, 1.0), (640, 1.0),
                 (100, 1.5), (200, 1.5), (400, 1.5)):
        run(env_pchip, N=N, grading=g, label=f"B 均匀/渐变 N={N} γ={g}")

    print("\n【C】时间步长上限（自适应误差控制）")
    for dtm in (1.0, 5.0, 10.0, 30.0, 60.0):
        run(env_pchip, dt_max=dtm, label=f"C dt_max={dtm:g} s")

    print("\n【D】时间格式")
    run(env_pchip, theta=0.5, label="D Crank–Nicolson θ=0.5")

    print("=" * 100)


if __name__ == "__main__":
    main()
