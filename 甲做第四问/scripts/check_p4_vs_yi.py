"""
★ 关键验证：本实现必须复现乙的 73.033 h          [甲 · 验证脚本]

为什么必须先做这一步
--------------------
本实现与乙的 `problem4_drying.m` 是两套独立代码。若不能在**同一配置**下
复现乙的结果，那么"我改了网格所以结果不同"这句话就**没有依据** ——
分不清是网格改动的效果，还是我自己的 bug。

故本脚本是问题4 的**准入门槛**：不通过，就不出具 result4.xlsx。

配置（与乙逐项对齐）
--------------------
    网格   N=20，grading=1.0（**节点式**，与乙的 xi=(0:N-1)/(N-1) 相同）
    物性   附录4
    半径   附件2 / R_fit.mat 同源（本实现用附件2 原始 145 点 PCHIP）
    时间   dt=10 s 固定（乙的取值）
    环境   ①乙的取法（t≤1800 s 用附件1，其后硬切 50/0.05）
           ②本项目的取法（附件1 全程，14400 s 后取平台均值）
    边界   取 t^n（乙的写法）

判据
----
    ① 环境下 t_dry 应 ≈ 73.03 h
    ② 环境下 t_dry 应 ≈ 73.12 h（环境差异实测 +0.084 h）

用法：python scripts/check_p4_vs_yi.py
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NUMERICS                     # noqa: E402
from src.data_prep.env_interp import load_env       # noqa: E402
from src.models.problem4 import GridXiN, load_radius, solve_p4   # noqa: E402

P = lambda *a: print(*a, flush=True)
TARGET = 0.15


class EnvYi:
    """乙的环境取法：t≤1800 s 用附件1，t>1800 s 硬切到常数 (50, 0.05)。"""

    def __init__(self, T_stage=50.0, C_stage=0.05, t_switch=1800.0):
        t, T, C = load_env().t, load_env().T, load_env().C
        self.tp = PchipInterpolator(t, T)
        self.cp = PchipInterpolator(t, C)
        self.t_end = float(t[-1])
        self.Ts, self.Cs, self.sw = T_stage, C_stage, t_switch

    def T_inf(self, tt):
        return self.Ts if tt > self.sw else float(self.tp(min(tt, self.t_end)))

    def C_inf(self, tt):
        return self.Cs if tt > self.sw else float(self.cp(min(tt, self.t_end)))


class EnvFull:
    """本项目取法：附件1 全程 PCHIP，14400 s 后取平台均值。"""

    def __init__(self, t_pre=14400.0):
        d = load_env()
        self.tp = PchipInterpolator(d.t, d.T)
        self.cp = PchipInterpolator(d.t, d.C)
        self.t_end = float(d.t[-1])
        m = d.t >= t_pre
        self.Tp, self.Cp = float(d.T[m].mean()), float(d.C[m].mean())

    def T_inf(self, tt):
        return self.Tp if tt > self.t_end else float(self.tp(tt))

    def C_inf(self, tt):
        return self.Cp if tt > self.t_end else float(self.cp(tt))


def tdry_of(env, rad, N, gamma, dt_max, t_max=6 * 86400.0, dt_out=60.0):
    num = dataclasses.replace(NUMERICS, dt_max=dt_max, dt_init=dt_max,
                              dt_min=1e-3)
    g = GridXiN(N=N, grading=gamma)
    t_out = np.arange(dt_out, t_max + dt_out, dt_out)
    t0 = time.time()
    r = solve_p4(env, rad, g, t_max, t_out, num=num, stop_below=TARGET)
    el = time.time() - t0
    Cm = r["C_max"]
    b = Cm < TARGET
    if not np.any(b):
        return float("nan"), r, el
    k = int(np.argmax(b))
    return float(r["times"][k]), r, el


def main():
    P("=" * 88)
    P("  ★ 问题4 准入门槛：本实现能否复现乙的 73.033 h")
    P("=" * 88)
    rad = load_radius()
    P(f"  {rad.summary()}")
    g = GridXiN(N=20, grading=1.0)
    P(f"  {g.summary()}")
    P(f"  节点 ξ[:3]={g.xi[:3].round(6)}  ξ[-1]={g.xi[-1]:.6f}")
    P(f"  面   f[:3]={g.f[:3].round(6)}   f[-1]={g.f[-1]:.6f}")
    P(f"  体积 w[:3]={g.w[:3].round(8)}  w[-1]={g.w[-1]:.8f}")
    P("  （乙：xf=(j-0.5)*dxi, dxi=1/19；w(1)=(dxi/2)^2/2, "
      "w(end)=(1-(1-dxi/2)^2)/2）")
    P("")

    ok = True
    for tag, env, expect in (("①乙的环境（1800 s 硬切）", EnvYi(), 73.033),
                             ("②本项目环境（附件1 全程）", EnvFull(), 73.117)):
        td, r, el = tdry_of(env, rad, N=20, gamma=1.0, dt_max=10.0)
        # t_out 是 60 s 网格，取首次跌破后再按斜率细化（这里只做量级核对）
        P(f"  {tag}")
        P(f"     t_dry = {td:.0f} s = {td/3600:.3f} h    期望 ≈ {expect:.3f} h"
          f"    偏差 {td/3600-expect:+.3f} h    （{el:.0f}s）")
        # 🔴 NaN 必须单独判：`abs(nan-x) > tol` 恒为 False，
        #    曾因此让一个"根本没达标"的结果被误判为 ✅ 通过。
        if not np.isfinite(td):
            P(f"     ❌ 未达标（6 天内 C_max 未跌破 0.15）—— 判定为未通过")
            ok = False
        elif abs(td / 3600 - expect) > 0.6:
            P(f"     ❌ 超出 ±0.6 h 容差")
            ok = False
        else:
            P(f"     ✅ 在容差内")

    P("")
    P("=" * 88)
    P(f"  准入门槛：{'✅ 通过 —— 本实现与乙的代码在同一配置下一致' if ok else '❌ 未通过'}")
    P("=" * 88)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
