"""
与网络参考答案的差异核查                [甲 · 验证脚本]

参考值（用户提供）
------------------
    问题2  3h 中心 49.8495 °C，表面 49.9664 °C
    问题3  基准模型临界根 57.46681156 h
    问题4  收缩模型严格达标时间 51.0823 h，终点半径 1.2 cm

本项目的值
----------
    问题2  49.8494 / 49.9666
    问题3  t* = 206 959.9521 s = 57.4889 h
    问题4  t_dry = 184 200 s = 51.1667 h（**60 s 输出网格上的**，未精确求根）

本脚本做的事
------------
① 问题4 用**二分精确求根**（像问题3 那样），消掉 60 s 量化，
   再看与 51.0823 h 的差还剩多少
② 环境平台的取值敏感性（50.0017/0.049998 vs 整数 50/0.05）
③ 问题3 在更细网格上的 t*，检验 79.5 s 是否落在空间离散误差内

用法：python scripts/compare_reference.py
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NUMERICS                      # noqa: E402
from src.data_prep.env_interp import load_env        # noqa: E402
from src.models.problem4 import GridXiN, load_radius, solve_p4   # noqa: E402

P = lambda *a: print(*a, flush=True)
TARGET = 0.15
REF_P3_H = 57.46681156
REF_P4_H = 51.0823
MY_P3_H = 57.4889


class Env:
    """附件1 全程 PCHIP；t>t_end 取平台。plateau=None 时用实测均值。"""

    def __init__(self, t_pre=14400.0, plateau=None):
        from scipy.interpolate import PchipInterpolator
        d = load_env()
        self.tp = PchipInterpolator(d.t, d.T)
        self.cp = PchipInterpolator(d.t, d.C)
        self.t_end = float(d.t[-1])
        if plateau is None:
            m = d.t >= t_pre
            self.Tp, self.Cp = float(d.T[m].mean()), float(d.C[m].mean())
            self.tag = f"实测平台 {self.Tp:.4f}/{self.Cp:.6f}"
        else:
            self.Tp, self.Cp = plateau
            self.tag = f"整数平台 {self.Tp}/{self.Cp}"

    def T_inf(self, t):
        return self.Tp if t > self.t_end else float(self.tp(t))

    def C_inf(self, t):
        return self.Cp if t > self.t_end else float(self.cp(t))


def c_max_at_p4(env, rad, grid, tq, num):
    """问题4：积分到 tq，返回 C_max(tq)。"""
    r = solve_p4(env, rad, grid, float(tq), np.array([float(tq)]), num=num)
    return float(r["C"][-1].max()) if len(r["C"]) else float("nan")


def bisect_p4(env, rad, grid, t_lo, t_hi, num, tol=1e-3):
    """问题4 的达标时刻二分（从固定锚点重积）。"""
    f_lo = c_max_at_p4(env, rad, grid, t_lo, num) - TARGET   # >0
    f_hi = c_max_at_p4(env, rad, grid, t_hi, num) - TARGET   # <0
    if not (f_lo > 0 > f_hi):
        return float("nan"), f_lo, f_hi
    n = 0
    while t_hi - t_lo > tol and n < 60:
        tm = 0.5 * (t_lo + t_hi)
        fm = c_max_at_p4(env, rad, grid, tm, num) - TARGET
        if fm > 0:
            t_lo = tm
        else:
            t_hi = tm
        n += 1
    return 0.5 * (t_lo + t_hi), f_lo, f_hi


def main():
    rad = load_radius()
    env_meas = Env()
    env_int = Env(plateau=(50.0, 0.05))
    P("=" * 92)
    P("  与参考答案的差异核查")
    P("=" * 92)
    P(f"  参考：问题3 {REF_P3_H} h   问题4 {REF_P4_H} h")
    P(f"  本项目：问题3 {MY_P3_H} h   问题4 51.1667 h（60 s 网格）")
    P("")

    # ---------- ① 问题4 精确求根 ----------
    P("① 问题4：把 60 s 网格量化消掉（二分到 1e-3 s，锚点固定在 180000 s）")
    grid = GridXiN(N=200, grading=1.5)
    num = dataclasses.replace(NUMERICS, dt_max=10.0, dt_init=10.0)
    t0 = time.time()
    t_star, f_lo, f_hi = bisect_p4(env_meas, rad, grid, 180000.0, 190000.0, num)
    P(f"   精确 t_dry = {t_star:.3f} s = {t_star/3600:.6f} h")
    P(f"   参考答案   = {REF_P4_H} h = {REF_P4_H*3600:.1f} s")
    P(f"   ⇒ 差 {t_star - REF_P4_H*3600:+.1f} s = {(t_star/3600-REF_P4_H):+.6f} h "
      f"（{(t_star/3600/REF_P4_H-1)*100:+.4f}%）   （{time.time()-t0:.0f}s）")
    P(f"   （原 60 s 网格值 184200 s 相对精确值偏 {184200-t_star:+.0f} s）")
    P("")

    # ---------- ② 环境平台敏感性 ----------
    P("② 环境平台取值敏感性（问题4，网格固定 N=200/γ=1.5）")
    for env in (env_meas, env_int):
        tq = 184200.0
        cm = c_max_at_p4(env, rad, grid, tq, num)
        P(f"   {env.tag:34s}  C_max(184200 s) = {cm:.8f}")
    P("")

    # ---------- ③ 问题3 参考值对比 ----------
    P("③ 问题3：参考 57.46681156 h，本项目 57.4889 h，差 +79.5 s")
    P("   由 Day3 的离散误差表：时间离散 ≈1.9 s，空间离散 ≈17.4 s（合计 ≈18 s）")
    P("   ⇒ 79.5 s 约为该估计的 4.4 倍，**不能全归给离散误差**，需查环境口径")
    P("")
    P("=" * 92)


if __name__ == "__main__":
    main()
