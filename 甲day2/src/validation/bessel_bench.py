"""
CODE-18（甲自用部分）  常系数解析基准      [甲 · M0 · Day1下午]

目的
----
验证 FVM 空间算子 + Robin 边界重构 + 时间格式阶数。这是整个数值框架的地基。

解析解（无限长圆柱，常物性，**恒定环境温度**，Robin 边界）
----------------------------------------------------------
    θ(r,t) = T(r,t) − T∞ = Σ_n A_n J0(μ_n r/R0) exp(−α μ_n² t / R0²)

    特征值 μ_n 为  μ J1(μ) = Bi·J0(μ)  的正根，Bi = h R0 / k
    系数        A_n = 2(T0 − T∞)·J1(μ_n) / [ μ_n ( J0(μ_n)² + J1(μ_n)² ) ]

🔴 **三个条件必须同时满足，缺一不可**
    ① 固定热学物性（ρ、cp、k 常数）
    ② 关闭蒸发热效应（M0 本身不含潜热项，故天然满足）
    ③ 环境温度**恒定**（解析解要求；不能用附件1 的时变曲线）

⚠️ 对比位置：**仅单元中心**。边界节点是离散重构量，与解析解在 r=R0 的点值
   是两个不同的对象，混在一起比会人为制造误差。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1

from ..config import H_COEF, PROPS_APP2
from ..numerics.fvm_cyl import Grid, assemble, implicit_step


# ==========================================================================
# 特征值
# ==========================================================================
def robin_eigenvalues(Bi, n_terms=30, mu_max=None):
    """
    求 μ J1(μ) = Bi·J0(μ) 的前 n_terms 个正根。

    做法：在 (0, mu_max] 上细扫 f(μ) = μJ1(μ) − Bi·J0(μ) 找变号区间，再 brentq。
    这比"按 J1 零点分区间"更稳健，且不依赖 J0/J1 零点交错的具体模式。
    """
    f = lambda mu: mu * j1(mu) - Bi * j0(mu)
    if mu_max is None:
        mu_max = 40.0 * n_terms ** 0.5 + 60.0

    grid = np.linspace(1e-9, mu_max, 200000)
    vals = f(grid)
    sign = np.sign(vals)
    idx = np.where(np.diff(sign) != 0)[0]
    roots = []
    for i in idx:
        try:
            r = brentq(f, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-15)
        except ValueError:
            continue
        if r > 1e-8:
            roots.append(r)
        if len(roots) >= n_terms:
            break
    roots = np.array(roots)
    if len(roots) < n_terms:
        raise RuntimeError(f"只找到 {len(roots)} 个特征值，少于要求的 {n_terms} 个")
    return roots[:n_terms]


def bessel_coeffs(mu, Bi, T0, T_inf):
    """A_n = 2(T0−T∞)·J1(μ_n) / [ μ_n (J0(μ_n)² + J1(μ_n)²) ]"""
    return 2.0 * (T0 - T_inf) * j1(mu) / (mu * (j0(mu) ** 2 + j1(mu) ** 2))


# ==========================================================================
# 解析解求值
# ==========================================================================
@dataclass
class CylinderAnalytic:
    R0: float
    alpha: float
    Bi: float
    T0: float
    T_inf: float
    n_terms: int = 30

    def __post_init__(self):
        self.mu = robin_eigenvalues(self.Bi, self.n_terms)
        self.A = bessel_coeffs(self.mu, self.Bi, self.T0, self.T_inf)

    def T(self, r, t):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        t = np.asarray(t, dtype=float)
        r, t = np.broadcast_arrays(r, t)
        out = np.zeros_like(r, dtype=float)
        for mu_n, A_n in zip(self.mu, self.A):
            decay = np.exp(-self.alpha * mu_n ** 2 * t / self.R0 ** 2)
            out += A_n * j0(mu_n * r / self.R0) * decay
        return self.T_inf + out

    def truncation_check(self, r, t, n_terms_alt=None):
        """项数自检：30 项与 60 项应一致到 1e-12。"""
        alt = CylinderAnalytic(self.R0, self.alpha, self.Bi, self.T0, self.T_inf,
                               n_terms=n_terms_alt or 2 * self.n_terms)
        return float(np.max(np.abs(self.T(r, t) - alt.T(r, t))))


# ==========================================================================
# 数值解 vs 解析解
# ==========================================================================
def solve_heat_constant(grid: Grid, T_inf_const, T0, t_out, theta=1.0,
                        n_sub=400):
    """
    用生产内核（同样的 assemble / implicit_step）解常系数导热问题。
    环境温度恒定，蒸发关闭。返回 (t_out, T_cell[N, n_t])。
    """
    alpha = PROPS_APP2.alpha
    h_bc = H_COEF / (PROPS_APP2.rho * PROPS_APP2.cp)
    alpha_cell = np.full(grid.N, alpha)
    op = assemble(alpha_cell, h_bc, T_inf_const, grid)

    T = np.full(grid.N, T0)
    t_hist, T_hist = [], []
    t = 0.0
    t_out = np.asarray(t_out, dtype=float)
    nxt = 0
    dt_max = 5.0 if theta == 1.0 else 2.0

    k = 0
    guard = 0
    while t < t_out[-1] - 1e-12 and guard < 2_000_000:
        guard += 1
        dt = dt_max
        if nxt < len(t_out) and t + dt > t_out[nxt]:
            dt = t_out[nxt] - t
        if dt <= 0:
            break
        # 子步（固定小步长，确保时间误差可忽略）
        n = max(1, int(np.ceil(dt / (dt_max / n_sub))) if n_sub else 1)
        h = dt / n
        for _ in range(n):
            T = implicit_step(op, T, h, theta=theta)
        t += dt
        if nxt < len(t_out) and abs(t - t_out[nxt]) < 1e-9:
            t_hist.append(t)
            T_hist.append(T.copy())
            nxt += 1
    return np.asarray(t_hist), np.asarray(T_hist).T


def run_benchmark(T_inf_const=40.0, T0=28.0, n_terms=30, verbose=True):
    """基准测试：与解析解逐点比对（仅单元中心）。"""
    R0 = 0.02
    alpha = PROPS_APP2.alpha
    Bi = H_COEF * R0 / PROPS_APP2.k
    ana = CylinderAnalytic(R0=R0, alpha=alpha, Bi=Bi, T0=T0, T_inf=T_inf_const,
                           n_terms=n_terms)
    t_out = np.array([60.0, 300.0, 900.0, 1800.0])
    grid = Grid(N=40, R0=R0)
    t_num, T_num = solve_heat_constant(grid, T_inf_const, T0, t_out, theta=1.0)

    tc = grid.rc
    T_ana = np.array([ana.T(tc, tt) for tt in t_out]).T   # (N, n_t)
    err = np.abs(T_num - T_ana)

    out = {
        "Bi": Bi, "mu_first5": ana.mu[:5],
        "trunc_err": ana.truncation_check(tc, t_out[-1]),
        "t_out": t_out,
        "max_err_per_t": err.max(axis=0),
        "T_num": T_num, "T_ana": T_ana, "rc": tc, "ana": ana, "grid": grid,
    }
    if verbose:
        print(f"[CODE-18] 常系数基准  Bi={Bi:.6f}  前5个特征值={np.round(ana.mu[:5],4)}")
        print(f"          项数自检(30 vs 60) 最大差 = {out['trunc_err']:.3e} °C")
        for i, tt in enumerate(t_out):
            print(f"          t={tt:6.0f}s  最大绝对误差 = {err[:, i].max():.3e} °C")
    return out


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    run_benchmark()
