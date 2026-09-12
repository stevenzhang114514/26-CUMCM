"""
审阅用：乙的「问题4」格式**逐行移植**与单因素扰动    [甲 · 审阅工具 · 非交付]

为什么移植而不是只读代码
------------------------
静态读代码能发现"写法可疑"，但**不能定量说明偏差有多大**。
本脚本把乙的 `problem4_drying.m` **逐行译成 Python**（同一套网格权重、
同一个 thomas、同一个 Picard 判据、同一个 tdry 判据），
先验证能复现乙报告的 t_dry = 73.03 h，再**每次只改一个因子**，
测出各因子的贡献。

纪律
----
* 移植版必须**先复现**乙的数字，否则说明移植不忠实，一切结论作废。
* 扰动项一律注明"这是乙的选择"还是"这是本项目的选择"，不混。
* 只报告**差异归属**，不声称"乙错、我对"——两边都可能是近似。

用法：python scripts/review_yi_p4.py            # 复现 + 单因素扰动
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ATT = ROOT.parent / "A题题目和附件" / "附件"
def P(*a):
    print(*a, flush=True)


# ==========================================================================
# 环境
# ==========================================================================
def load_att1():
    df = pd.read_excel(ATT / "附件1.xlsx")
    cols = {c.strip(): c for c in df.columns}
    t = df[cols["时间"]].to_numpy(float)
    T = df[cols["温度"]].to_numpy(float)
    C = df[cols["水分浓度"]].to_numpy(float)
    return t, T, C


def load_att2():
    df = pd.read_excel(ATT / "附件2.xlsx")
    cols = {c.strip(): c for c in df.columns}
    return df[cols["时间"]].to_numpy(float), df[cols["半径"]].to_numpy(float)


class EnvYi:
    """乙的取法：t≤1800 s 用附件1，t>1800 s 硬切到常数 (50, 0.05)。"""

    def __init__(self, T_stage=50.0, C_stage=0.05, t_switch=1800.0):
        t, T, C = load_att1()
        self.tp = PchipInterpolator(t, T)
        self.cp = PchipInterpolator(t, C)
        self.t_end = float(t[-1])
        self.T_stage, self.C_stage, self.t_switch = T_stage, C_stage, t_switch

    def T(self, tt):
        return self.T_stage if tt > self.t_switch else float(self.tp(min(tt, self.t_end)))

    def C(self, tt):
        return self.C_stage if tt > self.t_switch else float(self.cp(min(tt, self.t_end)))


class EnvFull:
    """本项目取法：全程用附件1（PCHIP），14400 s 之后取平台均值。"""

    def __init__(self, t_pre=14400.0):
        t, T, C = load_att1()
        self.tp = PchipInterpolator(t, T)
        self.cp = PchipInterpolator(t, C)
        self.t_end = float(t[-1])
        m = t >= t_pre
        self.T_plat = float(T[m].mean())
        self.C_plat = float(C[m].mean())

    def T(self, tt):
        return self.T_plat if tt > self.t_end else float(self.tp(tt))

    def C(self, tt):
        return self.C_plat if tt > self.t_end else float(self.cp(tt))


# ==========================================================================
# 半径
# ==========================================================================
def make_R(variant: str):
    t2, R2 = load_att2()
    t2 = t2.copy()
    R2 = R2.copy()                       # cm
    t_max = float(t2[-1])
    p = PchipInterpolator(t2, R2)
    slope = (R2[-1] - R2[-2]) / (t2[-1] - t2[-2])   # cm/s

    if variant == "clamp":               # 乙 problem4_drying_const.m
        return lambda tt: float(p(min(tt, t_max))) / 100.0
    if variant == "linear":              # 乙 problem4_drying_linear.m
        return lambda tt: (float(p(min(tt, t_max))) + max(0.0, tt - t_max) * slope) / 100.0
    if variant == "raw":                 # 乙 problem4_drying.m（超出范围行为未定义）
        def f(tt):
            if tt > t_max:
                return float(p(t_max) + (tt - t_max) * slope) / 100.0
            return float(p(tt)) / 100.0
        return f
    raise ValueError(variant)


# ==========================================================================
# thomas（与乙逐行一致）
# ==========================================================================
def thomas(dl, dd, du, b):
    n = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = du[0] / dd[0]
    dp[0] = b[0] / dd[0]
    for i in range(1, n):
        m = dd[i] - dl[i] * cp[i - 1]
        if i < n - 1:
            cp[i] = du[i] / m
        dp[i] = (b[i] - dl[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ==========================================================================
# 乙的求解核心（逐行移植）
# ==========================================================================
def solve_yi(N=20, dt=10.0, tmax=302400.0, env=None, Rof=None,
             bc_level="old", stages=False, grading=1.0):
    """
    bc_level:
        "old" —— 乙的写法：Te/Ce 在步前用 t 取值（显式边界）
        "new" —— 本项目写法：用 t+dt 取值（与 Backward Euler 自洽）
    stages: False = 单层物性（附录4 全程）；True = 前 1800 s 用附录2 再切附录4
    """
    env = env or EnvYi()
    Rof = Rof or make_R("raw")

    # ------------------------------------------------------------------
    # 网格：grading=1 时**逐位退化为乙的均匀 ξ 网格**（已在 §0 复现验证）。
    #        grading>1 为向表面加密，即本项目 Day1 选定的做法。
    #
    # 节点 ξ_i (i=0..N-1)；控制体界面 face_i = (ξ_{i-1}+ξ_i)/2，face_0=0, face_N=1。
    # 体积权重 w_i = ∫face_i^{face_{i+1}} ξ dξ = (face_{i+1}² − face_i²)/2
    #   —— 均匀网格下给出 w_i = ξ_i·dξ，与乙的 `w = xi*dxi` 完全一致，
    #      两端半控制体的修正也与乙的 `w(1)` / `w(end)` 补丁完全一致。
    # 面间距 Δξ_i = ξ_{i+1} − ξ_i（渐变网格下逐面不同，不能再用常数 dxi）。
    # ------------------------------------------------------------------
    zeta = np.arange(N) / (N - 1)
    xi = 1.0 - (1.0 - zeta) ** grading
    face = np.concatenate(([0.0], 0.5 * (xi[:-1] + xi[1:]), [1.0]))
    w = 0.5 * (face[1:] ** 2 - face[:-1] ** 2)
    xf = face[1:-1]                 # N-1 个内部面位置
    dfx = np.diff(xi)               # N-1 个面间距
    h, hm = 25.0, 8e-7
    Ctarget = 0.15

    T = np.full(N, 28.0)
    C = np.full(N, 2.55)

    n_samp = int(tmax // 60) + 1
    tsample = np.arange(60, 60 * (n_samp + 1), 60.0)
    Crec = np.zeros((len(tsample), N))
    Trec = np.zeros((len(tsample), N))
    trec = np.zeros(len(tsample))
    isamp = 0

    t = 0.0
    tdry = np.nan
    nstep = 0
    C_min = np.inf               # ★ 丙/清单关心的"是否出现非物理负值"
    while t < tmax:
        Co = C.copy()
        To = T.copy()
        R = float(np.asarray(Rof(t)).ravel()[0])
        tb = t + dt if bc_level == "new" else t
        Te = env.T(tb)
        Ce = env.C(tb)

        for _ in range(10):
            TK = T + 273.15
            if stages and t < 1800.0:
                rho = np.full(N, 820.0)
                cp_ = np.full(N, 2600.0)
                k_ = np.full(N, 0.36)
                D_ = 7e-9 * np.exp(-0.89 / np.maximum(C, 1e-12))
                hv, hmv = 25.0, 8e-7
            else:
                rho = 760 + 90 * C
                cp_ = 1850 + 2150 * C / (C + 1)
                k_ = 0.12 + 0.20 * C / (C + 1)
                D_ = 4.2e-4 * np.exp(-0.30 / np.maximum(C, 1e-12)) * np.exp(-3850 / TK)
                hv, hmv = h, hm

            Df = 2 * D_[:-1] * D_[1:] / (D_[:-1] + D_[1:])
            kf = 2 * k_[:-1] * k_[1:] / (k_[:-1] + k_[1:])
            G = xf * Df / (R ** 2 * dfx)
            GT = xf * kf / (R ** 2 * dfx)

            dl = np.concatenate(([0.0], -G))
            du = np.concatenate((-G, [0.0]))
            dd = w / dt + np.concatenate((G, [0.0])) + np.concatenate(([0.0], G))
            b = w / dt * Co
            dd[-1] += hmv / R
            b[-1] += hmv / R * Ce
            Cn = thomas(dl, dd, du, b)

            cap = rho * cp_ * w / dt
            dl = np.concatenate(([0.0], -GT))
            du = np.concatenate((-GT, [0.0]))
            dd = cap + np.concatenate((GT, [0.0])) + np.concatenate(([0.0], GT))
            b = cap * To
            dd[-1] += hv / R
            b[-1] += hv / R * Te
            Tn = thomas(dl, dd, du, b)

            done = np.max(np.abs(Cn - C)) < 1e-8 and np.max(np.abs(Tn - T)) < 1e-8
            C, T = Cn, Tn
            if done:
                break

        C_min = min(C_min, float(C.min()))
        t += dt
        nstep += 1

        while isamp < len(tsample) and t >= tsample[isamp]:
            Crec[isamp] = C
            Trec[isamp] = T
            trec[isamp] = t
            isamp += 1

        if np.isnan(tdry) and C.max() < Ctarget:
            tdry = np.ceil(t / 60.0) * 60.0
        if not np.isnan(tdry) and t >= tdry:
            break

    n = int(np.count_nonzero(trec))
    return {"tdry": tdry, "trec": trec[:n], "Crec": Crec[:n], "Trec": Trec[:n],
            "nstep": nstep, "t_end": t, "C_min": float(C_min),
            "R_final": float(np.asarray(Rof(tdry)).ravel()[0])}


# ==========================================================================
def case(label, **kw):
    t0 = time.time()
    r = solve_yi(**kw)
    P(f"  {label:42s} t_dry={r['tdry']/3600:8.3f} h  "
      f"C_max(终)={r['Crec'][-1].max():.4f}  C_min全程={r['C_min']:+.3e}  "
      f"步={r['nstep']:6d}  {time.time()-t0:5.1f}s")
    return r


def main():
    P("=" * 104)
    P("  乙「问题4」格式 —— 逐行移植 + 单因素扰动")
    P("  乙报告：t_dry = 73.03 h，终态半径 1.198 cm，终态 C_max = 0.1500")
    P("=" * 104)

    P("\n【0】复现（乙的全部选择：N=20, dt=10 s, 附录4 全程, 环境 1800 s 硬切, 半径 raw 外推）")
    base = case("S0 乙原样", N=20, dt=10.0, env=EnvYi(), Rof=make_R("raw"),
                bc_level="old", stages=False)
    P(f"      → 复现偏差 {base['tdry']/3600 - 73.03:+.3f} h")

    P("\n【1】环境边界：1800 s 硬切 → 全程用附件1（14400 s 后取平台）")
    case("S1 环境改用附件1 全程", N=20, dt=10.0, env=EnvFull(), Rof=make_R("raw"),
         bc_level="old", stages=False)

    P("\n【2】边界取值时刻：t^n（乙）→ t^(n+1)（与 BE 自洽）")
    case("S2 边界取 t^(n+1)", N=20, dt=10.0, env=EnvYi(), Rof=make_R("raw"),
         bc_level="new", stages=False)

    P("\n【3】时间步长 dt=10 s → 更小")
    for dtv in (5.0, 2.0):
        case(f"S3 dt={dtv:g} s", N=20, dt=dtv, env=EnvYi(), Rof=make_R("raw"),
             bc_level="old", stages=False)

    P("\n【4】空间网格 N=20 → 更细（最需要检查的一项）")
    for Nv in (40, 80, 160):
        case(f"S4 均匀 N={Nv}", N=Nv, dt=10.0, env=EnvYi(), Rof=make_R("raw"),
             bc_level="old", stages=False)
    P("     —— 同样节点数，改为向表面加密（本项目 Day1 选定的做法）：")
    for Nv, gv in ((20, 1.5), (40, 2.0)):
        case(f"S4 渐变 N={Nv} γ={gv}", N=Nv, dt=10.0, env=EnvYi(),
             Rof=make_R("raw"), bc_level="old", stages=False, grading=gv)

    P("\n【5】半径外推方式（t* > 259200 s，落在附件2 覆盖之外）")
    for v in ("clamp", "linear"):
        case(f"S5 半径 {v}", N=20, dt=10.0, env=EnvYi(), Rof=make_R(v),
             bc_level="old", stages=False)

    P("\n【6】组合：环境修正 + 边界自洽 + 细网格")
    case("S6 全部修正 N=80 dt=5", N=80, dt=5.0, env=EnvFull(),
         Rof=make_R("raw"), bc_level="new", stages=False)
    P("=" * 104)


if __name__ == "__main__":
    main()
