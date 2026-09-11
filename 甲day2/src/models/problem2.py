"""
CODE-12  问题2：变参数热湿双向耦合（全流程）   [甲 · M0/M1 · Day2上午]

题面要求（[Q-PDF] 问题2）
-------------------------
"烘干过程一般持续 2—3 天，预热平衡与恒温干燥阶段的参数有所不同。
  请建立整个烘干过程药材温度和水分浓度变化规律的数学模型
  （**为简化问题，相关经验公式统一采用附录 3 中的公式**）"

🔴 关键执行点（清单 §CODE-12）
-------------------------------
1. **从 t=0 统一采用附录3**，**不允许**先跑问题1 再在某时刻切换参数。
   题面已明确"统一采用附录3"，故不存在"阶段切换"，
   也就不需要"切换处保持状态连续"的处理 —— 这一点必须写进论文，
   否则会被误认为我们漏掉了阶段切换。
2. **变系数不得移出微分算子**：ρ(C)c_p(C)∂T/∂t 走守恒形式，
   经 fvm_cyl.assemble 的 `cap_cell` 参数实现（面导热系数用 k 的调和平均，
   热容用单元值，二者不可合并 —— 见该函数文档）。
3. T 与 C **联立隐式求解**（D 显含 T，问题1 的"先水后温"不再成立）。

控制方程（M0）
--------------
    ρ(C)c_p(C) ∂T/∂t = (1/r) ∂_r( k(C) r ∂T/∂r )
    ∂C/∂t            = (1/r) ∂_r( D(C,T) r ∂C/∂r )

    ρ = 650 + 128C,  c_p = 1450 + 2736C/(1+C),  k = 0.21 + 0.38C/(1+C)
    D = 2.4e-3 · exp(−0.45/C) · exp(−3850/T_K)          [Q-PDF 附录3]

初值 / 边界
-----------
    T(r,0) = 28 °C,  C(r,0) = 2.55 kg/kg
    r = 0 :  ∂T/∂r = ∂C/∂r = 0
    r = R0:  M0  −k ∂T/∂r = h(T_s − T∞),  −D ∂C/∂r = h_m(C_s − C∞ᵉᑫ)
             M1  −k ∂T/∂r = h(T_s − T∞) + λ J_w      （见 models/boundary.py）

长时边界：附件1 只到 14400 s，恒温段按框架 §5.3 方案 A 取常数平台值
（见 data_prep/env_extrap.py）。**该假设本身写入论文**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import (C_INIT, H_COEF, HM_COEF, NUMERICS, R_OUT_CM, T_INIT,
                      to_kelvin)
from ..numerics.coupled import block_picard_solve
from ..numerics.fvm_cyl import (Grid, assemble, surface_flux, surface_value)
from ..numerics.properties import D_app3, cp_app3, k_app3, rho_app3
from ..numerics.time_stepper import AdaptiveStepper, explicit_stability_limit
from .boundary import (BoundaryConfig, heat_ambient, moisture_surface_flux,
                       water_mass_flux)
from ..io.resample import center_value


# ==========================================================================
@dataclass
class Problem2Setup:
    grid: Grid
    env: object                       # PlateauEnv
    bc: BoundaryConfig = field(default_factory=BoundaryConfig)
    theta: float = 1.0
    T_init: float = T_INIT
    C_init: float = C_INIT
    t_end: float = 10800.0
    # ---- 情景扰动开关（CODE-25；默认全为 1，即题设基线）----
    x_D: float = 1.0                  # D 的倍率
    x_hm: float = 1.0                 # h_m 的倍率
    x_h: float = 1.0                  # h 的倍率
    label: str = "M0"
    # ---- 起始层（见 time_stepper.integrate 的说明）----
    # M1 的潜热项使 t=0 处边界条件阶跃 → 真解按 √t 演化 →
    # 步长加倍法在跨 t=0 的首步失效（局部误差 O(Δt^1/2) 而非 O(Δt²)）。
    # 故起始 warmup_steps 步强制等步长、不做误差控制。
    # M0 无此阶跃，保持 0。
    warmup_steps: int = 0
    warmup_dt: float = 1.0e-3

    def eff_h_m(self) -> float:
        return self.bc.h_m * self.x_hm

    def eff_h(self) -> float:
        return self.bc.h * self.x_h


# ==========================================================================
def _tol_dict():
    n = NUMERICS
    return (dict(atol_u=n.picard_atol_u, rtol_u=n.picard_rtol_u,
                 atol_r=n.picard_atol_r, rtol_r=n.picard_rtol_r),
            dict(atol_u=n.picard_atol_u, rtol_u=n.picard_rtol_u,
                 atol_r=n.picard_atol_r, rtol_r=n.picard_rtol_r))


# ==========================================================================
def build_step_fn(setup: Problem2Setup, num=None, stats=None):
    """
    构造一个时间步函数 ``step_fn(t, dt, state) -> (state_new, n_iters)``。

    独立成工厂函数的理由：**CODE-19 的定步长细化不能走自适应器**。
    把 dt_min 与 dt_max 都钉在 dt 会让自适应器在"拒绝→缩步"时
    永远缩不下去（dt 已被钳在 dt_min），形成无限拒绝循环。
    定步长必须绕开自适应控制，直接循环调用本函数。
    事件二分（CODE-15）与情景扫描（CODE-25）也复用同一实现。

    ⚠️ 闭包内的 ``T_inf_new`` / ``C_inf_new`` 取自 **t+dt**（隐式右端），
    这正是 Backward Euler 的要求，不能取 t。
    """
    g = setup.grid
    env = setup.env
    bc = setup.bc
    num = num if num is not None else NUMERICS
    theta = setup.theta
    h_m = setup.eff_h_m()
    h_bc = setup.eff_h()
    tol_T, tol_C = _tol_dict()
    st = stats if stats is not None else {"iters": [], "last_iters": 0}

    def step_fn(t, dt, state):
        t_new = t + dt
        T_inf_new = float(env.T_inf(t_new))
        C_inf_new = float(env.C_inf(t_new))

        def assemble_C(C_g, T_g):
            D_cell = setup.x_D * np.atleast_1d(
                np.asarray(D_app3(C_g, T_g), dtype=float))
            return assemble(D_cell, h_m, C_inf_new, g)

        def assemble_T(T_g, C_g):
            k_cell = np.atleast_1d(np.asarray(k_app3(C_g), dtype=float))
            cap_cell = np.atleast_1d(np.asarray(
                rho_app3(C_g) * cp_app3(C_g), dtype=float))
            # M1：用**同一个**离散水通量折算等效环境温度（避免重复计数）
            if bc.latent:
                D_g = setup.x_D * np.atleast_1d(
                    np.asarray(D_app3(C_g, T_g), dtype=float))
                J_w = water_mass_flux(C_g, D_g, C_inf_new, g, bc).J_w
            else:
                J_w = 0.0
            T_amb, _ = heat_ambient(T_inf_new, J_w, bc)
            return assemble(k_cell, h_bc, T_amb, g, cap_cell=cap_cell)

        T_new, C_new, it = block_picard_solve(
            state["T"], state["C"], dt, theta,
            assemble_T, assemble_C, tol_T, tol_C,
            max_iter=num.picard_max_iter)

        if not np.all(np.isfinite(T_new)):
            raise RuntimeError(f"t={t_new:.6g} 温度出现非有限值")

        st["last_iters"] = it.iters
        st["iters"].append(it.iters)
        return {"T": T_new, "C": C_new}, it.iters

    return step_fn, st


def solve_problem2(setup: Problem2Setup, t_output=None, p_order=None,
                   t0: float = 0.0, state0=None, collect_diag=True, num=None,
                   stop_fn=None):
    """
    求解问题2。返回 (IntegrateResult, extra)。

    参数
    ----
    t0, state0 : 起始时刻与起始状态 (T0, C0)。默认 (0, 初值)。
                 **CODE-15 的事件二分定位**靠它从已存状态重启，
                 避免为了试一个时刻而从 0 重算整段。
    collect_diag : 是否在每个输出时刻重构表面量。
                 二分试算时置 False 可省掉大部分开销。
    num        : 数值参数覆盖（默认全局 NUMERICS）。
    stop_fn    : callable(t, state) -> bool，每个接受步后询问，True 即停。
                 **CODE-15 的阈值粗扫与 CODE-25 的情景扫描靠它早停** ——
                 不早停就要跑满整个视界，浪费 2—3 倍甚至 20 倍机时。

    extra 含每个输出时刻的诊断量（表面通量、等效环境温度、Bi_m、Lu 等），
    供 G12、FIG-16、CODE-15、CODE-25 使用。
    """
    g = setup.grid
    env = setup.env
    bc = setup.bc
    num = num if num is not None else NUMERICS
    theta = setup.theta
    if p_order is None:
        p_order = 1.0 if theta == 1.0 else 2.0

    h_m = setup.eff_h_m()
    h_bc = setup.eff_h()

    # ---- 显式稳定性自检 ----
    # 用途是"论证为什么必须用隐式"，口径须与 Day1 一致：
    # 取**名义间距** Δr = R0/N（而非渐变网格的最小面距 7 µm），
    # 并把 α 取遍整个含水率区间的**最大值**（最不利情形）。
    # ⚠️ 渐变网格的最小面距会让 Δr²/(4α) 降到 ~1e-4 s，但这对
    #    **L-稳定的 Backward Euler 根本不是约束**，若拿它做断言会得出
    #    误导性结论。最小面距仅作信息量记录。
    C_probe = np.linspace(0.05, setup.C_init, 256)
    alpha_max = float(np.max(k_app3(C_probe) /
                             (rho_app3(C_probe) * cp_app3(C_probe))))
    dt_exp = explicit_stability_limit(float(g.R0 / g.N), alpha_max)
    dt_exp_graded = explicit_stability_limit(float(np.min(np.diff(g.rf))),
                                             alpha_max)
    # 🔴 **真正的稳定性判据只有这一条**：θ ≥ 0.5 时格式无条件稳定。
    # 显式上限 dt_exp 只作**信息量**记录（用于论证"为什么必须用隐式"），
    # **不是**对 dt 的约束 —— Backward Euler 是 L-稳定的，dt 可以远超 dt_exp。
    # （早先把 "dt_exp > dt_min" 写成断言，结果 CODE-19 强制大步长时被误伤。）
    assert theta >= 0.5, "θ<0.5 时格式非无条件稳定，本问题不采用"
    dt_min_below_explicit = bool(num.dt_min <= dt_exp)

    if state0 is None:
        T0 = np.full(g.N, setup.T_init)
        C0 = np.full(g.N, setup.C_init)
    else:
        T0 = np.array(state0[0], dtype=float)
        C0 = np.array(state0[1], dtype=float)
        assert T0.shape == (g.N,) and C0.shape == (g.N,), "state0 形状与网格不符"

    # 每个输出时刻的诊断记录
    diag = {k: [] for k in
            ("t", "T_center", "C_center", "T_surf", "C_surf",
             "J_C", "J_w", "q_conv", "q_lat", "T_inf_eff", "dT_evap",
             "C_max", "C_max_cell", "r_argmax", "Bi_m_surf", "Lu_surf",
             "alpha_surf", "D_surf", "iters")}

    stats = {"iters": [], "last_iters": 0}
    step_fn, _ = build_step_fn(setup, num=num, stats=stats)

    stepper = AdaptiveStepper(num)
    res = stepper.integrate(t0, {"T": T0, "C": C0}, setup.t_end, t_output,
                            step_fn, p_order=p_order,
                            warmup_steps=setup.warmup_steps,
                            warmup_dt=setup.warmup_dt,
                            stop_fn=stop_fn)

    if not collect_diag:
        extra = {"h_m": h_m, "h_bc": h_bc, "bc": bc, "setup": setup,
                 "alpha_max": alpha_max, "explicit_dt_limit": dt_exp,
                 "explicit_dt_limit_graded": dt_exp_graded,
                 "dt_min_below_explicit": dt_min_below_explicit,
                 "p_order": p_order, "theta": theta,
                 "n_accepted": res.n_accepted, "n_rejected": res.n_rejected,
                 "n_warmup": res.n_warmup}
        return res, extra

    # ------------------------------------------------------------------
    # 在每个输出时刻重构表面量并记录诊断
    # ------------------------------------------------------------------
    for k in range(len(res.times)):
        tk = float(res.times[k])
        Tk, Ck = res.T[k], res.C[k]
        T_inf = float(env.T_inf(tk))
        C_inf = float(env.C_inf(tk))

        Dk = setup.x_D * np.atleast_1d(np.asarray(D_app3(Ck, Tk), dtype=float))
        kk = np.atleast_1d(np.asarray(k_app3(Ck), dtype=float))
        capk = np.atleast_1d(np.asarray(rho_app3(Ck) * cp_app3(Ck), dtype=float))

        # ① 水分：表面通量与水通量（与求解所用的离散量完全一致）
        mf = moisture_surface_flux(Ck, Dk, C_inf, g, bc)
        J_C, C_s = mf.J_C, mf.C_s
        J_w = bc.rho_d * J_C if bc.latent else 0.0

        # ② 热：等效环境温度 → 表面温度
        T_amb, dT_evap = heat_ambient(T_inf, J_w, bc)
        T_s = surface_value(Tk, kk, h_bc, T_amb, g)
        q_conv = surface_flux(Tk, kk, h_bc, T_inf, g)
        q_lat = bc.lambda_vap * J_w

        # ③ 全域最大值（判据用，**不得用平均值代替**）
        C_max_cell = float(np.max(Ck))
        C_center_pt = float(center_value(Ck, g))
        C_max = max(C_max_cell, C_center_pt)
        r_arg = float(g.rc[int(np.argmax(Ck))])

        # ④ 无量纲与物性诊断
        D_N = float(Dk[-1])
        alpha_N = float(kk[-1] / capk[-1])
        T_K = float(to_kelvin(Tk[-1]))

        diag["t"].append(tk)
        diag["T_center"].append(float(center_value(Tk, g)))
        diag["C_center"].append(C_center_pt)
        diag["T_surf"].append(T_s)
        diag["C_surf"].append(C_s)
        diag["J_C"].append(J_C)
        diag["J_w"].append(J_w)
        diag["q_conv"].append(q_conv)
        diag["q_lat"].append(q_lat)
        diag["T_inf_eff"].append(T_amb)
        diag["dT_evap"].append(dT_evap)
        diag["C_max"].append(C_max)
        diag["C_max_cell"].append(C_max_cell)
        diag["r_argmax"].append(r_arg)
        diag["Bi_m_surf"].append(h_m * g.R0 / max(D_N, 1e-300))
        diag["Lu_surf"].append(D_N / max(alpha_N, 1e-300))
        diag["alpha_surf"].append(alpha_N)
        diag["D_surf"].append(D_N)
        diag["iters"].append(stats["last_iters"])

    extra = {k: np.asarray(v, dtype=float) for k, v in diag.items()}
    extra.update({
        "h_m": h_m, "h_bc": h_bc, "bc": bc, "setup": setup,
        "alpha_max": alpha_max, "explicit_dt_limit": dt_exp,
        "explicit_dt_limit_graded": dt_exp_graded,
        "dt_min_below_explicit": dt_min_below_explicit,
        "p_order": p_order, "theta": theta,
        "n_accepted": res.n_accepted, "n_rejected": res.n_rejected,
        "mean_iters": float(np.mean(stats["iters"])) if stats["iters"] else 0.0,
        "max_iters": int(np.max(stats["iters"])) if stats["iters"] else 0,
    })
    return res, extra


# ==========================================================================
# 时间尺度与量级（供论文与 FIG-16）
# ==========================================================================
def timescales(setup: Problem2Setup, T_ref=(28.0, 50.0),
               C_ref=(0.15, 2.55)) -> dict:
    """
    在状态区间角点上给出传热/传质特征时间。

    τ_heat = R0²/α,  τ_mass = R0²/D
    """
    out = {}
    R0 = setup.grid.R0
    for Tc in T_ref:
        for Cc in C_ref:
            Cv = np.array([Cc]); Tv = np.array([Tc])
            k_ = float(np.asarray(k_app3(Cv))[0])
            cap = float(np.asarray(rho_app3(Cv) * cp_app3(Cv))[0])
            D_ = float(np.asarray(D_app3(Cv, Tv))[0])
            out[f"T={Tc:.0f}C_C={Cc}"] = {
                "k": k_, "rho_cp": cap, "alpha": k_ / cap, "D": D_,
                "tau_heat_s": R0 ** 2 / (k_ / cap),
                "tau_mass_s": R0 ** 2 / max(D_, 1e-300),
                "Bi_m": setup.eff_h_m() * R0 / max(D_, 1e-300),
                "Lu": D_ / (k_ / cap),
            }
    return out
