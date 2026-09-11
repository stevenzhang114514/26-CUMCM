"""
CODE-10 / CODE-11  问题1：温度场与水分场   [甲 · M0 · Day1晚间]

模型（附录2 常物性；M0 **不含蒸发潜热项**）
-------------------------------------------
    ρcp ∂T/∂t = (1/r) ∂_r( k r ∂T/∂r )        k = 0.36 W/(m·K)
    ∂C/∂t     = (1/r) ∂_r( D(C) r ∂C/∂r )     D = 7e-9·exp(−0.89/C)
    初值  T = 28 °C,  C = 2.55 kg/kg
    r = 0 : ∂T/∂r = ∂C/∂r = 0
    r = R0: −k ∂T/∂r = h(T_s − T∞(t)),   −D ∂C/∂r = h_m(C_s − C∞ᵉq(t))
            h = 25 W/(m²·K), h_m = 8e-7 m/s

★ 结构性洞察：单向耦合
-----------------------
    附录2 的 D(C) **不依赖 T**  →  水分方程完全独立于温度场
    温度方程的 M0 形式不含蒸发项 →  温度方程也完全独立
    故：**先解水分场，再解温度场**（两者可用同一步长共同推进，互不影响）

    这是问题1 相比问题2 的最大计算简化。问题2 中 D 依赖 T，
    必须联立求解，该简化不再成立。

数值
----
* 空间：半控制体 FVM（CODE-06），Δr = R0/N
* 时间：θ-方法，生产用 Backward Euler (θ=1)，L-稳定
* 水分：Picard 迭代（CODE-09），M-矩阵保证 C 正性
* 温度：常系数算子**只装配一次**，每步仅更新右端项（T∞ 随时间变）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import H_COEF, HM_COEF, PROPS_APP2, NUMERICS, to_kelvin
from ..numerics.fvm_cyl import Grid, Operator, assemble, implicit_step, surface_value, surface_flux
from ..numerics.nonlinear import picard_solve
from ..numerics.properties import D_app2
from ..numerics.time_stepper import AdaptiveStepper, IntegrateResult, explicit_stability_limit


# ==========================================================================
@dataclass
class Problem1Setup:
    grid: Grid
    env: object                     # EnvInterpolator
    theta: float = 1.0
    T_init: float = 28.0
    C_init: float = 2.55
    t_end: float = 1800.0
    use_latent_heat: bool = False   # M0 关闭；M1 扩展见 CODE-36


# ==========================================================================
def _tol_dict():
    n = NUMERICS
    return dict(atol_u=n.picard_atol_u, rtol_u=n.picard_rtol_u,
                atol_r=n.picard_atol_r, rtol_r=n.picard_rtol_r)


def solve_problem1(setup: Problem1Setup, t_output=None, p_order=None):
    """
    求解问题1。返回 (IntegrateResult, extra)。

    extra 含：表面温度/含水率/通量历史、算子诊断、稳定性自检结果。
    """
    g, env = setup.grid, setup.env
    num = NUMERICS
    theta = setup.theta
    if p_order is None:
        p_order = 1.0 if theta == 1.0 else 2.0

    # ---- 显式稳定性自检：不满足即拒绝该设置（而非等待发散） ----
    alpha = PROPS_APP2.alpha
    dt_exp = explicit_stability_limit(g.dr, alpha)
    assert dt_exp > num.dt_min, "显式稳定性上限低于步长下限，配置不合理"

    # ---- 温度：常系数算子只装配一次 ----
    alpha_cell = np.full(g.N, alpha)
    h_bc_T = H_COEF / (PROPS_APP2.rho * PROPS_APP2.cp)     # m/s
    op_T_template = assemble(alpha_cell, h_bc_T, 0.0, g)   # b 稍后按 T∞ 更新

    # ---- 状态 ----
    T0 = np.full(g.N, setup.T_init)
    C0 = np.full(g.N, setup.C_init)

    surf_T, surf_C, flux_w = [], [], []
    tol = _tol_dict()

    def step_fn(t, dt, state):
        t_new = t + dt

        # ============ 水分场（先解；非线性 Picard） ============
        C_inf_new = float(env.C_inf(t_new))

        def assemble_C(C):
            D_cell = np.atleast_1d(D_app2(C)).astype(float)
            return assemble(D_cell, HM_COEF, C_inf_new, g)

        C_new, iters, _res = picard_solve(
            assemble_C, state["C"], dt, theta, tol, num.picard_max_iter,
        )

        # ============ 温度场（后解；线性，单次求解） ============
        T_inf_new = float(env.T_inf(t_new))
        op_T = Operator(N=g.N, diag=op_T_template.diag.copy(),
                        lower=op_T_template.lower.copy(),
                        upper=op_T_template.upper.copy(),
                        b=op_T_template.b.copy(), beta=op_T_template.beta,
                        Bi_delta=op_T_template.Bi_delta,
                        Gamma_face=op_T_template.Gamma_face)
        # β·T∞ 更新右端项
        beta_T = op_T.beta
        op_T.b[:-1] = 0.0
        op_T.b[-1] = beta_T * T_inf_new
        T_new = implicit_step(op_T, state["T"], dt, theta=theta)

        return {"T": T_new, "C": C_new}, iters

    stepper = AdaptiveStepper(num)
    res = stepper.integrate(0.0, {"T": T0, "C": C0}, setup.t_end, t_output, step_fn,
                            p_order=p_order)

    # ---- 表面量（在输出时刻上重构，供表1/表2 的 r=2.0 cm 列使用） ----
    for k in range(len(res.times)):
        tk = float(res.times[k])
        Tk, Ck = res.T[k], res.C[k]
        Tinf = float(env.T_inf(tk))
        Cinf = float(env.C_inf(tk))
        surf_T.append(surface_value(Tk, alpha_cell, h_bc_T, Tinf, g))
        Dk = np.atleast_1d(D_app2(Ck)).astype(float)
        surf_C.append(surface_value(Ck, Dk, HM_COEF, Cinf, g))
        flux_w.append(surface_flux(Ck, Dk, HM_COEF, Cinf, g))

    extra = {
        "surface_T": np.asarray(surf_T),
        "surface_C": np.asarray(surf_C),
        "surface_flux_w": np.asarray(flux_w),
        "alpha": alpha,
        "explicit_dt_limit": dt_exp,
        "h_bc_T": h_bc_T,
        "p_order": p_order,
        "theta": theta,
    }
    return res, extra
