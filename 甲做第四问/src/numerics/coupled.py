"""
CODE-12a  双向耦合的块 Picard 迭代        [甲 · M0 · Day2上午]

为什么问题2 不能沿用问题1 的"先水后温"
----------------------------------------
问题1 的结构性简化来自两条**同时**成立的事实：
    ① 附录2 的 D(C) 不含 T   → 水分场独立于温度场
    ② M0 的热方程不含蒸发项  → 温度场也独立
问题2 从 t=0 起统一采用附录3，其中

        D = 2.4e-3 · exp(−0.45/C) · exp(−3850/T_K)

**显含 T**，于是①失效：水分场必须知道温度才能推进。
（②在 M0 下仍成立，但 M1 启用潜热后也失效 —— 见 CODE-13。）

因此 T 与 C 必须**联立隐式求解**。

方法：块 Gauss–Seidel Picard
-----------------------------
外层迭代 k：

    1. 用 (C^k, T^k) 求 D            → 装配水分算子 → 解出 C^{k+1}
    2. 用 **最新的 C^{k+1}** 求 k, ρc_p
       （M1 时再用 C^{k+1} 求 J_w → 等效环境温度）
                                     → 装配温度算子 → 解出 T^{k+1}
    3. 双重判据同时检验两个场；不满足则继续

用 Gauss–Seidel（而非 Jacobi，即第 2 步用旧的 C^k）的理由：
ρc_p、k 对 C 的依赖是**单向**的（C 影响 T，T 不影响 ρc_p、k），
用最新 C 可显著加快收敛，且不破坏任何子问题的 M-矩阵性质。

为什么仍用 Picard 而不是 Newton
-------------------------------
每次子求解的迭代矩阵仍是三对角 M-矩阵 → 离散极值原理成立
→ **C 与 T 的正性/有界性由构造保证**，无需裁剪，
也不会像 Newton 那样过冲进入 exp(−0.45/C) 的悬崖区（C→0 时导数发散）。

收敛判据（**尺度归一化的双重判据，两个场、两个判据都必须满足**）
----------------------------------------------------------------
    更新量：U = max( max|ΔT|/(atol_uT + rtol_uT|T|),  max|ΔC|/(atol_uC + rtol_uC|C|) ) ≤ 1
    残差  ：R = max( ‖F_T‖_scaled, ‖F_C‖_scaled ) ≤ 1

只检查更新量会落入"更新量变小但残差停滞"的经典陷阱，故残差判据不可省。

停滞检测：更新量连续两轮不降到 0.5 倍以下 → 判定停滞，
交由外层（CODE-08）折半步长重试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fvm_cyl import implicit_step
from .nonlinear import NonlinearFailure


@dataclass
class CoupledIteration:
    """单个时间步内块迭代的收敛记录 —— 供 FIG-41、CODE-19 使用。"""
    iters: int = 0
    upd_T: float = np.inf
    upd_C: float = np.inf
    res_T: float = np.inf
    res_C: float = np.inf
    converged: bool = False
    note: str = ""


def block_picard_solve(T_n, C_n, dt, theta, assemble_T, assemble_C,
                       tol_T, tol_C, max_iter=50, T_init=None, C_init=None,
                       op_T_old=None, op_C_old=None):
    """
    求解一个隐式时间步内的耦合非线性方程。

    参数
    ----
    T_n, C_n     : (N,) 上一时刻的解
    dt, theta    : 时间步长与 θ（生产用 θ=1，Backward Euler）
    assemble_C   : callable(C_guess, T_guess) -> Operator
                   水分算子；D 依赖 (C, T)，故两者都要传
    assemble_T   : callable(T_guess, C_guess) -> Operator
                   温度算子；ρc_p、k 依赖 C；M1 时 b 还依赖 C 决定的 J_w
    tol_T, tol_C : dict(atol_u, rtol_u, atol_r, rtol_r)
    T_init/C_init: 初值猜测（None 时用上一时刻值）

    返回
    ----
    (T_new, C_new, CoupledIteration)

    异常
    ----
    NonlinearFailure : 未收敛或判定停滞（外层折半步长重试）
    """
    T = np.array(T_n if T_init is None else T_init, dtype=float)
    C = np.array(C_n if C_init is None else C_init, dtype=float)

    prev_c = np.inf
    upd_c = np.inf

    for k in range(1, max_iter + 1):
        C_prev = C
        T_prev = T

        # ---------- 1. 水分场（用到当前的 T） ----------
        op_C = assemble_C(C, T)
        C = implicit_step(op_C, C_n, dt, theta=theta, op_old=op_C_old)

        # ---------- 2. 温度场（用**最新**的 C，Gauss–Seidel） ----------
        op_T = assemble_T(T, C)
        T = implicit_step(op_T, T_n, dt, theta=theta, op_old=op_T_old)

        # ---------- 3. 双重判据 ----------
        upd_T = float(np.max(np.abs(T - T_prev) /
                             (tol_T["atol_u"] + tol_T["rtol_u"] * np.abs(T))))
        upd_C = float(np.max(np.abs(C - C_prev) /
                             (tol_C["atol_u"] + tol_C["rtol_u"] * np.abs(C))))

        # 真残差：在**新解处重新装配**算子后计算 F(u^{k+1})
        op_Cn = assemble_C(C, T)
        F_C = (C - C_n) / dt - op_Cn.matvec(C) - op_Cn.b
        res_C = float(np.max(np.abs(F_C) /
                             (tol_C["atol_r"] + tol_C["rtol_r"] *
                              np.maximum(np.abs(C / dt), 1e-3))))

        op_Tn = assemble_T(T, C)
        F_T = (T - T_n) / dt - op_Tn.matvec(T) - op_Tn.b
        # 温度残差按"每步温升尺度"归一化；1e-3 作下限防止全程恒温时过严
        res_T = float(np.max(np.abs(F_T) /
                             (tol_T["atol_r"] + tol_T["rtol_r"] *
                              np.maximum(np.abs(T / dt), 1e-3))))

        upd = max(upd_T, upd_C)
        res = max(res_T, res_C)

        if (upd <= 1.0) and (res <= 1.0):
            return T, C, CoupledIteration(iters=k, upd_T=upd_T, upd_C=upd_C,
                                          res_T=res_T, res_C=res_C,
                                          converged=True)

        # 停滞检测：更新量不再显著下降（对两个场的联合更新量判定）
        if k > 3 and upd > 0.5 * prev_c and upd > 1.0:
            raise NonlinearFailure(
                f"块 Picard 停滞于第 {k} 次迭代：联合更新量 {upd:.3e}"
                f"（T {upd_T:.3e} / C {upd_C:.3e}），残差 {res:.3e}"
                f"（T {res_T:.3e} / C {res_C:.3e}）"
            )
        prev_c = upd
        upd_c = upd

    raise NonlinearFailure(
        f"块 Picard 达到最大迭代次数 {max_iter}：联合更新量 {upd_c:.3e}，残差 {res:.3e}"
    )


# ==========================================================================
# 耦合强度诊断（供论文与 CODE-25 使用）
# ==========================================================================
def coupling_numbers(grid, D_cell, alpha_cell, C_cell, T_cell, dt,
                     h_m, h_bc_T, C_inf, T_inf):
    """
    量化"这一步里耦合到底有多强"，输出三个可报告的指标：

    * ``max_dD_dT_rel`` : 一步内由 ΔT 引起的 D 的相对变化
                          = |∂lnD/∂T|·ΔT，∂lnD/∂T = 3850/T_K²
      （附录3 的 D 对 T 呈 Arrhenius 型，3850/T_K² 在 300—330 K 上约为
        4.3e-2 ~ 3.5e-2 K⁻¹，即每升高 1 K 扩散系数增大约 3.5%~4.3%。）

    * ``Bi_m`` : h_m·R0/D —— 传质 Biot 数，判断内/外阻力谁主导
    * ``Lu``   : D/α —— Luikov 数，传质与传热时间尺度之比
    """
    T_K = np.asarray(T_cell, dtype=float) + 273.15
    dlnD_dT = 3850.0 / T_K ** 2
    return {
        "dlnD_dT_per_K": dlnD_dT,
        "Bi_m": h_m * grid.R0 / np.maximum(D_cell, 1e-300),
        "Lu": D_cell / np.asarray(alpha_cell, dtype=float),
    }
