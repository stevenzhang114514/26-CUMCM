"""
CODE-20  守恒与通量收支检验            [乙 · M0 · Day1晚间]

检验层次（由强到弱，必须分开表述）
----------------------------------
① **离散方程守恒**（应达机器精度）
   内部界面通量成对抵消，求和后只剩边界面贡献。
   这是离散格式的性质，与物理无关，必须严格成立。

② **所解方程的积分收支**（应达 1e-6 量级）
       ∫₀ᵗ A_surf·J_s(τ)dτ  ≈  Σ_i V_i[φ_i(t) − φ_i(0)]
   面积口径**必须一致**：整根圆柱侧面积是 2πR₀L₀；
   本模块统一按**单位轴向长度**计算（即 A = 2πR₀），
   因此写出的总量也是"单位长度总量"。

🔴 **不做的检验**
   "把 ρcp T 的差当作完整内能变化"——**变物性模型不宜如此**。
   严格的热力学检验需另行定义一致的能量状态基准。
   本模块只检验**所解方程**的积分收支，不冒充完整能量守恒。
"""

from __future__ import annotations

import numpy as np

from ..numerics.fvm_cyl import face_diffusivity, flux_loop


# ==========================================================================
# ① 内部界面通量成对抵消
# ==========================================================================
def flux_cancellation(phi, Gamma_cell, h_bc, phi_inf, grid, op=None):
    """
    内部界面通量成对抵消 + 算子装配正确性检验（**量纲形式**）。

        Σ_i V_i·(dφ_i/dt) = Σ_i [A_{i+1/2}J_{i+1/2} − A_{i−1/2}J_{i−1/2}] = A_N J_s

    内部面在求和时逐项抵消（望远镜），只剩边界面 Robin 通量。
    这里**独立地**用面通量循环重算散度，与算子装配给出的 op.matvec 逐点比较。

    返回 dict:
        telescope_leak : 望远镜求和的泄漏（结构性，应达机器精度）
        matvec_diff    : 通量循环 vs op.matvec 的最大相对差（应达机器精度）
    """
    N = grid.N
    A = grid.A_face                     # 2π r_f
    V = grid.V                          # 控制体体积（单位轴向长度）
    phi = np.asarray(phi, dtype=float)

    # 独立面通量循环（非均匀网格通用）
    net, Jface = flux_loop(phi, Gamma_cell, h_bc, phi_inf, grid)

    total = float(net.sum())
    boundary = -A[N] * Jface[N]
    scale = float(np.abs(net).sum()) + 1e-300
    telescope_leak = abs(total - boundary) / scale

    # ---- 与算子装配对比（扣除边界源项 b）----
    matvec_diff = 0.0
    if op is not None:
        lhs = V * op.matvec(phi)            # V·(Aφ)
        rhs = net - V * op.b                # 通量循环给出的散度 − V·b
        matvec_diff = float(np.max(np.abs(lhs - rhs))
                            / (float(np.max(np.abs(rhs))) + 1e-300))

    return {"telescope_leak": telescope_leak, "matvec_diff": matvec_diff,
            "boundary_term": boundary, "J_surface": float(Jface[N]),
            "scale": scale, "leak": max(telescope_leak, matvec_diff)}


# ==========================================================================
# ② 积分收支
# ==========================================================================
def mass_budget(times, flux_hist, C_hist, grid):
    """
    水分积分收支（**单位轴向长度**，面积口径 A = 2πR₀）。

        ∫₀ᵗ 2πR₀·J_w(τ)dτ   vs   Σ_i V_i[C_i(t) − C_i(0)]

    注意：模型以干基含水率 C 为变量、方程为 ∂C/∂t = ∇·(D∇C)，
    故左式与右式均不含 ρ_d；这是**所解方程**的收支，
    不是"严格实际水质量守恒"（后者需 ρ_d 定义，见 CODE-35）。
    """
    A_surf = 2.0 * np.pi * grid.R0            # 单位轴向长度的侧面积
    V = grid.V                                  # 单位轴向长度的控制体体积
    times = np.asarray(times, dtype=float)
    flux_hist = np.asarray(flux_hist, dtype=float)
    C_hist = np.asarray(C_hist, dtype=float)

    cum_out = np.concatenate(([0.0],
                              np.cumsum(0.5 * (flux_hist[1:] + flux_hist[:-1])
                                        * np.diff(times)))) * A_surf
    stored = (C_hist - C_hist[0]) @ V
    rel = np.abs(cum_out + stored) / max(np.abs(stored).max(), 1e-300)
    return {"cum_out": cum_out, "stored": stored, "rel_err": rel,
            "A_surf_per_unit_length": A_surf}


def heat_budget(times, flux_hist, T_hist, grid, rho, cp):
    """
    温度积分收支（**单位轴向长度**）。

        ∫₀ᵗ 2πR₀·q_s(τ)dτ  vs  Σ_i V_i·ρcp·[T_i(t) − T_i(0)]

    其中 q_s 是**向外的**对流通量  h(T_s − T∞)，由边界离散给出。
    本模块只检验**所解方程** ∂T/∂t = (1/ρcp)∇·(k∇T) 的收支；
    因常物性（附录2），此处 ρcp 为常数，收支关系式成立。
    ⚠️ 变物性时（问题2+）不得直接沿用此式。
    """
    A_surf = 2.0 * np.pi * grid.R0
    V = grid.V
    times = np.asarray(times, dtype=float)
    flux_hist = np.asarray(flux_hist, dtype=float)
    T_hist = np.asarray(T_hist, dtype=float)

    cum_in = np.concatenate(([0.0],
                             np.cumsum(0.5 * (flux_hist[1:] + flux_hist[:-1])
                                       * np.diff(times)))) * A_surf
    stored = ((T_hist - T_hist[0]) @ V) * rho * cp
    # 能量平衡：  Δ(stored) = −∮q·n dA = −cum_in   （q 为**向外**的对流通量）
    # 初期 Ts < T∞ ⇒ q < 0（热量流入）⇒ cum_in < 0，而 stored > 0，故二者相加为零。
    rel = np.abs(cum_in + stored) / max(np.abs(stored).max(), 1e-300)
    return {"cum_in": cum_in, "stored": stored, "rel_err": rel}


# ==========================================================================
def report(cons, mass, heat, verbose=True):
    out = {
        "telescope_leak": float(cons["telescope_leak"]),
        "matvec_diff": float(cons["matvec_diff"]),
        "mass_rel_err": float(np.max(mass["rel_err"])),
        "heat_rel_err": float(np.max(heat["rel_err"])),
    }
    if verbose:
        print("[CODE-20] 守恒与通量收支检验")
        print(f"    ① 内部界面通量望远镜求和泄漏 = {out['telescope_leak']:.3e}  (结构性，应达机器精度)")
        print(f"       算子装配 vs 通量循环最大相对差 = {out['matvec_diff']:.3e}")
        print(f"    ② 水分积分收支最大相对误差   = {out['mass_rel_err']:.3e}")
        print(f"    ③ 温度积分收支最大相对误差   = {out['heat_rel_err']:.3e}")
        print(f"    口径：单位轴向长度，A_surf = 2πR₀ = {mass['A_surf_per_unit_length']:.6f} m")
        print(f"    ⚠ 本检验是**所解方程**的积分收支，不是完整热力学能量守恒；")
        print(f"      变物性模型不宜直接把 ρcp·ΔT 当作完整内能变化（见 CODE-20 说明）。")
    return out
