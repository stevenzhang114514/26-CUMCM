"""
CODE-16  计算网格 → 输出采样         [丙 · M0 · Day1下午]

输出节点
--------
    距离：0.0, 0.1, 0.2, …, 2.0 cm  共 21 列
    时间：1, 2, …, 1800 s

三个区域的取值方式（各不相同，必须分开处理）
--------------------------------------------
1. **r = 0（中心）**
   不能直接用中心半控制体的**单元平均值**（那是 [0,Δr] 上的体积平均，
   对偶函数 φ = c₀ + c₂r² 有  φ̄₀ = φ(0) + c₂Δr²/2，偏差 O(Δr²)）。
   改用**偶函数二次外推**：由 (rc₀,φ₀)、(rc₁,φ₁) 两点拟合 φ = c₀ + c₂r²，
        φ(0) = (φ₀·rc₁² − φ₁·rc₀²)/(rc₁² − rc₀²)

2. **内部节点 0.1–1.9 cm**
   PCHIP 保形插值。选 PCHIP 而非三次样条的理由：**保形 → 结构上不可能过冲**；
   干燥前沿附近 C 陡降，三次样条会振铃到负值。
   （"三次插值不天然保正"—— v3 清单 CODE-16 明确要求验证无过冲）

3. **r = 2.0 cm（末列）**
   **不外插**。直接写入由 Robin 边界离散重构出的**表面值** φ_s，
   这与传热/传质边界条件所依据的量完全一致。

**输出插值不自动守恒**：题目需要的是**点值采样**，不是重映射控制体平均量。
因此这里验证的是**采样误差与无过冲**，不要求保守映射。
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..config import R_OUT_CM


def center_value(phi_cells, grid, r_ref=None):
    """
    由单元平均值外推 r = 0 处的点值（偶函数二次外推，二阶）。

    若给定 r_ref 且其网格与 grid 不同，则先插值到 grid 的单元中心，
    保证外推使用的两点来自同一套网格。
    """
    phi = np.asarray(phi_cells, dtype=float)
    rc = grid.rc
    if r_ref is not None:
        phi = PchipInterpolator(r_ref, phi_cells)(rc)
    r0, r1 = rc[0], rc[1]
    return float((phi[0] * r1 ** 2 - phi[1] * r0 ** 2) / (r1 ** 2 - r0 ** 2))


def resample_profile(phi_cells, grid, surface_phi, r_out_m):
    """
    单条径向剖面 → 输出节点。

    参数
    ----
    phi_cells  : (N,)  单元值
    surface_phi: float  边界重构得到的表面值（写到最后一个输出节点）
    r_out_m    : (M,)  输出半径（米），递增，最后一个应为 R0

    返回 (M,) 输出值
    """
    phi = np.asarray(phi_cells, dtype=float)
    r_out = np.asarray(r_out_m, dtype=float)
    rc = grid.rc
    R0 = grid.R0

    out = np.empty_like(r_out)
    # 内部节点：PCHIP（保形，无过冲）
    interp = PchipInterpolator(rc, phi, extrapolate=False)

    tol = 1e-12
    is_center = r_out <= tol
    is_surface = np.abs(r_out - R0) <= tol
    is_inner = ~(is_center | is_surface)

    out[is_inner] = interp(r_out[is_inner])
    if np.any(is_center):
        out[is_center] = center_value(phi, grid)
    if np.any(is_surface):
        out[is_surface] = float(surface_phi)
    return out


def check_no_overshoot(phi_out, phi_cells, tol=1e-12, r_out_m=None):
    """
    采样无过冲检验 —— **仅针对插值列**。

    ⚠️ 两端的列**不是插值结果**，必须排除：
        * 第 0 列（r=0）是偶函数二次**外推**到轴心
        * 末  列（r=R0）是 Robin 边界**重构**出的表面值 φ_s
      φ_s 合理地落在单元值域之外（φ_s = (φ_N + Bi_Δ·φ_∞)/(1+Bi_Δ)，
      当 φ_∞ 明显大于 φ_N 时 φ_s > φ_N），这不是过冲，而是物理正确的边界值。
    因此过冲检验只对 0 < r < R0 的内部插值列进行。

    返回 (ok, excess, info)
    """
    phi_out = np.asarray(phi_out, dtype=float)
    lo, hi = float(np.min(phi_cells)), float(np.max(phi_cells))
    if r_out_m is None:
        inner = phi_out[..., 1:-1]
    else:
        r = np.asarray(r_out_m, dtype=float)
        R0 = r[-1]
        mask = (r > 1e-12) & (r < R0 - 1e-12)
        inner = phi_out[..., mask]
    excess = max(0.0, float(np.max(inner)) - hi, lo - float(np.min(inner)))
    info = {"cell_range": (lo, hi),
            "interior_range": (float(np.min(inner)), float(np.max(inner))),
            "edge_cols_excluded": [0, phi_out.shape[-1] - 1]}
    return excess <= tol, excess, info


def sample_field(field_hist, grid, surface_phi_hist, r_out_m):
    """
    整段历史 → 输出矩阵。

    field_hist       : (n_t, N)
    surface_phi_hist : (n_t,)
    返回 (n_t, M)
    """
    n_t = field_hist.shape[0]
    M = len(r_out_m)
    out = np.empty((n_t, M), dtype=float)
    for k in range(n_t):
        out[k] = resample_profile(field_hist[k], grid, surface_phi_hist[k], r_out_m)
    return out


def output_radii_m():
    """输出半径（米）：0.0, 0.1, …, 2.0 cm"""
    return np.array(R_OUT_CM, dtype=float) * 1e-2
