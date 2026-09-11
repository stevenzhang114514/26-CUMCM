"""
CODE-21  物理合理性检查器              [乙 · M0 · Day2晚间]

逐条断言（失败即报警，不静默）

⚠️ **本模块刻意不做的一件事**
    **不设"总水量必须下降"的无条件断言。**
    含水率是否单调下降**取决于环境条件**；当环境可能造成**回湿**时
    （即 C∞ᵉq 高于当地含水率），局部含水率可以回升。
    因此这里只检查单调性的**一致性**（若回升，必须由边界驱动势解释），
    而不强制全线单调。
"""

from __future__ import annotations

import numpy as np


def check_positivity(C_hist, T_hist, C_floor=-1e-12, T_range=(0.0, 100.0)):
    C = np.asarray(C_hist, dtype=float)
    T = np.asarray(T_hist, dtype=float)
    return {
        "C_min": float(C.min()),
        "C_nonneg": bool(C.min() >= C_floor),
        "T_min": float(T.min()),
        "T_max": float(T.max()),
        "T_in_range": bool(T.min() >= T_range[0] - 1e-9 and T.max() <= T_range[1] + 1e-9),
        "all_finite": bool(np.all(np.isfinite(C)) and np.all(np.isfinite(T))),
    }


def check_symmetry(T_hist, C_hist, grid, n_probe=5):
    """
    中心对称：r=0 处应满足 ∂φ/∂r = 0。
    离散形式下等价于  φ(r) 在中心附近呈偶函数 —— 用
    |φ₀ − φ(0)| 的量级（O(Δr²)）表征。
    """
    T = np.asarray(T_hist, dtype=float)
    C = np.asarray(C_hist, dtype=float)
    rc = grid.rc
    # 二次外推的余项 ~ φ''·Δr²；用 相邻两点线性外推到 0 与二次外推之差 作为度量
    def deviation(phi):
        r0, r1 = rc[0], rc[1]
        quad = (phi[:, 0] * r1 ** 2 - phi[:, 1] * r0 ** 2) / (r1 ** 2 - r0 ** 2)
        lin = phi[:, 0] + (phi[:, 0] - phi[:, 1]) * r0 / (r1 - r0)
        return float(np.max(np.abs(quad - lin)))
    return {"T_center_dev": deviation(T), "C_center_dev": deviation(C),
            "dr": grid.dr}


def check_no_overshoot(T_hist, C_hist, T_inf_hist, T_init=28.0):
    """
    温度不得超出 [min(T∞,T0), max(T∞,T0)] 的合理外包线（无非物理过冲）。
    允许微小数值裕量。
    """
    T = np.asarray(T_hist, dtype=float)
    Tinf = np.asarray(T_inf_hist, dtype=float)
    lo = np.minimum(Tinf.min(), T_init)
    hi = np.maximum(Tinf.max(), T_init)
    return {"T_lo": float(lo), "T_hi": float(hi),
            "T_min": float(T.min()), "T_max": float(T.max()),
            "no_overshoot": bool(T.min() >= lo - 1e-6 and T.max() <= hi + 1e-6)}


def check_monotonicity_consistency(C_hist, C_inf_hist):
    """
    单调性一致性：当地含水率若回升，必须发生在 C∞ᵉq 高于该点 C 的时刻
    （即由边界驱动势解释），否则为数值异常。
    """
    C = np.asarray(C_hist, dtype=float)
    Cinf = np.asarray(C_inf_hist, dtype=float)
    dC = np.diff(C, axis=0)
    rise = dC > 1e-10
    if not rise.any():
        return {"has_rise": False, "unexplained": 0, "ok": True}
    # 回升点处：表面附近 C 与 C∞ 的关系
    c_mid = 0.5 * (C[1:] + C[:-1])
    driven = Cinf[1:, None] > c_mid          # 环境更湿 → 回升可解释
    unexplained = int(np.sum(rise & ~driven))
    return {"has_rise": True, "rise_points": int(rise.sum()),
            "unexplained": unexplained, "ok": unexplained == 0}


def run_all(T_hist, C_hist, T_inf_hist, C_inf_hist, grid, verbose=True):
    rep = {}
    rep.update(check_positivity(C_hist, T_hist))
    rep.update(check_no_overshoot(T_hist, C_hist, T_inf_hist))
    rep.update(check_monotonicity_consistency(C_hist, C_inf_hist))
    rep.update(check_symmetry(T_hist, C_hist, grid))
    rep["ok"] = all([rep["C_nonneg"], rep["T_in_range"], rep["all_finite"],
                     rep["no_overshoot"], rep["ok"] if "ok" in rep else True])
    if verbose:
        print("[CODE-21] 物理合理性检查")
        for k in ("C_nonneg", "T_in_range", "all_finite", "no_overshoot",
                  "has_rise", "unexplained"):
            if k in rep:
                print(f"    {k:16s} = {rep[k]}")
        print(f"    C ∈ [{rep['C_min']:.6f}, ]   T ∈ [{rep['T_min']:.4f}, {rep['T_max']:.4f}] °C")
        print(f"    总体: {'通过' if rep['ok'] else '**未通过**'}")
    return rep
