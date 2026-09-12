"""
CODE-19  空间 / 时间误差**分开**细化        [甲 · M0 · Day2晚间]

为什么必须分开
--------------
若同时加密网格与缩小步长，量到的收敛阶是两者混合的结果，
既不能证明空间二阶、也不能证明时间一阶。
做法：**一个固定到极细，只细化另一个**。

清单 §CODE-19 的三条硬性要求
-----------------------------
1. **斜率符号**：横轴为**步长 h** 且 E ∝ h² 时斜率为 **+2**；
   横轴取 1/h 或网格数 N 时才是 −2。本模块统一以 h 为横轴，故报告 **+p**。
2. **不要求所有误差都二阶**：Backward Euler 时间一阶（比值→2），
   Crank–Nicolson 在足够光滑时二阶（比值→4）。二者的期望值不同，**分开判定**。
3. **基准网格的离散误差应显著小于 4 位小数的舍入量级（5e-5）**。

关注的量（清单指定）
--------------------
    中心温度 T(0,t)、中心含水率 C(0,t)、全域最大值 C_max(t)
（t_* 的误差由 CODE-15 单独给出，因为它是**阈值的通过时刻**，
 对误差的敏感度与场量不同 —— 场量误差 1e-3 可能对应 t_* 误差数十分钟。）

⚠️ 本模块报告的是**数值离散误差**，不含模型误差与参数误差。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import NUMERICS
from ..io.resample import center_value
from ..numerics.fvm_cyl import Grid
from ..models.boundary import BoundaryConfig
from ..models.problem2 import Problem2Setup, solve_problem2


# ==========================================================================
# 定步长推进（时间细化专用：必须能**强制**步长，不能交给自适应控制）
# ==========================================================================
def integrate_fixed_dt(setup: Problem2Setup, dt: float, t_end: float,
                       dt_warmup: float | None = None, t_warmup: float = 0.0):
    """
    以**严格等步长** dt 推进到 t_end，返回终态 (T, C)。

    🔴 为什么不能靠"把 dt_min/dt_max 钉在 dt"来实现
    -------------------------------------------------
    自适应器在步被拒绝时会执行 ``dt ← max(dt·factor, dt_min)``。
    一旦 dt_min = dt，步长**永远缩不下去**，于是同一时刻无限重试、
    永不前进 —— 实测直接卡死（不是变慢，是死循环）。
    故定步长必须**绕开 AdaptiveStepper**，直接循环调用 build_step_fn。

    dt_warmup/t_warmup：M1 的 √t 起始层（见 time_stepper 的说明）。
    ⚠️ 收敛阶统计必须**所有网格/步长用同一套起始层处理**，
       否则量到的差异里混进了起始层误差，阶数不可信。
    """
    import dataclasses
    from ..models.problem2 import build_step_fn

    sub = dataclasses.replace(setup, t_end=float(t_end))
    step_fn, _ = build_step_fn(sub)
    state = {"T": np.full(sub.grid.N, sub.T_init),
             "C": np.full(sub.grid.N, sub.C_init)}
    t = 0.0

    if t_warmup > 0.0 and dt_warmup is not None:
        n_w = int(round(t_warmup / dt_warmup))
        for _ in range(n_w):
            state, _ = step_fn(t, dt_warmup, state)
            t += dt_warmup

    n = int(round((t_end - t) / dt))
    assert abs(t + n * dt - t_end) < 1e-9 * max(1.0, t_end), \
        f"dt={dt:g} 无法整除剩余区间 [{t:g}, {t_end:g}]"
    for _ in range(n):
        state, _ = step_fn(t, dt, state)
        t += dt
    return state["T"], state["C"]


# ==========================================================================
# 观测泛函
# ==========================================================================
def observables(T_cells, C_cells, grid) -> dict:
    """一次运行 → 三个标量观测量。"""
    return {
        "T_center": float(center_value(T_cells, grid)),
        "C_center": float(center_value(C_cells, grid)),
        "C_max": max(float(np.max(C_cells)), float(center_value(C_cells, grid))),
    }


# ==========================================================================
# 收敛阶拟合
# ==========================================================================
def fit_order(hs, es) -> float:
    """
    以 log h 为横轴做最小二乘，返回斜率（= 收敛阶 p，**正号**）。

    ⚠️ 横轴是步长 h，E ∝ h^p 时斜率为 **+p**；写成 −p 是横轴取了 1/h 或 N 的缘故。
    """
    hs = np.asarray(hs, dtype=float)
    es = np.asarray(es, dtype=float)
    m = (hs > 0) & (es > 0) & np.isfinite(es)
    if m.sum() < 2:
        return float("nan")
    p = np.polyfit(np.log(hs[m]), np.log(es[m]), 1)
    return float(p[0])


def successive_ratios(es) -> list:
    """相邻误差比（用于与期望比值 2^p 对照）。"""
    e = np.asarray(es, dtype=float)
    return [float(e[i] / e[i + 1]) for i in range(len(e) - 1) if e[i + 1] > 0]


# ==========================================================================
# 空间细化（时间固定到极细）
# ==========================================================================
def spatial_study(env, t_end=10800.0, Ns=(40, 80, 160), N_ref=320,
                  refine=2, dt_fine=None, grading=1.0, theta=1.0,
                  bc=None, ref_factory=None) -> dict:
    """
    空间阶数。

    时间固定为 dt_fine（默认取 N_ref 下的一个很小值），
    使时间误差**远小于**最粗网格的空间误差，从而比值反映空间阶。

    ⚠️ 时间误差是所有网格**共有**的常数项，会污染最细网格的点；
    因此判定阶数时以**前几个较粗网格的相邻比**为准（那里空间误差主导），
    并在结果中显式标出被污染的点。
    """
    bc = bc or BoundaryConfig()
    if dt_fine is None:
        dt_fine = 1.0
    old_num = None
    out = {"dt_fine": dt_fine, "t_end": t_end, "grading": grading,
           "rows": [], "Ns": list(Ns), "N_ref": N_ref}

    ref = ref_factory() if ref_factory else None
    if ref is None:
        grid_ref = Grid(N=N_ref, R0=0.02, grading=grading)
        Tre, Cre = integrate_fixed_dt(
            Problem2Setup(grid=grid_ref, env=env, bc=bc, theta=theta), dt_fine, t_end)
        ref = observables(Tre, Cre, grid_ref)

    out["ref"] = ref
    for N in Ns:
        grid = Grid(N=N, R0=0.02, grading=grading)
        T, C = integrate_fixed_dt(
            Problem2Setup(grid=grid, env=env, bc=bc, theta=theta), dt_fine, t_end)
        obs = observables(T, C, grid)
        out["rows"].append({
            "N": N, "h": 0.02 / N,
            **{f"err_{k}": abs(obs[k] - ref[k]) for k in ref},
            **{f"val_{k}": obs[k] for k in ref},
        })
    for k in ref:
        es = [r[f"err_{k}"] for r in out["rows"]]
        out[f"ratio_{k}"] = successive_ratios(es)
        out[f"order_{k}"] = fit_order([r["h"] for r in out["rows"]], es)
    return out


# ==========================================================================
# 时间细化（空间固定到生产网格或更细）
# ==========================================================================
def temporal_study(env, t_end=10800.0, dts=(40.0, 20.0, 10.0, 5.0),
                   dt_ref=None, N=200, grading=1.5, theta=1.0, bc=None) -> dict:
    """
    时间阶数。空间固定，只缩小步长。

    期望：θ=1.0（Backward Euler）→ 比值 → 2，斜率 → **+1**；
          θ=0.5（Crank–Nicolson，光滑条件下）→ 比值 → 4，斜率 → **+2**。
    """
    bc = bc or BoundaryConfig()
    if dt_ref is None:
        dt_ref = dts[-1] / 2.0
    grid = Grid(N=N, R0=0.02, grading=grading)
    out = {"t_end": t_end, "N": N, "grading": grading, "theta": theta,
           "dt_ref": dt_ref, "rows": [], "dts": list(dts)}

    T, C = integrate_fixed_dt(
        Problem2Setup(grid=grid, env=env, bc=bc, theta=theta), dt_ref, t_end)
    ref = observables(T, C, grid)
    out["ref"] = ref

    for dt in dts:
        T, C = integrate_fixed_dt(
            Problem2Setup(grid=grid, env=env, bc=bc, theta=theta), dt, t_end)
        obs = observables(T, C, grid)
        out["rows"].append({
            "dt": dt, "h": dt,
            **{f"err_{k}": abs(obs[k] - ref[k]) for k in ref},
            **{f"val_{k}": obs[k] for k in ref},
        })
    for k in ref:
        es = [r[f"err_{k}"] for r in out["rows"]]
        out[f"ratio_{k}"] = successive_ratios(es)
        out[f"order_{k}"] = fit_order([r["h"] for r in out["rows"]], es)
    expected = 1.0 if theta == 1.0 else 2.0
    out["expected_order"] = expected
    out["expected_ratio"] = 2.0 ** expected
    return out


# ==========================================================================
# 生产网格 vs 细参考网格
# ==========================================================================
def production_vs_reference(env, t_end=10800.0, prod=(200, 1.5),
                            ref_grids=((640, 1.0), (400, 1.5)),
                            dt_fine=1.0, bc=None) -> dict:
    """
    生产网格的可信度：与更细的网格逐点比对（覆盖全部 21 个输出列）。
    判据：最大偏差是否**显著小于 4 位小数的舍入量子 5e-5**。
    """
    bc = bc or BoundaryConfig()
    from ..io.resample import output_radii_m, sample_field
    from ..numerics.fvm_cyl import surface_value
    from ..numerics.properties import D_app3, cp_app3, k_app3, rho_app3

    r_out = output_radii_m()

    def run(N, grading):
        grid = Grid(N=N, R0=0.02, grading=grading)
        # 🔴 t_end 必须显式传！Problem2Setup 的默认 t_end=10800 s（问题2 的 3 h），
        #    不传就会无视本函数的 t_end 参数、每次都跑满 3 h ——
        #    表现是"无论把对比时长设成 600 还是 3600，耗时都一样长"。
        #    （同一类"漏传参数、静默落回默认值"的 bug 本日出现三次，
        #      另两次在 scenarios.run_scenarios 与 sustained_check。）
        # 两套网格都用**同一组自适应容差**（不强制等步长）：
        # 这样量到的是"两套离散的总差异"，正是"生产网格够不够用"要回答的问题。
        setup = Problem2Setup(grid=grid, env=env, bc=bc, t_end=float(t_end))
        res, ex = solve_problem2(setup, t_output=np.array([float(t_end)]))
        return grid, res, ex

    grid_p, res_p, ex_p = run(*prod)
    Tp = sample_field(res_p.T, grid_p, ex_p["T_surf"], r_out)
    Cp = sample_field(res_p.C, grid_p, ex_p["C_surf"], r_out)

    rows = []
    for N, grading in ref_grids:
        grid_r, res_r, ex_r = run(N, grading)
        Tr = sample_field(res_r.T, grid_r, ex_r["T_surf"], r_out)
        Cr = sample_field(res_r.C, grid_r, ex_r["C_surf"], r_out)
        rows.append({
            "N": N, "grading": grading,
            "max_dT": float(np.max(np.abs(Tp - Tr))),
            "max_dC": float(np.max(np.abs(Cp - Cr))),
            "max_dC_inner": float(np.max(np.abs(Cp[0, :-1] - Cr[0, :-1]))),
        })
    return {"t_end": t_end, "prod": list(prod), "rows": rows,
            "rounding_quantum": 5e-5,
            "ok": all(r["max_dT"] < 5e-5 and r["max_dC"] < 5e-5 for r in rows)}


# ==========================================================================
# 与解析基准（Bessel）的对照 —— 复用 CODE-18，此处只做接口
# ==========================================================================
def against_bessel(bi, alpha, R0=0.02, times=(60.0, 300.0, 900.0, 1800.0),
                   N=40, n_terms=60) -> dict:
    """
    无限长圆柱 Robin 问题的解析解对照（三个条件必须同时满足：
    ① 固定热学物性 ② 关闭蒸发热效应 ③ 满足解析解要求的恒定环境边界）。
    """
    from .bessel_bench import robin_eigenvalues, CylinderAnalytic
    mu = robin_eigenvalues(bi, n_terms)
    ana = CylinderAnalytic(R0=R0, alpha=alpha, T_inf=50.0, T_init=28.0, mu=mu)
    return {"bi": bi, "n_terms": n_terms, "mu1": float(mu[0]),
            "mu_head": [float(x) for x in mu[:5]],
            "T_at": {t: float(ana.T(0.0, t)) for t in times}}
