"""
CODE-09  非线性迭代器                  [甲 · M0 · Day1下午]

问题
----
水分方程  ∂C/∂t = ∇·(D(C)∇C) 中 D 依赖 C，且 D 在 C∈[0.15, 2.55] 上跨越约 7 个数量级，
方程强刚性。隐式格式每步需求解非线性代数方程组。

方法：Picard 迭代（主）
-----------------------
    (I/Δt − A(C^(k))) C^(k+1) = C^n/Δt + b(C^(k))

选 Picard 而非 Newton 的理由：
  * 迭代矩阵保持 M-矩阵性质 → 每个迭代满足极值原理 → **C 的正性由构造保证**，
    无需任何裁剪，也不会像 Newton 那样过冲进入 exp(−0.89/C) 的悬崖区
  * 实现简单、无需 Jacobian
Newton 作为可选后备（final tightening / Day2 耦合问题复用）。

收敛判据（**尺度归一化的双重判据，两个都必须满足**）
----------------------------------------------------
    更新量：  max_i |C^(k+1)_i − C^(k)_i| / (atol_u + rtol_u·|C^(k+1)_i|) ≤ 1
    残差  ：  max_i |F_i(C^(k+1))| / (atol_r + rtol_r·max(|(C/Δt)_i|, 1e-3)) ≤ 1

**残差判据是关键**：只检查更新量会落入"更新量变小但残差停滞"的经典陷阱。

停滞检测
--------
若更新量范数在某次迭代中未能缩小到上一轮的 0.5 倍以下 → 判定停滞，返回失败。
失败由外层步长折半重试处理（见 CODE-08）。
"""

from __future__ import annotations

import numpy as np

from .fvm_cyl import implicit_step


class NonlinearFailure(RuntimeError):
    """非线性迭代未收敛。由外层步长折半重试处理。"""


def picard_solve(assemble_fn, phi_n, dt, theta, tol, max_iter,
                 phi_init=None, op_old=None):
    """
    Picard 迭代求解单个隐式时间步。

    参数
    ----
    assemble_fn : callable(phi) -> Operator     给定 phi 装配算子
    phi_n       : (N,) 上一时刻的解
    dt          : 步长
    theta       : 时间格式参数
    tol         : 收敛容限对象（需含 atol_u/rtol_u/atol_r/rtol_r）
    max_iter    : 最大迭代次数
    phi_init    : 初值猜测；None 时用 phi_n
    op_old      : θ<1 时上一时刻的算子

    返回
    ----
    (phi, n_iter, residual_norm)

    异常
    ----
    NonlinearFailure : 迭代未收敛或判定停滞
    """
    phi = np.array(phi_n if phi_init is None else phi_init, dtype=float)
    prev_upd = np.inf

    for k in range(1, max_iter + 1):
        op = assemble_fn(phi)
        phi_new = implicit_step(op, phi_n, dt, theta=theta, op_old=op_old)

        denom_u = tol["atol_u"] + tol["rtol_u"] * np.abs(phi_new)
        upd = float(np.max(np.abs(phi_new - phi) / denom_u))

        # 真残差：在 phi_new 处重新装配算子后计算 F(phi_new)
        op_new = assemble_fn(phi_new)
        F = (phi_new - phi_n) / dt - op_new.matvec(phi_new) - op_new.b
        denom_r = tol["atol_r"] + tol["rtol_r"] * np.maximum(np.abs(phi_new / dt), 1e-3)
        res = float(np.max(np.abs(F) / denom_r))

        if (upd <= 1.0) and (res <= 1.0):
            return phi_new, k, res

        # 停滞检测：更新量不再显著下降
        if k > 2 and upd > 0.5 * prev_upd and upd > 1.0:
            raise NonlinearFailure(
                f"Picard 停滞于第 {k} 次迭代：更新量 {upd:.3e}，残差 {res:.3e}"
            )
        prev_upd = upd
        phi = phi_new

    raise NonlinearFailure(
        f"Picard 达到最大迭代次数 {max_iter}：更新量 {upd:.3e}，残差 {res:.3e}"
    )


def jacobian_diag_D(C, D, dD_dC):
    """
    Newton 后备用的 Jacobian 对角元（Day2 耦合问题复用）。

    对界面通量使用调和平均时，dΓ_{i+1/2}/dC_i = 2 D_{i+1}²/(D_i+D_{i+1})² · dD_i/dC_i
    此处仅返回单元中心的 dD/dC，供上层组装。
    """
    C = np.asarray(C, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        d = 0.89 / np.maximum(C, 1e-12) ** 2 * np.asarray(D, dtype=float)
    d[~np.isfinite(d)] = 0.0
    return d
