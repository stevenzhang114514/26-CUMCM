"""
CODE-26  二维轴对称有限圆柱模型（端面效应）        [甲 · M0 · Day3 追加]

为什么需要它
------------
问题1—3 全部用**无限长圆柱**的一维径向模型。题面给的药材长 25 cm、半径 2 cm，
长径比 $AR=L/(2R_0)=6.25$ —— 端面**不是**无限小：

* 端面面积 / 侧面面积 $= 2\\pi R_0^2/(2\\pi R_0 L) = R_0/L = 8\\%$
* 研报 [R4-F20] 称忽略端面会使**全域平均含水率**偏差 3.5%~6.0%

故必须回答：一维径向简化到底错多少？本模块用**二维轴对称**有限圆柱直接算。

几何与坐标
----------
    r ∈ [0, R0]     径向（R0 = 2 cm）
    z ∈ [0, L/2]    轴向，**取半长**并用 z=0 处的对称性把计算域减半
                    （L/2 = 12.5 cm；用全长建模要多一倍网格而结果相同）

控制方程（附录3 变物性，与一维同一套）
--------------------------------------
    ρ(C)c_p(C) ∂T/∂t = (1/r)∂_r(k r ∂_r T) + ∂_z(k ∂_z T)
    ∂C/∂t            = (1/r)∂_r(D r ∂_r C) + ∂_z(D ∂_z C)

边界
----
    r = R0 : −k∂T/∂r = h(T−T∞)      −D∂C/∂r = h_m(C−C∞)
    z = L/2: −k∂T/∂z = h(T−T∞)      −D∂C/∂z = h_m(C−C∞)     ← 端面，与侧面同系数
    r = 0  : 对称（A_r = 2πr·dz → 0）
    z = 0  : 对称（A_z 置 0）

🔴 内建回归检验
--------------
`check_insulated_end()`：把**端面**的 h、h_m 设为 0，
二维解必须**逐点退化为一维径向解**。这是本模块唯一有意义的正确性证明
—— 不通过就不能用它的端面结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..config import H_COEF, HM_COEF, NUMERICS
from ..numerics.properties import D_app3, cp_app3, k_app3, rho_app3
from ..numerics.fvm_cyl import Grid, face_diffusivity


# ==========================================================================
@dataclass
class Grid2D:
    """(r, z) 二维网格。r 向表面加密、z 向端面加密（两者都有边界层）。"""
    Nr: int = 40
    Nz: int = 25
    R0: float = 0.02
    H: float = 0.125          # 半长 = L/2
    grading_r: float = 1.5
    grading_z: float = 1.2

    def __post_init__(self):
        def faces(N, L, g):
            zeta = np.arange(N + 1) / N
            return L * (1.0 - (1.0 - zeta) ** g)

        self.rf = faces(self.Nr, self.R0, self.grading_r)
        self.zf = faces(self.Nz, self.H, self.grading_z)
        self.rc = 0.5 * (self.rf[:-1] + self.rf[1:])
        self.zc = 0.5 * (self.zf[:-1] + self.zf[1:])
        self.dr = np.diff(self.rf)
        self.dz = np.diff(self.zf)

    # ---- 几何量（只用径向差与面位置，π 会约掉，故都不带 π）----
    def volumes(self):
        """V[i,j] ∝ (rf[i+1]²−rf[i]²)·dz[j]"""
        ann = (self.rf[1:] ** 2 - self.rf[:-1] ** 2)          # (Nr,)
        return ann[:, None] * self.dz[None, :]                # (Nr,Nz)

    def area_r(self):
        """径向导热/扩散面：A_r[i] ∝ 2·rf[i]·dz[j]；i=0 时 rf=0 → 自然为 0"""
        return 2.0 * self.rf[:, None] * self.dz[None, :]      # (Nr+1, Nz)

    def area_z(self):
        """轴向面：A_z[j] ∝ (rf[i+1]²−rf[i]²)；j=0 处置 0（对称）"""
        ann = (self.rf[1:] ** 2 - self.rf[:-1] ** 2)
        A = np.repeat(ann[:, None], self.Nz + 1, axis=1)
        A[:, 0] = 0.0
        return A                                              # (Nr, Nz+1)

    def summary(self):
        return (f"2D 轴对称 Nr={self.Nr}×Nz={self.Nz}, R0={self.R0*100:.1f}cm, "
                f"H={self.H*100:.1f}cm (L={2*self.H*100:.0f}cm), "
                f"γr={self.grading_r}, γz={self.grading_z}, "
                f"Δr={self.dr.min()*1e3:.3f}~{self.dr.max()*1e3:.3f}mm, "
                f"Δz={self.dz.min()*1e3:.3f}~{self.dz.max()*1e3:.3f}mm")


# ==========================================================================
def _face_dx(c, N):
    """
    面到两侧单元中心的**中心距**（不是半距）。

    🔴 这里踩过一次坑：最初写成 `0.5*(c[1:]-c[:-1])`（半距），
    使相邻单元的扩散导度**大一倍**，表现为二维扩散快一倍 ——
    在"端面绝热时 2D 必须等于 1D"的回归检验里被抓出来
    （2D 与 1D 差 0.489，而沿 z 的不均匀度只有 1e-14，
      说明错在径向而不在轴向）。

    一维内核 `fvm_cyl` 的约定是 **h[i] = rc[i] − rc[i−1]**（全长），
    本函数必须与之一致，否则两个求解器不可比。
    """
    d = np.empty(N + 1)
    d[0] = c[0]                    # 轴心到第 0 个中心；该面 A=0，不参与通量
    d[N] = c[N - 1]                # 占位；边界面用 h·A，不用此值
    d[1:N] = c[1:] - c[:-1]        # ← 全长中心距
    return d


def _harm2d(a, b):
    """逐元素的调和平均 2ab/(a+b)，带零保护。"""
    s = a + b
    out = np.zeros_like(s, dtype=float)
    m = s > 0
    out[m] = 2.0 * a[m] * b[m] / s[m]
    return out


def _assemble2d(g: Grid2D, Gam, h_bc, phi_inf, cap=None,
                mask_beta=None, h_end=None):
    """
    组装二维算子（**只对扩散部分**，时间项在调用处处理）。

    Gam       : (Nr,Nz) 单元扩散系数（k 或 D）
    h_bc      : 侧面（r=R0）传递系数
    h_end     : 端面（z=H）传递系数；None → 与 h_bc 相同（题设口径）
                ★ 置 0 可得"端面绝热"，用于内建回归检验
    phi_inf   : 环境值
    cap       : (Nr,Nz) 时间项系数；None → 1
    mask_beta : (Nr,Nz) 扩散系数修正（结壳：壳内 ×β）

    返回 (A, b, NV)：A φ_new = b，NV 为 V·cap（时间项对角的乘子）。
    """
    Nr, Nz = g.Nr, g.Nz
    V = g.volumes()                                   # (Nr,Nz)
    Ar = g.area_r()                                   # (Nr+1,Nz)
    Az = g.area_z()                                   # (Nr,Nz+1)
    if cap is None:
        cap = np.ones((Nr, Nz))
    if mask_beta is not None:
        Gam = Gam * mask_beta
    if h_end is None:
        h_end = h_bc

    # ---- 面上的扩散系数（二维调和平均）----
    Gr = np.zeros((Nr + 1, Nz))
    Gr[1:Nr] = _harm2d(Gam[:-1, :], Gam[1:, :])       # (Nr-1,Nz)
    Gz = np.zeros((Nr, Nz + 1))
    Gz[:, 1:Nz] = _harm2d(Gam[:, :-1], Gam[:, 1:])    # (Nr,Nz-1)

    drf = _face_dx(g.rc, Nr)                          # (Nr+1,)
    dzf = _face_dx(g.zc, Nz)

    # ---- 面导度：cond = A·Γ/Δx ----
    Cr = np.zeros((Nr + 1, Nz))
    Cr[1:Nr] = Ar[1:Nr] * Gr[1:Nr] / drf[1:Nr, None]
    Cz = np.zeros((Nr, Nz + 1))
    Cz[:, 1:Nz] = Az[:, 1:Nz] * Gz[:, 1:Nz] / dzf[1:Nz][None, :]

    # ---- 表面导度：外膜与半单元的**串联热阻/质阻** ----
    # 🔴 口径必须与一维内核 `fvm_cyl` 一致，否则两个求解器不可比。
    #    一维的做法是 beta = h·A/(V·(1+Bi_d))，其中 Bi_d = h·d/Γ，d 为
    #    **最外层单元中心到表面的距离**。等价地：
    #        有效传递系数 = 1/(1/h + d/Γ)   （外膜 1/h 与半单元 d/Γ 串联）
    #    若直接用 h，边界通量会偏大 O(Bi_d)，端面绝热的回归检验里表现为
    #    2D 与 1D 差 3.4e-3（改用串联式后应降到判据以下）。
    #    h=0 时给出 0，绝热边界自动成立 ✓
    # 🔴 h=0 时必须给出**导度 0**：`1/(1/h + d/Γ)` 在 h=0 处是 `1/(∞+d/Γ)=0`。
    #    曾写成 `1.0/np.where(h>0, h, np.inf)` —— 那是把 1/h 算成了 0，
    #    分母只剩 d/Γ，反而给出一个**有限**导度，端面绝热立刻失效
    #    （回归检验里"沿 z 的不均匀度"从 1e-14 暴涨到 0.15~0.93）。
    #    h 是标量，直接用 Python 判断最清楚。
    one_over_h_r = np.inf if h_bc <= 0 else 1.0 / h_bc
    one_over_h_z = np.inf if h_end <= 0 else 1.0 / h_end
    d_r = g.R0 - g.rc[-1]
    Cr[Nr] = Ar[Nr] / (one_over_h_r + d_r / Gam[-1, :])
    d_z = g.H - g.zc[-1]
    Cz[:, Nz] = Az[:, Nz] / (one_over_h_z + d_z / Gam[:, -1])

    # ---- 组装 ----
    NV = V * cap
    diag = (Cr[1:, :] + Cr[:-1, :] + Cz[:, 1:] + Cz[:, :-1]).ravel()
    # ---- 边界的 phi_inf 源项 ----
    # 🔴 只能加到**紧邻边界的那一层**单元上！
    #    最初写成 `(Cr[Nr][None,:] + Cz[:,Nz][:,None])*phi_inf`，那等于给
    #    **每一个单元**都加了 h·A·phi_inf 的源 —— 整个域被直接驱向 phi_inf，
    #    表现为二维剖面被抹平成一条水平线（而总水量却是对的，因为边界通量
    #    总量没变），在"端面绝热时 2D 必须等于 1D"的回归检验里被抓出来。
    b = np.zeros((Nr, Nz))
    b[Nr - 1, :] += Cr[Nr] * phi_inf          # 侧面（r=R0）
    b[:, Nz - 1] += Cz[:, Nz] * phi_inf       # 端面（z=H）
    b = b.ravel()

    idx = np.arange(Nr * Nz).reshape(Nr, Nz)

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(v.ravel())

    add(idx, idx, diag)
    # 径向邻居
    add(idx[1:, :], idx[:-1, :], -Cr[1:Nr, :])
    add(idx[:-1, :], idx[1:, :], -Cr[1:Nr, :])
    # 轴向邻居
    add(idx[:, 1:], idx[:, :-1], -Cz[:, 1:Nz])
    add(idx[:, :-1], idx[:, 1:], -Cz[:, 1:Nz])

    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(Nr * Nz, Nr * Nz)).tocsr()
    return A, b, NV.ravel()


# ==========================================================================
def solve_2d(env, grid2d: Grid2D, t_end, t_output, theta=1.0,
             num=None, T_init=28.0, C_init=2.55,
             h_end_scale=1.0, beta=None, verbose=False):
    """
    二维轴对称问题2/3 求解（附录3 物性、固定几何、M0 口径）。

    返回 dict：times, T, C（形状 (n_t, Nr, Nz)），C_max 轨迹等。
    """
    num = num or NUMERICS
    Nr, Nz = grid2d.Nr, grid2d.Nz
    V = grid2d.volumes()

    T = np.full((Nr, Nz), T_init)
    C = np.full((Nr, Nz), C_init)

    t = 0.0
    dt = num.dt_init
    t_out = np.asarray(t_output, dtype=float)
    k_out = 0
    n_acc = n_rej = 0

    times, Ts, Cs, Cmax_tr = [], [], [], []

    while t < t_end - 1e-12:
        dt = min(dt, t_end - t)
        if k_out < len(t_out):
            dt = min(dt, max(t_out[k_out] - t, 1e-9))
        ok = True
        try:
            T2, C2 = _step(grid2d, T, C, t, dt, env, theta, num, V,
                           h_end_scale=h_end_scale, beta=beta)
        except Exception:
            ok = False
            T2 = C2 = None

        if not ok or not (np.all(np.isfinite(T2)) and np.all(np.isfinite(C2))):
            n_rej += 1
            dt *= 0.5
            if dt < num.dt_min:
                raise RuntimeError(f"t={t:.4g} 步长触底")
            continue

        t += dt
        T, C = T2, C2
        n_acc += 1

        # ---- 步长：**几何增长到 dt_max，不做误差控制** ----
        # 🔴 为什么不在这里做步长加倍法：每接受一步就再算一步来估误差，代价直接 ×2，
        #    而二维单步本来就贵（一次稀疏求解 ≈ 1D 的 20 倍）。实测原写法
        #    在 600 s 的算例上跑掉 900 s CPU 仍未结束。
        #    改用 Backward Euler（L-稳定，无稳定性上限）+ 几何增长：
        #    时间离散误差为 O(dt_max)，本项目已实测其在 t* 上只有几秒量级
        #    （`run_day3.py --stage sens` 的 A 段：容差 ×100 只让 t* 动 16.7 s）。
        #    另设 `dt_refine` 供验证：把 dt_max 减半重跑，看 t* 变动多少。
        dt = min(dt * 1.25, num.dt_max)

        while k_out < len(t_out) and t >= t_out[k_out] - 1e-9:
            times.append(t)
            Ts.append(T.copy())
            Cs.append(C.copy())
            Cmax_tr.append(float(C.max()))
            k_out += 1
        if verbose and n_acc % 5000 == 0:
            print(f"      t={t:9.1f}s  steps={n_acc}  Cmax={C.max():.5f}  dt={dt:.3g}")

    return {"times": np.array(times), "T": np.array(Ts), "C": np.array(Cs),
            "C_max": np.array(Cmax_tr), "n_accepted": n_acc,
            "n_rejected": n_rej, "t_final": t}


def _step(g, T, C, t, dt, env, theta, num, V, h_end_scale=1.0, beta=None):
    """一个 Backward Euler 步（块 Gauss-Seidel Picard，T/C 联立）。"""
    Nr, Nz = g.Nr, g.Nz
    Tn, Cn = T.copy(), C.copy()
    T_inf = float(env.T_inf(t + dt))
    C_inf = float(env.C_inf(t + dt))
    tol = (num.picard_atol_u, num.picard_rtol_u)

    for it in range(num.picard_max_iter):
        T_old, C_old = Tn.copy(), Cn.copy()

        # ---- 水分 ----
        D = D_app3(Cn, Tn)
        if beta is not None:
            D = D * beta(Cn, Tn)
        A, b, NV = _assemble2d(g, D, HM_COEF, C_inf,
                               h_end=HM_COEF * h_end_scale)
        rhs = b + NV / dt * C.ravel()
        A = A + sp.diags(NV / dt)
        Cn_new = spla.spsolve(A, rhs).reshape(Nr, Nz)

        # ---- 温度 ----
        k = k_app3(Cn_new)
        cap = (rho_app3(Cn_new) * cp_app3(Cn_new)).reshape(Nr, Nz)
        A2, b2, NV2 = _assemble2d(g, k, H_COEF, T_inf, cap=cap,
                                  h_end=H_COEF * h_end_scale)
        rhs2 = b2 + NV2 / dt * T.ravel()
        A2 = A2 + sp.diags(NV2 / dt)
        Tn = spla.spsolve(A2, rhs2).reshape(Nr, Nz)
        Cn = Cn_new

        du = max(np.max(np.abs(Tn - T_old)), np.max(np.abs(Cn - C_old)))
        scale = max(1.0, np.max(np.abs(Tn)), np.max(np.abs(Cn)))
        if du < tol[0] + tol[1] * scale:
            break
    return Tn, Cn


# ==========================================================================
def check_insulated_end(Nr=20, Nz=12, t_end=600.0, tol_C=2e-3, dt_max=1.0):
    """
    ★ 内建回归：端面 h=h_m=0 时，二维解必须**逐点退化**为一维径向解。

    🔴 dt_max 必须取小（默认 1 s）
    ------------------------------
    一维参照 `solve_problem2` 用**自适应**步长（600 s 内约 1500 步，平均 dt≈0.4 s），
    而二维用几何增长的定步长。若二维的 dt_max 取大，**量到的差主要是时间离散差**，
    不是组装错误。实测（t_end=600 s）：

        dt_max = 30 s  → 差 4.10e-3
        dt_max =  5 s  → 差 7.25e-4
        dt_max =  1 s  → 差 1.05e-4     ← 默认
        dt_max = 0.2 s → 差 2.10e-5

    相邻比 ≈ 5~7，与 dt 的比一致 ⇒ **一阶收敛**（Backward Euler 的应有行为）。
    这说明残差纯粹是时间离散，二维组装本身正确。
    """
    import dataclasses

    from .problem2 import Problem2Setup, solve_problem2
    from .boundary import BoundaryConfig
    from ..data_prep.env_extrap import build_env
    from ..data_prep.env_interp import load_env

    env = build_env(load_env(), mode="faithful", t_pre=14400.0)
    g = Grid2D(Nr=Nr, Nz=Nz, grading_r=1.5, grading_z=1.0)
    num = dataclasses.replace(NUMERICS, dt_max=dt_max,
                              dt_init=min(NUMERICS.dt_init, dt_max))

    # 端面绝热：h_end_scale = 0
    res2 = solve_2d(env, g, t_end=t_end, t_output=np.array([t_end]),
                    h_end_scale=0.0, num=num)
    C2d = res2["C"][-1]

    # 一维径向参照
    grid1d = Grid(N=Nr, R0=g.R0, grading=g.grading_r)
    st = Problem2Setup(grid=grid1d, env=env, bc=BoundaryConfig(latent=False),
                       t_end=t_end)
    r1, _ = solve_problem2(st, t_output=np.array([t_end]))
    C1d = r1.C[-1]

    # 端面绝热时解沿 z 应处处相同 → 与整场比较取最大差
    d_all = float(np.max(np.abs(C2d - C1d[:, None])))
    d_mid = float(np.max(np.abs(C2d[:, Nz // 2] - C1d)))
    ok = d_all < tol_C
    print(f"[CODE-26 回归] 端面绝热：2D vs 1D 径向  最大逐点差 = {d_all:.3e}"
          f"（判据 {tol_C:g}） → {'✅ 通过' if ok else '❌ 未通过'}")
    print(f"                中层比较 {d_mid:.3e}；沿 z 的自身不均匀度 "
          f"{d_all - d_mid:.2e}")
    return {"通过": bool(ok), "最大逐点差": d_all, "中层差": d_mid,
            "判据": tol_C, "t_end": t_end, "Nr": Nr, "Nz": Nz}


if __name__ == "__main__":
    print(Grid2D().summary())
    check_insulated_end()
