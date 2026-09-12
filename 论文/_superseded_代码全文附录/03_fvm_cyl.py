"""
CODE-06  一维圆柱有限体积离散内核     [甲主 / 乙验 · M0 · Day1上午]

离散方案
--------
同心环形控制体，**中心用半径 Δr/2 的实心半圆柱**，天然规避 1/r 奇点：
FVM 离散的是守恒形式  d/dt∫φ dV = ∮Γ ∂φ/∂n dA ，从不写出 1/r ∂_r(r ∂_r φ)，
因此不会出现除以趋零半径的运算。

网格（N 个控制体，均匀 Δr = R0/N）
    faces   rf[j] = j·Δr          j = 0..N      （rf[0]=0 处通量为 0）
    centers rc[i] = (i+0.5)·Δr    i = 0..N-1
    体积(单位轴向长度) V_0 = πΔr² ，  V_i = π(rf[i+1]²−rf[i]²) = 2π·rc[i]·Δr
    面积(单位轴向长度) A_j = 2π·rf[j]

界面扩散系数用**调和平均**
    Γ_{j} = 2Γ_{j-1}Γ_j / (Γ_{j-1}+Γ_j)
这是分段常数 Γ 的精确串联电阻法则。D 跨约 7 个数量级时，
算术平均会被湿侧主导、高估干燥侧受扩散限制的通量，调和平均是必需的。

离散方程（无量纲化后统一形式  dφ/dt = A φ + b）
    dφ_0/dt   = 2Γ_1(φ_1−φ_0)/Δr²
    dφ_i/dt   = [rf[i]Γ_i(φ_{i−1}−φ_i) + rf[i+1]Γ_{i+1}(φ_{i+1}−φ_i)] / (rc[i]Δr²)
    dφ_{N−1}/dt = [rf[N−1]Γ_{N−1}(φ_{N−2}−φ_{N−1}) − β(φ_{N−1}−φ_∞)] / (rc[N−1]Δr²)

Robin 边界（单元中心格式的边界处理，二阶）
    边界半控制体内设线性剖面  ∂φ/∂r ≈ (φ_s−φ_N)/(Δr/2)，与 Robin 条件联立消去 φ_s：
        φ_s = (φ_N + Bi_Δ·φ_∞)/(1 + Bi_Δ),      Bi_Δ = h_bc·Δr/(2Γ_N)
        J_s = h_bc(φ_s − φ_∞) = h_bc(φ_N − φ_∞)/(1 + Bi_Δ)
    β = R0·h_bc·Δr/(1+Bi_Δ)
注意：表面对流系数在传热方程中须除以 ρcp（即使用 h/(ρcp)），
      在水分方程中直接用 h_m。两者量纲均为 m/s，与 Γ/R0 一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import solve_banded


# ==========================================================================
# 网格几何缓存
# ==========================================================================
@dataclass(frozen=True)
class _Geometry:
    rf: np.ndarray          # 面半径 rf[0..N]
    rc: np.ndarray          # 单元中心 rc[0..N-1]
    drf: np.ndarray         # 面间距 rf[j+1] − rf[j]
    h: np.ndarray           # 相邻单元中心距，i=1..N-1
    V: np.ndarray           # 控制体体积（单位轴向长度）
    A_face: np.ndarray      # 面面积（单位轴向长度）


@lru_cache(maxsize=64)
def _geometry(N: int, R0: float, grading: float) -> _Geometry:
    """
    由 (N, R0, grading) 唯一确定的全部几何量，算一次缓存起来。

    ⚠️ 返回的数组是**共享的**：调用方**不得就地修改**。
    （本项目所有用法都是只读的；assemble 只做乘加，不写入这些数组。）
    """
    z = np.linspace(0.0, 1.0, N + 1)
    rf = R0 * z if grading == 1.0 else R0 * (1.0 - (1.0 - z) ** grading)
    rc = 0.5 * (rf[:-1] + rf[1:])
    return _Geometry(
        rf=rf, rc=rc, drf=np.diff(rf), h=np.diff(rc),
        V=np.pi * (rf[1:] ** 2 - rf[:-1] ** 2),
        A_face=2.0 * np.pi * rf,
    )


# ==========================================================================
# 网格
# ==========================================================================
@dataclass(frozen=True)
class Grid:
    """
    一维圆柱径向网格。

    grading = 1.0 → 均匀网格（Δr = R0/N），向后兼容
    grading > 1.0 → **向表面 r=R0 加密**的渐变网格：

        r(ζ) = R0·[1 − (1−ζ)^γ],   ζ = j/N,  γ = grading

      * ζ=0 → r=0，ζ=1 → r=R0
      * dr/dζ = R0·γ(1−ζ)^(γ−1) → 在表面处趋于 0，故表面附近步长最小
      * 最大/最小步长比 ≈ N^(γ−1)

    选此设计的理由：水分在**表面**形成极薄边界层
    （t=1 s 时厚度仅 √(Dt) ≈ 0.07 mm），均匀网格即便 N=640 仍欠分辨；
    而中心区剖面平坦，无需加密。渐变网格以很小的代价解决该矛盾。
    """
    N: int
    R0: float
    grading: float = 1.0

    @property
    def dr(self) -> float:
        return self.R0 / self.N

    # ★ 几何量全部走模块级缓存（Day2 新增）
    # 原先是每次访问都重算 np.linspace 与幂运算的 property。短程积分看不出来，
    # 但 120 h 的长时积分里 rf 被调用 30 万次、独占 36% 的运行时间（实测 profile）。
    # 几何只由 (N, R0, grading) 决定，缓存后**数值完全不变**，只是不再重复计算。
    @property
    def _geom(self):
        return _geometry(self.N, float(self.R0), float(self.grading))

    @property
    def rf(self) -> np.ndarray:
        """面半径 rf[0..N]"""
        return self._geom.rf

    @property
    def rc(self) -> np.ndarray:
        """单元中心半径 rc[0..N-1]（相邻两面中点）"""
        return self._geom.rc

    @property
    def h(self) -> np.ndarray:
        """
        相邻**单元中心**间距 h[i] = rc[i] − rc[i−1]，i=1..N-1。
        界面梯度用中心距而非面距，这是非均匀 FVM 的正确做法。
        """
        return self._geom.h

    @property
    def V(self) -> np.ndarray:
        """控制体体积（单位轴向长度）"""
        return self._geom.V

    @property
    def A_face(self) -> np.ndarray:
        """面面积（单位轴向长度）A_face[j] = 2π rf[j]"""
        return self._geom.A_face

    def summary(self) -> str:
        h = self._geom.drf * 1e3
        return (f"Grid N={self.N}, γ={self.grading:g}, R0={self.R0*1e3:.1f}mm | "
                f"Δr: {h.min():.5f} ~ {h.max():.5f} mm (比 {h.max()/h.min():.0f}:1)")


def harmonic_mean(g_left, g_right):
    """调和平均，0 值安全。"""
    a = np.asarray(g_left, dtype=float)
    b = np.asarray(g_right, dtype=float)
    s = a + b
    out = np.zeros_like(s)
    m = s > 0
    out[m] = 2.0 * a[m] * b[m] / s[m]
    return out


def face_diffusivity(Gamma_cell):
    """
    由单元中心值得到 N+1 个面上的扩散系数。
    face[0]  (r=0)    : 对称面，不参与通量计算，置 Gamma_cell[0]
    face[j]  (1..N-1) : 调和平均
    face[N]  (r=R0)   : 边界面，用边界单元值
    """
    G = np.asarray(Gamma_cell, dtype=float)
    N = G.size
    Gf = np.empty(N + 1, dtype=float)
    Gf[0] = G[0]
    Gf[1:N] = harmonic_mean(G[:-1], G[1:])
    Gf[N] = G[-1]
    return Gf


# ==========================================================================
# 算子装配
# ==========================================================================
@dataclass
class Operator:
    """三对角算子  dφ/dt = A φ + b  的带状存储"""
    N: int
    diag: np.ndarray      # A[i,i]
    lower: np.ndarray     # A[i,i-1]  (lower[0] 未用)
    upper: np.ndarray     # A[i,i+1]  (upper[N-1] 未用)
    b: np.ndarray         # 常数项
    beta: float           # 边界系数，供通量复算
    Bi_delta: float
    Gamma_face: np.ndarray

    def matvec(self, phi):
        out = self.diag * phi
        out[1:] += self.lower[1:] * phi[:-1]
        out[:-1] += self.upper[:-1] * phi[1:]
        return out


def assemble(Gamma_cell, h_bc, phi_inf, grid: Grid, cap_cell=None) -> Operator:
    """
    装配算子。

    参数
    ----
    Gamma_cell : (N,) 单元中心扩散系数（传热用 α=k/(ρcp)，传质用 D）
    h_bc       : 表面对流系数（传热用 h/(ρcp) 或 h；传质用 h_m），单位 m/s
    phi_inf    : 环境值（标量）
    grid       : Grid
    cap_cell   : (N,) 体积热容 (ρc_p)，**仅变物性传热的守恒形式用**

    ★ cap_cell —— 变物性传热的正确写法（Day2 新增）
    -------------------------------------------------
    控制方程  ρ(C)c_p(C) ∂T/∂t = (1/r) ∂_r( k(C) r ∂T/∂r)  中
    **变系数不得移出微分算子**。把守恒形式在控制体上积分：

        (ρc_p)_i V_i dT_i/dt = A_{i+1} k_{i+1/2}(T_{i+1}−T_i)/h_i
                             − A_i     k_{i/2}  (T_{i−1}−T_i)/h_{i−1}

    ⇒ 右端整体除以 (ρc_p)_i V_i。**面导热系数 k 用调和平均，
       而热容 (ρc_p)_i 用单元值** —— 两者性质不同，不可合并：

        ❌ 错误做法：令 Γ = k/(ρc_p) 再走原路径。那样得到的是
           (k/(ρc_p))_face，即把 k 与 ρc_p 在**同一个面上**一起做调和平均，
           与正确的 k_face/(ρc_p)_cell 不等价。本工况 ρc_p 沿 r 变化可达
           数倍（C 从 2.55 降到 0.15 时 ρc_p 由 3.34e6 降到 2.20e6），
           该差异会污染温度场。

    传入 cap_cell 时，全部除 V[i] 改为除 V[i]·cap_cell[i]，
    且 h_bc 应传**未归一化的 h**（W/(m²·K)）、Gamma_cell 传 **k**（W/(m·K)）。
    cap_cell=None 时行为与 Day1 完全一致（向后兼容）。
    """
    N, R0 = grid.N, grid.R0
    rf, rc = grid.rf, grid.rc
    h = grid.h                      # 单元中心间距 h[i] = rc[i] − rc[i−1]，i=1..N−1
    V = grid.V
    A = grid.A_face
    Gf = face_diffusivity(Gamma_cell)

    # 体积除数：V_i，或变物性传热时的 (ρc_p)_i·V_i
    if cap_cell is None:
        Vd = V
    else:
        cap = np.asarray(cap_cell, dtype=float)
        assert cap.shape == V.shape, "cap_cell 形状须与网格一致"
        assert np.all(cap > 0.0), "体积热容必须为正"
        Vd = V * cap

    diag = np.zeros(N)
    lower = np.zeros(N)
    upper = np.zeros(N)
    b = np.zeros(N)

    # ---- 中心控制体 i = 0：内侧面 r=0 通量恒为 0 ----
    # dφ₀/dt = A[1]·J₁/V[0],  J₁ = −Γ₁(φ₁−φ₀)/h[0]
    c0 = A[1] * Gf[1] / (h[0] * Vd[0])
    diag[0] = -c0
    upper[0] = +c0

    # ---- 内部及边界控制体（向量化；逐元素表达式与原循环逐一对应） ----
    ii = np.arange(1, N)                       # i = 1..N−1
    # 内侧面 rf[i] 的贡献
    a_in = A[ii] * Gf[ii] / (h[ii - 1] * Vd[ii])
    diag[ii] -= a_in
    lower[ii] = a_in

    jj = np.arange(1, N - 1)                   # i = 1..N−2
    # 外侧面 rf[i+1] 的贡献
    a_out = A[jj + 1] * Gf[jj + 1] / (h[jj] * Vd[jj])
    diag[jj] -= a_out
    upper[jj] = a_out

    # ---- Robin 边界（r = R0，落在 i = N−1）----
    # 边界单元中心到表面的距离 d = R0 − rc[N−1]
    d = R0 - rc[N - 1]
    GN = float(Gf[N])
    if GN > 0.0 and d > 0.0:
        Bi_d = h_bc * d / GN
        beta = A[N] * h_bc / ((1.0 + Bi_d) * Vd[N - 1])
    else:
        Bi_d = np.inf
        beta = A[N] * h_bc / Vd[N - 1]
    diag[N - 1] -= beta
    b[N - 1] = beta * phi_inf

    return Operator(N=N, diag=diag, lower=lower, upper=upper, b=b,
                    beta=beta, Bi_delta=Bi_d, Gamma_face=Gf)


# ==========================================================================
# 隐式单步（线性方程求解）
# ==========================================================================
def _as_banded(diag, lower, upper, alpha):
    """构造 solve_banded 需要的 (2, N) 带状矩阵： 解 (alpha·I - A) x = rhs"""
    N = diag.size
    ab = np.zeros((3, N))
    ab[0, 1:] = -upper[:-1]        # 超对角
    ab[1, :] = alpha - diag        # 主对角
    ab[2, :-1] = -lower[1:]        # 次对角
    return ab


def implicit_step(op: Operator, phi_n, dt, theta=1.0, op_old: Operator | None = None):
    """
    θ-方法单步：  (I/Δt − θA) φ^{n+1} = (I/Δt + (1−θ)A) φ^n + θ b^{n+1} + (1−θ) b^n

    theta=1.0 → Backward Euler（L-稳定，生产用）
    theta=0.5 → Crank–Nicolson（仅用于线性温度基准验证时间二阶）
    """
    if op_old is None:
        op_old = op
    N = op.N
    rhs = phi_n / dt
    if theta < 1.0:
        rhs += (1.0 - theta) * op_old.matvec(phi_n)
        rhs += (1.0 - theta) * op_old.b
    rhs += theta * op.b

    ab = _as_banded(op.diag * theta, op.lower * theta, op.upper * theta, 1.0 / dt)
    return solve_banded((1, 1), ab, rhs, check_finite=False)


# ==========================================================================
# 表面重构与通量（供输出、守恒检验使用）
# ==========================================================================
def surface_distance(grid: Grid) -> float:
    """边界单元中心到表面 r=R0 的距离 d = R0 − rc[N−1]（均匀网格时为 Δr/2）。"""
    return float(grid.R0 - grid.rc[-1])


def _bi_delta(gamma_N, h_bc, grid: Grid):
    """边界半单元 Biot 数 Bi_d = h_bc·d/Γ_N，d 为边界单元中心到表面的距离。"""
    d = surface_distance(grid)
    if gamma_N <= 0.0 or d <= 0.0:
        return np.inf
    return h_bc * d / float(gamma_N)


def surface_value(phi, Gamma_cell, h_bc, phi_inf, grid: Grid):
    """
    由单元中心值重构表面值（二阶）：
        φ_s = (φ_N + Bi_d·φ_∞)/(1 + Bi_d),   Bi_d = h_bc·d/Γ_N
    """
    G_N = float(np.asarray(Gamma_cell)[-1])
    Bi_d = _bi_delta(G_N, h_bc, grid)
    if not np.isfinite(Bi_d):
        return float(phi_inf)
    return float((phi[-1] + Bi_d * phi_inf) / (1.0 + Bi_d))


def surface_flux(phi, Gamma_cell, h_bc, phi_inf, grid: Grid):
    """表面处**向外**的通量 J_s = h_bc(φ_N − φ_∞)/(1 + Bi_d)"""
    G_N = float(np.asarray(Gamma_cell)[-1])
    Bi_d = _bi_delta(G_N, h_bc, grid)
    if not np.isfinite(Bi_d):
        return 0.0
    return float(h_bc * (phi[-1] - phi_inf) / (1.0 + Bi_d))


def flux_loop(phi, Gamma_cell, h_bc, phi_inf, grid: Grid):
    """
    独立用面通量循环重算散度（非均匀网格通用形式）：

        dφ_i/dt = [A_{i+1}J_{i+1} − A_i J_i]/V_i,   J_j = −Γ_j(φ_j−φ_{j−1})/h_j

    返回 (net, Jface)：net 为各控制体的 ∫散度·dV（量纲），Jface 为面通量。
    """
    Gf = face_diffusivity(Gamma_cell)
    N, R0 = grid.N, grid.R0
    rc, h, V, A = grid.rc, grid.h, grid.V, grid.A_face
    phi = np.asarray(phi, dtype=float)

    Jface = np.zeros(N + 1)
    Jface[1:N] = -Gf[1:N] * (phi[1:] - phi[:-1]) / h
    Jface[0] = 0.0
    G_N = float(Gf[N])
    Bi_d = _bi_delta(G_N, h_bc, grid)
    Jface[N] = 0.0 if not np.isfinite(Bi_d) else h_bc * (phi[-1] - phi_inf) / (1.0 + Bi_d)

    net = A[:-1] * Jface[:-1] - A[1:] * Jface[1:]
    return net, Jface
