"""
问题4  考虑尺寸变化（收缩）的烘干时长            [甲 · M0 · 接替乙的问题4]

与问题2/3 的唯一区别
--------------------
药材半径随失水收缩：$R = R(t)$，由**附件2** 给出（145 点，0—259200 s，2.000→1.198 cm）。

建模：材料坐标（material coordinate）
-------------------------------------
令 $\\xi = r/R(t) \\in [0,1]$ —— 同一个 $\\xi$ 永远对应同一块物料。

在**均匀仿射收缩**假设下（$r = R(t)\\,\\xi$），材料导数满足

    ρ(C)c_p(C) ∂T/∂t|_ξ = (1/(R²ξ)) ∂_ξ( k ξ ∂_ξ T )
    ∂C/∂t|_ξ            = (1/(R²ξ)) ∂_ξ( D ξ ∂_ξ C )

🔴 **方程里没有 Ṙ 对流项。** 这不是漏写 —— 材料导数 $\\partial_t|_\\xi$
本身已经把材料运动吸收进去了（分工 v3 §四 明确要求这一形式）。

🔴🔴 网格约定：**节点式**，与乙的 `problem4_drying.m` **逐位一致**
------------------------------------------------------------
本模块最初用了"单元式"约定（$N$ 个单元、界面落在 $\\xi=1$），
与乙的"节点式"（$N$ 个节点、末节点落在 $\\xi=1$）在**均匀网格上就不等价**：
最外控制体宽度差 **1.9 倍**（0.0500 vs 0.0263），
对边界层主导的问题这会显著改变 $t_{dry}$，且**无法与已验证的参照对照**。

现改为节点式，并显式验证：**均匀 $N$=20、$\\gamma$=1.0 时必须复现乙的 73.033 h**
（见 `scripts/check_p4_vs_yi.py`）。复现不了就不往下走。

约定细节（$\\gamma=1$ 时逐项退化为乙的写法）
-----------------------------------------
    节点     ξ_i = 1−(1−ζ_i)^γ,  ζ_i = i/(N−1),  i = 0…N−1      ξ_{N−1} = 1
    内部面   f_j = (ξ_{j−1}+ξ_j)/2,  j = 1…N−1     均匀时 = (j−0.5)Δξ   ← 乙的 `xf`
    中心距   Δξ_j = ξ_j − ξ_{j−1}                   均匀时 = Δξ        ← 乙的 `dxi`
    体积权   w_i = ∫_{左界面}^{右界面} ξ dξ
             两端为**半控制体**：w_0 = f_1²/2，w_{N−1} = (1−f_{N−1}²)/2

离散（与乙同形）
----------------
    w_i dφ_i/dt = G_{i+1/2}(φ_{i+1}−φ_i) − G_{i−1/2}(φ_i−φ_{i−1})
                  − (h/R)(φ_i − φ_∞)

    G_j = f_j · Γ_j /(R² Δξ_j)          Γ_j 取相邻节点的**调和平均**
    表面项系数 = h/R                     ← 导热/扩散是 1/R²，表面是 1/R，**二者不可合并**

量纲核对：除以 $2\\pi L R^2$ 后，导热项得 $1/R^2$、表面项得 $1/R$，
即 $\\mathrm{Bi}\\propto R$，与固定域一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import H_COEF, HM_COEF, NUMERICS, T_INIT, C_INIT
from ..numerics.properties import D_app4, cp_app4, k_app4, rho_app4


# ==========================================================================
@dataclass
class Radius:
    """附件2 的半径数据 + PCHIP 插值（内部用米）。"""

    t: np.ndarray
    R_cm: np.ndarray

    def __post_init__(self):
        from scipy.interpolate import PchipInterpolator
        self._p = PchipInterpolator(self.t, self.R_cm)
        self.t_min, self.t_max = float(self.t[0]), float(self.t[-1])
        self.R0 = float(self.R_cm[0]) / 100.0
        self.R_end = float(self.R_cm[-1]) / 100.0

    def __call__(self, t):
        tt = min(max(float(t), self.t_min), self.t_max)
        return float(self._p(tt)) / 100.0

    def summary(self) -> str:
        return (f"附件2 半径：{len(self.t)} 点，t∈[{self.t_min:.0f},{self.t_max:.0f}] s，"
                f"R: {self.R_cm[0]:.3f} → {self.R_cm[-1]:.3f} cm，"
                f"收缩 {100*(1-self.R_cm[-1]/self.R_cm[0]):.1f}%")


def load_radius(path=None) -> Radius:
    from ..config import ATTACH_DIR
    p = path or (ATTACH_DIR / "附件2.xlsx")
    df = pd.read_excel(p)
    cols = {c.strip(): c for c in df.columns}
    return Radius(t=df[cols["时间"]].to_numpy(float),
                  R_cm=df[cols["半径"]].to_numpy(float))


# ==========================================================================
@dataclass
class GridXiN:
    """
    材料坐标 ξ∈[0,1] 上的**节点式**网格（与乙同约定）。

    grading = 1.0 时**逐位退化**为乙的均匀网格；>1 为向表面 ξ=1 加密。
    """
    N: int = 20
    grading: float = 1.0

    def __post_init__(self):
        z = np.arange(self.N) / (self.N - 1)
        self.xi = 1.0 - (1.0 - z) ** self.grading        # N 个节点，末节点 = 1
        self.f = 0.5 * (self.xi[:-1] + self.xi[1:])      # N-1 个内部面
        self.dx = np.diff(self.xi)                       # N-1 个中心距
        w = np.empty(self.N)
        w[0] = 0.5 * self.f[0] ** 2                      # 半控制体
        w[-1] = 0.5 * (1.0 - self.f[-1] ** 2)            # 半控制体
        w[1:-1] = 0.5 * (self.f[1:] ** 2 - self.f[:-1] ** 2)
        self.w = w

    def summary(self):
        return (f"节点式 ξ 网格 N={self.N}, γ={self.grading}, "
                f"Δξ={self.dx.min():.2e}~{self.dx.max():.2e}, "
                f"最外控制体宽={1.0-self.f[-1]:.2e}")


def _harm(a, b):
    s = a + b
    out = np.zeros_like(s, dtype=float)
    m = s > 0
    out[m] = 2.0 * a[m] * b[m] / s[m]
    return out


# ==========================================================================
def assemble(g: GridXiN, Gam, h_over_R, R, src_phi_inf=None, cap=None):
    """
    组装节点式三对角算子（**不含时间项**）。

    Gam        : (N,) 节点扩散系数（k 或 D）
    h_over_R   : 表面传递系数 / R 的**已除好**的值（传热 h/R，传质 h_m/R）
    R          : 当前半径（用于 1/R² 因子）
    src_phi_inf: 表面源项 = h_over_R * φ_∞；None → 0
    cap        : (N,) 时间项系数（ρc_p）；None → 1

    返回 (dl, dd, du, b, NV)：
        dd·φ_new + dl·φ_{i-1} + du·φ_{i+1} = b + NV/dt·φ_old
    """
    N = g.N
    if cap is None:
        cap = np.ones(N)

    Gf = _harm(Gam[:-1], Gam[1:])                        # N-1 内部面（调和平均）
    G = g.f * Gf / (R ** 2 * g.dx)                       # 长度 N-1（与乙的 G 同形）

    dd = np.zeros(N)
    dd[:-1] += G
    dd[1:] += G
    dl = np.zeros(N)
    dl[1:] = -G
    du = np.zeros(N)
    du[:-1] = -G

    dd[-1] += h_over_R                                   # 表面（节点式：末节点在 ξ=1）
    b = np.zeros(N)
    if src_phi_inf is not None:
        b[-1] = src_phi_inf

    return dl, dd, du, b, (g.w * cap).astype(float)


# ==========================================================================
def solve_p4(env, rad: Radius, grid: GridXiN, t_end: float,
             t_output: np.ndarray, theta: float = 1.0, num=None,
             R_mode: str = "pchip", stop_below: float | None = None,
             R_override: float | None = None, verbose: bool = False):
    """
    求解问题4。返回 dict：times, T, C（(n_t, N)）、C_max 轨迹、步数等。

    R_override: 给定则半径恒定（用于"固定半径"退化检验）。
    R_mode    : "pchip" / "const_end" / "linear"（CODE-05 半径外推敏感性）
    """
    num = num or NUMERICS
    N = grid.N

    def R_of(t):
        if R_override is not None:
            return float(R_override)
        if t <= rad.t_max or R_mode == "pchip":
            return rad(t)
        if R_mode == "const_end":
            return rad.R_end
        if R_mode == "linear":
            slope = (rad.R_cm[-1] - rad.R_cm[-2]) / (rad.t[-1] - rad.t[-2]) / 100.0
            return rad.R_end + (t - rad.t_max) * slope
        return rad(t)

    T = np.full(N, T_INIT)
    C = np.full(N, C_INIT)
    t = 0.0
    dt = num.dt_init
    t_out = np.asarray(t_output, float)
    k_out = 0
    Cmin = np.inf
    times, Ts, Cs, Cmax = [], [], [], []

    while t < t_end - 1e-12:
        dt = min(dt, t_end - t)
        if k_out < len(t_out):
            dt = min(dt, max(t_out[k_out] - t, 1e-9))
        Tn, Cn, it = _step(grid, T, C, t, dt, env, R_of, num)
        if not (np.all(np.isfinite(Tn)) and np.all(np.isfinite(Cn))):
            dt *= 0.5
            if dt < num.dt_min:
                raise RuntimeError(f"t={t:.4g} 步长触底")
            continue
        t += dt
        T, C = Tn, Cn
        Cmin = min(Cmin, float(C.min()))
        dt = min(dt * 1.25, num.dt_max)

        if stop_below is not None and float(C.max()) < stop_below:
            # 先把落在本步内的输出时刻补齐
            while k_out < len(t_out) and t >= t_out[k_out] - 1e-9:
                times.append(t); Ts.append(T.copy()); Cs.append(C.copy())
                Cmax.append(float(C.max())); k_out += 1
            # 🔴 再把**达标时刻本身**追加为末点。
            #    阈值跨越通常落在两个输出时刻之间（本例 60 s 一格），
            #    若只靠输出网格上的采样，最后一个已记录点的 C_max 仍 > 0.15，
            #    下游按"C_max<0.15"找达标点就会**找不到**，误判为"6 天未达标"
            #    （实测踩过：末记录点 262860 s 的 C_max=0.1500159，
            #      而真正的跨越在 262910 s）。
            #    这与 result3 的"末行为达标时刻（不必落在网格上）"是同一处理。
            times.append(t); Ts.append(T.copy()); Cs.append(C.copy())
            Cmax.append(float(C.max()))
            break

        while k_out < len(t_out) and t >= t_out[k_out] - 1e-9:
            times.append(t); Ts.append(T.copy()); Cs.append(C.copy())
            Cmax.append(float(C.max())); k_out += 1
        if verbose and len(times) and len(times) % 200 == 0:
            print(f"      t={t:9.1f}s Cmax={C.max():.5f} dt={dt:.3g}", flush=True)

    return {"times": np.array(times), "T": np.array(Ts), "C": np.array(Cs),
            "C_max": np.array(Cmax), "C_min": Cmin, "t_final": t,
            "grid": grid.summary(), "R_mode": R_mode,
            "R_override": R_override, "n_out": len(times)}


def _step(g: GridXiN, T, C, t, dt, env, R_of, num):
    """一个 Backward Euler 步（块 Gauss-Seidel Picard，T/C 联立）。"""
    N = g.N
    Tn, Cn = T.copy(), C.copy()
    t_new = t + dt
    R = R_of(t_new)
    T_inf, C_inf = float(env.T_inf(t_new)), float(env.C_inf(t_new))
    it = 0

    for it in range(1, num.picard_max_iter + 1):
        T_old, C_old = Tn.copy(), Cn.copy()

        # ---------- 水分 ----------
        Dc = np.atleast_1d(np.asarray(D_app4(Cn, Tn), float))
        dl, dd, du, b, NV = assemble(g, Dc, HM_COEF / R, R,
                                     src_phi_inf=HM_COEF / R * C_inf)
        Cn_new = _thomas(dl, dd + NV / dt, du, b + NV / dt * C)

        # ---------- 温度 ----------
        kc = np.atleast_1d(np.asarray(k_app4(Cn_new), float))
        cap = np.atleast_1d(np.asarray(rho_app4(Cn_new) * cp_app4(Cn_new), float))
        dl2, dd2, du2, b2, NV2 = assemble(g, kc, H_COEF / R, R, cap=cap,
                                          src_phi_inf=H_COEF / R * T_inf)
        Tn = _thomas(dl2, dd2 + NV2 / dt, du2, b2 + NV2 / dt * T)
        Cn = Cn_new

        du_max = max(np.max(np.abs(Tn - T_old)), np.max(np.abs(Cn - C_old)))
        if du_max < num.picard_atol_u + num.picard_rtol_u * max(
                1.0, float(np.max(np.abs(Tn))), float(np.max(np.abs(Cn)))):
            break
    return Tn, Cn, it


def _thomas(dl, dd, du, b):
    """三对角求解（与乙的 thomas.m 同一算法）。"""
    n = len(b)
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = du[0] / dd[0]; dp[0] = b[0] / dd[0]
    for i in range(1, n):
        m = dd[i] - dl[i] * cp[i - 1]
        if i < n - 1:
            cp[i] = du[i] / m
        dp[i] = (b[i] - dl[i] * dp[i - 1]) / m
    x = np.zeros(n); x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
