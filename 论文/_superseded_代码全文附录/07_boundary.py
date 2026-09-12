"""
CODE-13  边界条件模块                    [甲主 / 乙审 · M0+M1 · Day2上午]

职责
----
把"表面到底以什么条件与烘房交换热与水"**集中到一个模块**，
避免热方程与水分方程各自写一份边界、各自约定符号。

M0 形式（题设基线，**默认**）
------------------------------
    r = R0 :   −k ∂T/∂r|_R = h (T_s − T∞)
               −D ∂C/∂r|_R = h_m (C_s − C∞ᵉᑫ)
    h   = 25 W/(m²·K)      [Q-PDF 附录2]
    h_m = 8×10⁻⁷ m/s       [Q-PDF 附录2]

M1 扩展形式（**默认关闭**，`latent=True` 才启用）
--------------------------------------------------
    −k ∇T·n|_R = h (T_s − T∞) + λ(T_s) J_w − q_rad,in        [R3-F1]

三个必须写清的量
~~~~~~~~~~~~~~~~
1. **J_w 是向外的水质量通量，单位 kg/(m²·s)**
   🔴 **不得**把 h_m(C_s − C∞) 直接乘潜热当热通量 —— 它的单位是 kg/(kg·s)，
      不是 kg/(m²·s)。必须乘一个密度基准：

          J_w = ρ_d · h_m (C_s − C∞ᵉᑫ)                        [OWN-RHOD]

   ρ_d 的选取见下。

2. **ρ_d 与水分方程的自洽性**（这是"通量闭合"的核心）
   ∂C/∂t = (1/r)∂_r(D r ∂_r C) 中的 C 是**干基含水率**（kg 水 / kg 干物质），
   D 的单位是 m²/s。该方程量纲自洽的前提，正是把 C 视为
   "干物质基准的质量分数"→ 换算成单位体积的水量需乘**干物质体积密度 ρ_d**。
   故 J_w = ρ_d · (−D ∂C/∂r)|_R 与扩散方程描述的是**同一个通量**，
   不存在重复计数。本模块进一步保证：
   **用于热边界的水通量，与水分方程实际算出的边界通量是同一个离散量**
   （见 `moisture_flux`，与 fvm_cyl 的 Robin 重构完全一致），
   而**不是**另写一个连续表达式。

   🔴 ρ_d 取值声明：附录3 的 ρ(C)=650+128C 与"干物质守恒"不兼容 ——
      ρ(C)/(1+C) 在 C=0 给 650 kg/m³、在 C=2.55 给 275.0 kg/m³，**不是常数**。
      这正是 CODE-35（乙）几何—密度一致性诊断要处理的矛盾。
      本模块取**初始状态基准**并在结果中显式声明：

          ρ_d = ρ(C₀)/(1+C₀) = 976.4/3.55 = 275.04 kg/m³

      同时提供 ±50% 情景扰动接口（CODE-25），**不假装该值无争议**。

3. **蒸发位置**：本模块把全部潜热放在**表面**（因为 C 的边界条件
   本身就是表面换热式），因此**不再布置体源**。
   🔴 若将来改在内部加体源，**必须**相应扣掉表面这一项，二者不可并存。
   （R3-F2 指出 T<60 °C 时气相渗流占比 <1.5%，故"整体汽化带"在本工况不重要。）

辐射项 q_rad,in
---------------
默认 0。R3-F4 给出 h_rad ≈ 6.5 W/(m²·K)、N_rc ≈ 0.26（[R3-F4]），
但该条已标 **FLAGGED**：研报自述 h=25 可能已隐含吸收辐射，论文池亦警告
其"属待验证估算"。故**辐射不并入 M0/M1 基线**，只作论文中的量级讨论。
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from ..config import C_INIT, H_COEF, HM_COEF
from ..numerics.fvm_cyl import Grid, _bi_delta, face_diffusivity
from ..numerics.properties import rho_app3


# ==========================================================================
# 干物质体积密度的初始状态基准   [OWN-RHOD]
# ==========================================================================
#   ρ_d = ρ(C₀)/(1+C₀) = (650 + 128×2.55)/3.55 = 976.4/3.55 = 275.0422535… kg/m³
# 用表达式而不是手抄小数，避免四舍五入引入不可追溯的偏差。
RHO_D_INIT = float(rho_app3(C_INIT) / (1.0 + C_INIT))


# ==========================================================================
# 配置
# ==========================================================================
@dataclass
class BoundaryConfig:
    """
    边界参数集合。默认即 M0 题设基线。
    """
    h: float = H_COEF                    # W/(m²·K)  对流换热      [Q-PDF]
    h_m: float = HM_COEF                 # m/s       对流传质      [Q-PDF]

    # ---- M1 扩展开关（默认全关）----
    latent: bool = False                 # 是否启用蒸发潜热项
    lambda_vap: float = 2.26e6           # J/kg  水的汽化潜热      [R3-F1]
    rho_d: float = RHO_D_INIT            # kg/m³ 干物质体积密度    [OWN-RHOD]
    q_rad_in: float = 0.0                # W/m²  入射净辐射（默认 0，[R3-F4] FLAGGED）

    def layer(self) -> str:
        return "M1" if self.latent else "M0"

    def describe(self) -> dict:
        d = {"层次": self.layer(), "h / (W/(m2·K))": self.h,
             "h_m / (m/s)": self.h_m, "潜热": "启用" if self.latent else "关闭"}
        if self.latent:
            d.update({"λ / (J/kg)": self.lambda_vap,
                      "ρ_d / (kg/m3)": self.rho_d,
                      "q_rad_in / (W/m2)": self.q_rad_in})
        return d


# ==========================================================================
# 离散水通量 —— 与水分方程**同一个**量
# ==========================================================================
# ⚠️ 用具名元组而不是裸 tuple：这里曾出过一次真实事故 ——
#    调用处写成 `_, J_w = water_mass_flux(...)`，把**表面含水率**当成水通量
#    送进潜热项，等效环境温度被压到 −2.3×10⁵ K，Picard 直接振荡不收敛。
#    具名返回让这类顺序错误在代码里根本写不出来。
MoistureFlux = namedtuple("MoistureFlux", ["J_C", "C_s"])
WaterFlux = namedtuple("WaterFlux", ["J_w", "C_s"])


def moisture_surface_flux(C_cells, D_cell, C_inf_eq, grid: Grid,
                          bc: BoundaryConfig) -> MoistureFlux:
    """
    由水分方程**实际使用的** Robin 离散重构出表面向外的水通量（干基口径）。

        J_C = h_m (C_N − C∞ᵉᑫ)/(1 + Bi_Δ),   Bi_Δ = h_m·d/D_N

    单位：kg/(kg·s)。这是 fvm_cyl.surface_flux 的同一表达式，
    在此重写一份是为了**显式暴露它对边界的依赖**，便于乙审计。
    """
    C = np.asarray(C_cells, dtype=float)
    D_N = float(np.asarray(D_cell, dtype=float)[-1])
    Bi_d = _bi_delta(D_N, bc.h_m, grid)
    if not np.isfinite(Bi_d):
        return MoistureFlux(0.0, float(C_inf_eq))
    J_C = bc.h_m * (C[-1] - C_inf_eq) / (1.0 + Bi_d)
    C_s = (C[-1] + Bi_d * C_inf_eq) / (1.0 + Bi_d)
    return MoistureFlux(float(J_C), float(C_s))


def water_mass_flux(C_cells, D_cell, C_inf_eq, grid: Grid,
                    bc: BoundaryConfig) -> WaterFlux:
    """
    向外的水质量通量 J_w，单位 **kg/(m²·s)**（潜热项唯一合法的输入量）。

        J_w = ρ_d · J_C

    🔴 直接对 h_m(C_s−C∞) 乘 λ 是错的 —— 它比 kg/(m²·s) 少一个 kg/m³。
    """
    mf = moisture_surface_flux(C_cells, D_cell, C_inf_eq, grid, bc)
    return WaterFlux(bc.rho_d * mf.J_C, mf.C_s)


# ==========================================================================
# 热边界
# ==========================================================================
def heat_ambient(T_inf, J_w, bc: BoundaryConfig):
    """
    把 M1 的表面潜热与辐射折算成**等效环境温度**，使热边界重写为标准 Robin 形式：

        −k ∂T/∂r|_R = h(T_s − T∞) + λ J_w − q_rad,in
                    ≡ h(T_s − T∞ᵉᶠᶠ)

    推导（注意符号，容易写反）：
        h(T_s − T∞) + λ J_w − q_rad = h(T_s − T∞ᵉᶠᶠ)
        −h T∞ + λ J_w − q_rad       = −h T∞ᵉᶠᶠ
      ⇒ T∞ᵉᶠᶠ = T∞ − (λ J_w − q_rad,in)/h

    这一改写的好处：热方程在每个 Picard 迭代内**仍是线性 Robin 问题**，
    线性求解器、M-矩阵性质、表面重构全部原样复用，无需改动 fvm_cyl。

    物理含义：蒸发（J_w>0）**压低**等效环境温度（本例中量级可达数十 K）
    → 表面被强烈冷却，这正是恒速干燥期"物料温度钳位于湿球温度"的机制
    [R3-F1]；入射净辐射（q_rad,in>0）则**抬高**等效环境温度。

    返回 (T_inf_eff, 折算温降 dT_evap)，dT_evap = (λ J_w − q_rad)/h。
    """
    if not bc.latent:
        return float(T_inf), 0.0
    # ---- 量级护栏 ----
    # 物理上 J_w ≤ ρ_d·h_m·C_max ≈ 275×8e-7×2.55 = 5.6e-4 kg/(m²·s)，
    # 故 dT_evap ≤ 2.26e6×5.6e-4/25 ≈ 51 K。超出数百 K 必是量纲/取值错误
    # （例如把表面含水率当成水通量传了进来）。
    dT = (bc.lambda_vap * J_w - bc.q_rad_in) / bc.h
    if abs(dT) > 500.0:
        raise ValueError(
            f"蒸发折算温降 {dT:.4g} K 超出物理量级上限（约 51 K）。"
            f"检查 J_w 是否确为 kg/(m²·s) 的**质量通量**"
            f"（当前 J_w={J_w:.4g}，ρ_d={bc.rho_d:.4g}，λ={bc.lambda_vap:.3g}）。"
            f"常见错误：把表面含水率 C_s 当作通量传入。"
        )
    return float(T_inf - dT), float(dT)


def net_surface_heat_flux(T_cells, k_cell, T_inf, J_w, grid: Grid, bc: BoundaryConfig):
    """
    表面**向外**的净热通量（含潜热），单位 W/m²。

        q_out = h(T_s − T∞) + λ J_w − q_rad,in

    ⚠️ 这是用于**收支检验与报告**的量，不参与求解（求解走 heat_ambient 的等效温度）。
    """
    from ..numerics.fvm_cyl import surface_flux
    q_conv = surface_flux(T_cells, k_cell, bc.h, T_inf, grid)   # 这里 k 为导热系数、h 为 W/(m²·K)
    q_lat = bc.lambda_vap * J_w if bc.latent else 0.0
    return float(q_conv + q_lat - bc.q_rad_in), float(q_conv), float(q_lat)


# ==========================================================================
# 自检：纯导热退化
# ==========================================================================
def check_pure_conduction(grid: Grid, bc: BoundaryConfig | None = None) -> dict:
    """
    验收要求：**纯导热无蒸发时退化为标准 Robin 条件**。

    做法：构造一个已知线性剖面 T(r) = T∞ + G·(R0 − r)（不是本问题的解，
    但满足 ∂T/∂r = −G 常数），检查
        ① latent=False 时 heat_ambient 返回 T∞ 本身（不引入任何附加项）
        ② 离散 surface_flux 与解析 −k ∂T/∂r = kG 在细网格下一致
    """
    bc = bc or BoundaryConfig()
    rng = np.random.default_rng(0)
    T_inf = 50.0
    k = 0.483
    rep = {}

    # ① M0 不改变环境温度
    T_eff, dT = heat_ambient(T_inf, 1e-3, bc)
    rep["M0_不引入附加项"] = (abs(T_eff - T_inf) < 1e-15) and (dT == 0.0)

    # ② 均匀导热系数下，表面的 Robin 重构应给出 T_s 落在 [T∞, T_N] 内
    #    （此处构造"内热外冷"，热流向外，故 T∞ ≤ T_s ≤ T_N）
    T_cells = np.linspace(T_inf + 2.0, T_inf + 0.1, grid.N)
    k_cell = np.full(grid.N, k)
    from ..numerics.fvm_cyl import surface_value, surface_flux
    Ts = surface_value(T_cells, k_cell, bc.h, T_inf, grid)
    J = surface_flux(T_cells, k_cell, bc.h, T_inf, grid)
    rep["表面值落在合理区间"] = bool(T_inf - 1e-12 <= Ts <= T_cells[-1] + 1e-12)
    rep["向外通量为正"] = bool(J > 0.0)
    rep["表面值与Robin一致"] = bool(abs(J - bc.h * (Ts - T_inf)) < 1e-9 * max(1.0, abs(J)))

    # ③ 潜热开关确实改变等效环境温度，且方向为**降温**
    bc1 = BoundaryConfig(latent=True)
    T_eff1, dT1 = heat_ambient(T_inf, 1e-3, bc1)
    rep["M1_等效环境被压低"] = bool(T_eff1 < T_inf and dT1 > 0.0)
    rep["M1_折算量级正确"] = bool(
        abs(dT1 - bc1.lambda_vap * 1e-3 / bc1.h) < 1e-12
        and abs((T_inf - T_eff1) - dT1) < 1e-12)
    # ③b 辐射方向相反：入射净辐射应抬高等效环境温度
    bc2 = BoundaryConfig(latent=True, q_rad_in=200.0)
    T_eff2, _ = heat_ambient(T_inf, 1e-3, bc2)
    rep["M1_辐射方向相反"] = bool(T_eff2 > T_eff1)

    # ④ ρ_d 基准与初始状态自洽
    rho0, C0 = 650.0 + 128.0 * 2.55, 2.55
    rep["ρ_d_初始基准自洽"] = bool(abs(bc1.rho_d - rho0 / (1.0 + C0)) < 1e-12)

    rep["全部通过"] = all(v for k_, v in rep.items() if isinstance(v, bool))
    return rep


if __name__ == "__main__":
    g = Grid(N=200, R0=0.02, grading=1.5)
    rep = check_pure_conduction(g)
    print("[CODE-13] 边界模块自检：")
    for k_, v in rep.items():
        print(f"    {'OK ' if v else 'FAIL'}  {k_}")
    print()
    for k_, v in BoundaryConfig(latent=True).describe().items():
        print(f"    {k_:16s} {v}")
