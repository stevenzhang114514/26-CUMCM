"""
CODE-01  全局配置中心与单位约定          [甲 · M0 · Day1上午]

职责
----
1. 冻结全局单位约定（时间 s / 长度 m / 温度内部用 °C，D 公式内转 K）
2. 集中存放几何、物性、初值、边界系数、数值参数、输出规格
3. 提供四档扩散系数关键值自检（check_units），任一不符即报错

单位约定（全局唯一，禁止在别处再转换）
--------------------------------------
    T   : 摄氏度 °C   —— 场变量、初值、边界
    T_K : 开尔文 K    —— 仅用于附录3/附录4 的 D 公式，由 to_kelvin() 完成
    r   : 米 m
    t   : 秒 s
    C   : kg/kg（干基含水率）

变量基准（CODE-34 入口检查第 1 项结论）
---------------------------------------
    C        = m_w / m_d     药材干基含水率
    Y        = m_v / m_da    空气含湿量
    C_inf_eq = 已明确基准的"等效环境平衡含水率"，即附件1 的水分浓度列。
               本文按题设等效约定直接将其作为药材表面的等效平衡含水率，
               不假装它是由真实空气状态求出的吸附平衡结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]        # .../26-CUMCM/甲day1
ATTACH_DIR = PROJECT_ROOT.parent / "A题题目和附件" / "附件"
FIG_DIR = PROJECT_ROOT / "figs"
RESULT_DIR = PROJECT_ROOT / "results" / "M0"
DOC_DIR = PROJECT_ROOT / "docs"

# --------------------------------------------------------------------------
# 几何（题面：圆柱，长 25 cm，半径 2 cm）
# --------------------------------------------------------------------------
R0 = 0.02          # 初始半径 m
L0 = 0.25          # 长度 m
L_HALF = L0 / 2.0  # 半长 m —— 端面尺度分析必须用半长，不得用全长


def to_kelvin(T_celsius):
    """摄氏度 → 开尔文。全局唯一转换点。"""
    return T_celsius + 273.15


# --------------------------------------------------------------------------
# 初值
# --------------------------------------------------------------------------
T_INIT = 28.0      # °C
C_INIT = 2.55      # kg/kg（干基）


# --------------------------------------------------------------------------
# 物性（三套，按问题取用）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ConstProps:
    """附录2 —— 问题1 用（常数）"""
    rho: float = 820.0       # kg/m3
    cp: float = 2600.0       # J/(kg·K)
    k: float = 0.36          # W/(m·K)

    @property
    def alpha(self) -> float:
        """热扩散率 m2/s"""
        return self.k / (self.rho * self.cp)


PROPS_APP2 = ConstProps()

# --------------------------------------------------------------------------
# 边界传递系数（题设）
# --------------------------------------------------------------------------
H_COEF = 25.0       # 对流换热系数 W/(m2·K)
HM_COEF = 8.0e-7    # 对流传质系数 m/s

# --------------------------------------------------------------------------
# 数值参数
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Numerics:
    # ---- 网格 ----
    # 生产网格：N=200 且 grading=1.5（向表面加密的渐变网格）。
    # 依据（见 docs/Day1_结论.md §3）：水分在表面形成极薄边界层
    # （t=1 s 时厚度仅 √(Dt) ≈ 0.07 mm），均匀网格即便 N=640 仍欠分辨；
    # 而中心区剖面平坦无需加密。渐变网格 N=200/γ=1.5 以**约一半的计算量**
    # 达到优于均匀 N=640 的精度（内部各列偏差 < 4 位小数舍入量子 5e-5）。
    N: int = 200
    grading: float = 1.5        # 1.0 = 均匀；>1 = 向 r=R0 加密
    theta: float = 1.0          # 时间格式：1.0 = Backward Euler（L-稳定）；0.5 = Crank-Nicolson
    dt_init: float = 0.5        # 初始步长 s
    dt_min: float = 1.0e-4      # 步长下限，触底即报错（不静默）
    dt_max: float = 30.0        # 步长上限 s
    safety: float = 0.9         # 误差控制安全因子
    grow_max: float = 2.0       # 单步最大放大倍数
    shrink_max: float = 0.2     # 单步最小收缩倍数
    # 误差容限（步长加倍法估计）
    atol_T: float = 1.0e-5
    rtol_T: float = 1.0e-7
    atol_C: float = 1.0e-8
    rtol_C: float = 1.0e-7
    # 非线性迭代（Picard）
    picard_max_iter: int = 50
    picard_rtol_u: float = 1.0e-8
    picard_atol_u: float = 1.0e-10
    picard_rtol_r: float = 1.0e-8
    picard_atol_r: float = 1.0e-10
    max_step_halving: int = 8   # 单步最多折半次数（失败则报错）


NUMERICS = Numerics()

# --------------------------------------------------------------------------
# 输出规格（严格按附件3 模板实测结果）
# --------------------------------------------------------------------------
T_START = 1         # result1/2 的 A 列从 1 起（不是 0）
T_END = 1800        # 问题1 时间上限 s
T_TARGET = 0.15      # 问题3/4 的达标阈值 kg/kg（题面）
R_OUT_CM = tuple(round(0.1 * i, 1) for i in range(21))   # 0.0, 0.1, ..., 2.0 cm
N_DECIMALS = 4      # 所有结果保留 4 位小数
SHEET_T = "温度"
SHEET_C = "水分浓度"
HEADER_A1 = "时间\\到药材中心的距离"      # 含反斜杠，已与附件3 模板逐字符比对一致


# --------------------------------------------------------------------------
# 四档扩散系数关键值自检
# --------------------------------------------------------------------------
# 附录2: D = 7e-9 * exp(-0.89/C)
# 附录3: D = 2.4e-3 * exp(-0.45/C) * exp(-3850/T_K)
D_CHECKPOINTS = [
    # (标签, 计算函数, 目标值, 相对容限)
    ("附录2 @ C=0.05", lambda: 7.0e-9 * math.exp(-0.89 / 0.05), 1.3021e-16, 5e-3),
    ("附录2 @ C=2.55", lambda: 7.0e-9 * math.exp(-0.89 / 2.55), 4.94e-9, 5e-3),
    ("附录3 @ C=2.55, T_K=300",
     lambda: 2.4e-3 * math.exp(-0.45 / 2.55) * math.exp(-3850.0 / 300.0),
     5.3718662e-9, 1e-5),
    ("附录3 @ C=0.15, T_K=323",
     lambda: 2.4e-3 * math.exp(-0.45 / 0.15) * math.exp(-3850.0 / 323.0),
     7.9570613e-10, 1e-5),
]


def check_units(verbose: bool = True) -> dict:
    """
    四档 D 关键值自检。任一不符即抛错。
    这是防止"T 误用摄氏度""公式系数抄错"的看门人。
    """
    report = {}
    bad = []
    for label, fn, target, rtol in D_CHECKPOINTS:
        got = fn()
        rel = abs(got - target) / abs(target)
        ok = rel <= rtol
        report[label] = {
            "computed": got, "target": target, "rel_err": rel, "ok": ok,
            "to_kelvin_used": "附录3" in label,
        }
        if not ok:
            bad.append(f"  {label}: 计算 {got:.6e} vs 期望 {target:.6e} (相对误差 {rel:.2e})")
    if bad:
        raise AssertionError(
            "CODE-01 单位/公式自检失败，请检查物性公式与温度单位：\n" + "\n".join(bad)
        )
    if verbose:
        print("[CODE-01] 单位与物性自检通过：")
        for label, d in report.items():
            print(f"    {label:28s} = {d['computed']:.6e}  (期望 {d['target']:.6e})")
    return report


def derived_numbers() -> dict:
    """一些供论文引用的派生量。"""
    a = PROPS_APP2.alpha
    return {
        "alpha": a,
        "Bi_T": H_COEF * R0 / PROPS_APP2.k,
        "tau_heat_s": R0 ** 2 / a,
        "explicit_dt_limit_s": (R0 / NUMERICS.N) ** 2 / (4.0 * a),
        "dr_m": R0 / NUMERICS.N,
        "end_face_scale_ratio": (L_HALF / R0) ** 2,
    }


if __name__ == "__main__":
    check_units()
    for k_, v in derived_numbers().items():
        print(f"  {k_:24s} = {v:.6g}")
