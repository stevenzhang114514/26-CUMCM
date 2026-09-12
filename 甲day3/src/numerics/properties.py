"""
CODE-07  物性求值模块                  [甲主 / 乙审 · M0 · Day1上午]

职责
----
统一提供三套物性公式（附录2 / 附录3 / 附录4）的求值接口。

约定
----
* rho  : 湿物料体积密度 —— **只用于热容项 rho*cp**
* rho_d: 干固体密度 rho/(1+C) —— **只用于几何—密度一致性诊断（CODE-35）**
         二者定义不同，不允许互换（见 v3 清单 §3.1 CODE-07）
* T 传入的是**摄氏度**，本模块内部完成 T_K = T + 273.15（唯一转换点之二，
  与 config.to_kelvin 同源）
* C 是**场变量**（同一时刻同一空间点的含水率），不是全场平均、不是初值

数值处理
--------
D 随 C 指数变化，在 C→0 时下溢。求值在 log 空间完成并做下截断，
保证返回有限值（0 或极小正数），不产生 NaN/Inf。
"""

from __future__ import annotations

import numpy as np

from ..config import to_kelvin

# D 的下截断：低于此值视为 0（物理上已完全干燥）
_D_FLOOR = 1.0e-300


# ==========================================================================
# 附录2 —— 问题1（常数物性 + D 只依赖 C）
# ==========================================================================
def D_app2(C):
    """
    附录2 水分扩散系数  D = 7e-9 * exp(-0.89/C)   [m2/s]

    注意：本式**不含温度**，这正是问题1 中水分场与温度场解耦的结构性原因。
    """
    C = np.asarray(C, dtype=float)
    out = np.zeros_like(C)
    m = C > 0
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        out[m] = 7.0e-9 * np.exp(-0.89 / C[m])
    out[~np.isfinite(out)] = 0.0
    out[out < _D_FLOOR] = 0.0
    return out if out.ndim else float(out)


# ==========================================================================
# 附录3 —— 问题2、问题3（C 与 T 的函数）
# ==========================================================================
def _log_D_app3(C, T_K):
    """log(D)，避免中间下溢。"""
    C = np.maximum(np.asarray(C, dtype=float), 1e-12)
    T_K = np.maximum(np.asarray(T_K, dtype=float), 1.0)
    return np.log(2.4e-3) - 0.45 / C - 3850.0 / T_K


def D_app3(C, T_celsius):
    """
    附录3  D = 2.4e-3 * exp(-0.45/C) * exp(-3850/T_K)   [m2/s]

    T_celsius 传入摄氏度，内部转 K。
    """
    T_K = to_kelvin(np.asarray(T_celsius, dtype=float))
    ld = _log_D_app3(C, T_K)
    out = np.exp(np.clip(ld, -700.0, 700.0))
    out[ld < np.log(_D_FLOOR)] = 0.0
    out[~np.isfinite(out)] = 0.0
    return out


def rho_app3(C):
    """附录3 湿物料体积密度  rho = 650 + 128C   [kg/m3]"""
    return 650.0 + 128.0 * np.asarray(C, dtype=float)


def cp_app3(C):
    """附录3 比热容  cp = 1450 + 2736*C/(1+C)   [J/(kg·K)]"""
    C = np.asarray(C, dtype=float)
    return 1450.0 + 2736.0 * C / (1.0 + C)


def k_app3(C):
    """附录3 导热系数  k = 0.21 + 0.38*C/(1+C)   [W/(m·K)]"""
    C = np.asarray(C, dtype=float)
    return 0.21 + 0.38 * C / (1.0 + C)


# ==========================================================================
# 附录4 —— 问题4（收缩）
# ==========================================================================
def _log_D_app4(C, T_K):
    C = np.maximum(np.asarray(C, dtype=float), 1e-12)
    T_K = np.maximum(np.asarray(T_K, dtype=float), 1.0)
    return np.log(4.2e-4) - 0.30 / C - 3850.0 / T_K


def D_app4(C, T_celsius):
    """附录4  D = 4.2e-4 * exp(-0.30/C) * exp(-3850/T_K)   [m2/s]"""
    T_K = to_kelvin(np.asarray(T_celsius, dtype=float))
    ld = _log_D_app4(C, T_K)
    out = np.exp(np.clip(ld, -700.0, 700.0))
    out[ld < np.log(_D_FLOOR)] = 0.0
    out[~np.isfinite(out)] = 0.0
    return out


def rho_app4(C):
    """附录4 湿物料体积密度  rho = 760 + 90C   [kg/m3]"""
    return 760.0 + 90.0 * np.asarray(C, dtype=float)


def cp_app4(C):
    """附录4 比热容  cp = 1850 + 2150*C/(1+C)   [J/(kg·K)]"""
    C = np.asarray(C, dtype=float)
    return 1850.0 + 2150.0 * C / (1.0 + C)


def k_app4(C):
    """附录4 导热系数  k = 0.12 + 0.20*C/(1+C)   [W/(m·K)]"""
    C = np.asarray(C, dtype=float)
    return 0.12 + 0.20 * C / (1.0 + C)


# ==========================================================================
# 干固体密度（**仅用于 CODE-35 几何—密度一致性诊断**）
# ==========================================================================
def rho_dry(rho, C):
    """
    rho_d = rho / (1 + C)

    前提：附录给出的 rho 是**湿物料体积密度**。
    若 rho 实为热学等效参数，则不得用于严格干质量检验（v3 清单 §3.1 CODE-35）。
    """
    return np.asarray(rho, dtype=float) / (1.0 + np.asarray(C, dtype=float))


# ==========================================================================
# 物性诊断：打印关键值供人工复核
# ==========================================================================
def describe():
    """打印三套物性在状态区间端点上的取值。"""
    import math
    Cs = [0.05, 0.15, 0.5, 1.0, 2.55]
    print("  C        D_app2        D_app3@323K   D_app4@323K    rho3    cp3     k3")
    for C in Cs:
        print(f"  {C:5.2f}  {D_app2(C):12.4e}  {D_app3(C,50):12.4e}"
              f"  {D_app4(C,50):12.4e}  {rho_app3(C):6.1f} {cp_app3(C):7.1f} {k_app3(C):6.4f}")
    print(f"\n  D_app3 跨 C∈[0.15,2.55] 的量级比 = "
          f"{D_app3(2.55,50)/max(D_app3(0.15,50),1e-300):.1f} 倍")


if __name__ == "__main__":
    describe()
