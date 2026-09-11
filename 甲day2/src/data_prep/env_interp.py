"""
CODE-02  附件1 环境数据插值            [丙 · M0 · Day1上午]

实测事实（已核实，非转述）
--------------------------
    附件1: 241 行 × 3 列（时间 / 温度 / 水分浓度）
    时间 0 → 14400 s，**步长恒为 60 s**，严格单调递增
    温度   28.0000 → 50.1650 °C   (min 28.0000, max 50.2460)
    水分   0.019630 → 0.049860 kg/kg (min 0.019630, max 0.050250)
    温度差分**变号 116 次**、水分浓度变号 **119 次** —— **波动遍布全程，不只尾部**

两个版本（v3 清单 CODE-02 要求）
--------------------------------
    faithful : PCHIP 保形插值，**保留有效测量**。实测波动是真实边界条件的一部分，
               **不天然属于噪声**，故此为默认版本。
    smoothed : 受控平滑（Savitzky–Golay），仅作**对照**，
               并须在论文中说明平滑依据（传感器精度 / 时间序列特征）。

禁止事项
--------
    * 不得推断控制器类型（不能说"这是 PD 超调"）
    * 不得据此推断烘房结构
    * 禁止高阶多项式（会产生不合理振荡）

变量基准
--------
    附件1 的"水分浓度"列在此按**题设等效约定**读入，记作 C_inf_eq
    （等效环境平衡含水率），**不是**由真实空气状态求出的吸附平衡结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from ..config import ATTACH_DIR


# ==========================================================================
# 读取
# ==========================================================================
@dataclass
class EnvData:
    t: np.ndarray          # s
    T: np.ndarray          # °C
    C: np.ndarray          # kg/kg
    source: str = ""

    def __post_init__(self):
        assert np.all(np.diff(self.t) > 0), "时间列必须严格递增"
        assert len(self.t) == len(self.T) == len(self.C), "列长度不一致"

    @property
    def t_span(self):
        return float(self.t[0]), float(self.t[-1])

    def summary(self) -> str:
        return (
            f"附件1: {len(self.t)} 点, t ∈ [{self.t[0]:.0f}, {self.t[-1]:.0f}] s, "
            f"步长 {self.t[1]-self.t[0]:.0f} s\n"
            f"  温度 {self.T.min():.4f} → {self.T.max():.4f} °C\n"
            f"  水分 {self.C.min():.6f} → {self.C.max():.6f} kg/kg\n"
            f"  温度差分变号 {(np.diff(np.sign(np.diff(self.T))) != 0).sum()} 次, "
            f"水分差分变号 {(np.diff(np.sign(np.diff(self.C))) != 0).sum()} 次"
        )


def load_env(path: Path | None = None) -> EnvData:
    p = Path(path) if path else (ATTACH_DIR / "附件1.xlsx")
    df = pd.read_excel(p)
    cols = {c.strip(): c for c in df.columns}
    return EnvData(
        t=df[cols["时间"]].to_numpy(dtype=float),
        T=df[cols["温度"]].to_numpy(dtype=float),
        C=df[cols["水分浓度"]].to_numpy(dtype=float),
        source=str(p),
    )


# ==========================================================================
# 插值器
# ==========================================================================
class EnvInterpolator:
    """
    环境边界函数封装。

    两个版本通过 `mode` 选择：
        "faithful" —— PCHIP，过所有原始点（默认）
        "smoothed" —— 先 Savitzky–Golay 平滑再做 PCHIP（**仅作对照**）
    """

    def __init__(self, data: EnvData, mode: str = "faithful",
                 smooth_window: int = 21, smooth_poly: int = 2,
                 smooth_fraction: float = 0.0):
        assert mode in ("faithful", "smoothed")
        self.mode = mode
        self.data = data
        T, C = data.T.copy(), data.C.copy()
        if mode == "smoothed":
            w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            w = min(w, len(T) - (1 - len(T) % 2))
            poly = min(smooth_poly, w - 1)
            T = savgol_filter(T, w, poly)
            C = savgol_filter(C, w, poly)
            self.smooth_window = w
        self._T = PchipInterpolator(data.t, T, extrapolate=False)
        self._C = PchipInterpolator(data.t, C, extrapolate=False)
        self.t_min, self.t_max = data.t[0], data.t[-1]

    # ------------------------------------------------------------------
    def T_inf(self, t):
        v = self._T(t)
        if np.any(~np.isfinite(v)):
            raise ValueError(
                f"T_inf({t}) 超出附件1 覆盖范围 [{self.t_min:.0f}, {self.t_max:.0f}] s。"
                f"恒温段外推方案须显式声明（见框架 §5.3）。"
            )
        return v

    def C_inf(self, t):
        v = self._C(t)
        if np.any(~np.isfinite(v)):
            raise ValueError(
                f"C_inf({t}) 超出附件1 覆盖范围 [{self.t_min:.0f}, {self.t_max:.0f}] s。"
            )
        return v

    # ------------------------------------------------------------------
    def plateau_stats(self, t_from: float = 9600.0) -> dict:
        """平台段统计（用于恒温干燥段边界设定，框架 §5.3 方案A）。"""
        m = self.data.t >= t_from
        return {
            "t_from": t_from,
            "n": int(m.sum()),
            "T_mean": float(self.data.T[m].mean()),
            "T_std": float(self.data.T[m].std()),
            "C_mean": float(self.data.C[m].mean()),
            "C_std": float(self.data.C[m].std()),
        }

    def __repr__(self):
        return f"<EnvInterpolator mode={self.mode}  t∈[{self.t_min:.0f},{self.t_max:.0f}]s>"


if __name__ == "__main__":
    d = load_env()
    print(d.summary())
    for mode in ("faithful", "smoothed"):
        it = EnvInterpolator(d, mode=mode)
        print(f"\n{mode}: {it}")
        for tt in (0, 300, 600, 900, 1200, 1500, 1800):
            print(f"   t={tt:5d}s   T∞={float(it.T_inf(tt)):8.4f} °C   "
                  f"C∞={float(it.C_inf(tt)):9.6f} kg/kg")
    print("\n平台段统计:", EnvInterpolator(d).plateau_stats())
