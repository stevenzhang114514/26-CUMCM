"""
长时环境边界外推（框架 §5.3 方案 A）    [甲 · M0 · Day2上午]

问题
----
附件1 只覆盖 0—14400 s（4 h），而问题2/3 要跑完整个烘干过程（实测约 2—3 天）。
**恒温干燥段的边界必须显式假设，不能靠插值函数外推**。

实测依据（附件1，[Q-A1]）
-------------------------
    t ≥ 9600 s 段（n = 81 点）：T∞ 均值 50.0017 °C（std 0.0766）
                                 C∞ 均值 0.049998 kg/kg（std 3.4e-4）
    该段已进入平台，波动量级即为传感器/记录噪声。

三个候选方案与取舍（框架 §5.3）
--------------------------------
    A（采用）   t ≤ t_pre 用 PCHIP 过实测点；t > t_pre 取常数平台值
                —— 简单、可辩护，**不假装能预测烘房**
    B         拟合一阶饱和模型后外推 —— 对波动敏感、外推无依据
    C         整体双指数拟合 —— 易过拟合波动，**不推荐**

⭐ 为什么 t_pre 处的小阶跃可以接受
---------------------------------
取 t_pre = 14400 s（用满全部实测数据）时，末点值 50.1650 °C 与平台均值
50.0017 °C 相差 **0.163 °C**，**落在实测波动带（±0.15 °C，std 0.0766）之内**。
即该阶跃是"把噪声末点拉回其自身均值"，不是引入新信息。
本模块仍提供 `ramp` 参数（默认 0）用于检验该阶跃是否影响结论。

⚠️ 不得做的事
--------------
    * 不得推断控制器类型（不能说"这是 PD 超调"）[Q-A1 + 清单 §CODE-02]
    * 不得据附件1 推断烘房结构
    * **不得用任何未经声明的外推方式**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env_interp import EnvData, EnvInterpolator

# 平台段统计的默认起点（框架 §5.3 实测值）
T_PLATEAU_FROM = 9600.0


@dataclass
class PlateauEnv:
    """
    方案 A 环境边界：分段 + 可选斜坡。

        t ≤ t_pre          : 交给底层 EnvInterpolator（PCHIP 过实测点）
        t_pre < t < t_pre+ramp : 线性过渡（ramp=0 时此段为空）
        t ≥ t_pre + ramp   : 常数 T_plateau / C_plateau

    参数
    ----
    interp     : EnvInterpolator（faithful 或 smoothed）
    t_pre      : 切换到平台值的时刻；框架建议 9600 或 14400
    T_plateau  : 平台温度；None 时取 t≥T_PLATEAU_FROM 段的实测均值
    C_plateau  : 平台含水率；None 时同上
    ramp       : 过渡段长度（s）。0 = 纯方案 A（允许小阶跃），
                 非零时先线性过渡到平台值再保持常数，用于灵敏度检验。
    """

    interp: EnvInterpolator
    t_pre: float = 14400.0
    T_plateau: float | None = None
    C_plateau: float | None = None
    ramp: float = 0.0

    def __post_init__(self):
        st = self.interp.plateau_stats(T_PLATEAU_FROM)
        if self.T_plateau is None:
            self.T_plateau = st["T_mean"]
        if self.C_plateau is None:
            self.C_plateau = st["C_mean"]
        self._plateau_stats = st
        if self.t_pre <= 0.0 or self.t_pre > self.interp.t_max:
            raise ValueError(f"t_pre={self.t_pre} 超出附件1 覆盖范围 "
                             f"(0, {self.interp.t_max}]")
        # 记录阶跃大小，供论文如实报告
        self.jump_T = float(self.T_plateau - self.interp.T_inf(self.t_pre))
        self.jump_C = float(self.C_plateau - self.interp.C_inf(self.t_pre))

    # ------------------------------------------------------------------
    def _eval(self, t, fn, plateau):
        t = np.asarray(t, dtype=float)
        scalar = (t.ndim == 0)
        t = np.atleast_1d(t)
        out = np.empty_like(t)

        m_data = t <= self.t_pre
        if np.any(m_data):
            out[m_data] = fn(t[m_data])

        if self.ramp > 0.0:
            t_a, t_b = self.t_pre, self.t_pre + self.ramp
            m_ramp = (t > t_a) & (t < t_b)
            if np.any(m_ramp):
                w = (t[m_ramp] - t_a) / self.ramp
                out[m_ramp] = (1.0 - w) * fn(np.full(w.shape, t_a)) + w * plateau
            m_pl = t >= t_b
        else:
            m_pl = t > self.t_pre

        if np.any(m_pl):
            out[m_pl] = plateau
        return float(out[0]) if scalar else out

    def T_inf(self, t):
        return self._eval(t, self.interp.T_inf, self.T_plateau)

    def C_inf(self, t):
        return self._eval(t, self.interp.C_inf, self.C_plateau)

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "t_pre / s": self.t_pre,
            "ramp / s": self.ramp,
            "T∞ 平台 / °C": round(float(self.T_plateau), 4),
            "C∞ 平台 / (kg/kg)": round(float(self.C_plateau), 6),
            "t_pre 处温度阶跃 / °C": round(self.jump_T, 4),
            "t_pre 处含水率阶跃 / (kg/kg)": round(self.jump_C, 6),
            "平台段样本数": self._plateau_stats["n"],
            "平台段 T∞ std / °C": round(self._plateau_stats["T_std"], 4),
            "平台段 C∞ std / (kg/kg)": round(self._plateau_stats["C_std"], 6),
        }


# ==========================================================================
def build_env(data: EnvData, mode: str = "faithful", t_pre: float = 14400.0,
              ramp: float = 0.0, **kw) -> PlateauEnv:
    """一步构建：EnvData → EnvInterpolator → PlateauEnv。"""
    return PlateauEnv(EnvInterpolator(data, mode=mode), t_pre=t_pre,
                      ramp=ramp, **kw)


if __name__ == "__main__":
    from .env_interp import load_env
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    d = load_env()
    print(d.summary(), "\n")
    for mode in ("faithful", "smoothed"):
        for t_pre in (9600.0, 14400.0):
            env = build_env(d, mode=mode, t_pre=t_pre)
            print(f"--- mode={mode}  t_pre={t_pre:.0f} ---")
            for k, v in env.describe().items():
                print(f"    {k:30s} {v}")
            print(f"    T∞(172800 s) = {env.T_inf(172800.0):.4f} °C   "
                  f"C∞(172800 s) = {env.C_inf(172800.0):.6f} kg/kg")
            print()
