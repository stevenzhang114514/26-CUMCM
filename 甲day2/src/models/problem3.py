"""
CODE-15  阈值通过时刻求解器（问题3）      [甲 · M0 · Day2上午→下午]

题面要求（[Q-PDF] 问题3）
-------------------------
"按照烘干要求，药材**各处**的水分浓度应低于 0.15 kg/kg，
  请确定药材烘干所需要的时间（单位：h）。"

判据（清单 §CODE-15 逐条对应的执行）
-------------------------------------
    C_max(t) = max_{r∈[0,R0]} C(r,t),     t_* = inf{ t : C_max(t) < 0.15 }

1. **不得用平均值代替最大值** —— "各处"即全域，而含水率沿 r 单调递减，
   最大值出现在**中心**。本模块仍对全域取 max 并**记录 argmax 位置**，
   不直接假定它在中心（假定必须被验证）。
2. **用未舍入数值定位** —— 4 位小数只在写盘时施加（CODE-17）。
3. **连续模型的交点通常满足 C_max = 0.15 而非严格小于** ——
   故报告的是**「阈值通过时刻」**，并额外验证其后的一个受控小时间增量。
4. **区分「首次达标」与「持续达标」** —— 若环境可能回湿，两者不同。
   本工况恒温段 C∞ᵉᑫ ≡ 0.05 < 0.15，驱动势始终为正，预期不会回湿，
   但仍逐点核验（不靠预期）。
5. 若输出规定"最早整数秒达标"，采用该规则并注明。

三类误差必须**分开报告**（清单 §CODE-15 验收）
-----------------------------------------------
    ① 事件定位误差 —— 二分搜索的收敛容差
    ② 空间离散误差 —— 由 Δr 引起（CODE-19 给出）
    ③ 时间积分误差 —— 由格式阶数与步长引起（CODE-19 给出）
🔴 **定位到 1 s 不代表干燥时间整体准确到 1 s。**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.resample import center_value
from .problem2 import Problem2Setup, solve_problem2

C_TARGET = 0.15        # kg/kg（题面）


# ==========================================================================
@dataclass
class ThresholdResult:
    """阈值通过时刻及其误差分解。"""
    t_star: float                       # 阈值通过时刻（未舍入）s
    found: bool                         # 是否在给定视界内达标
    C_max_at: float                     # C_max(t_*) —— 应≈0.15，非严格小于
    r_argmax: float                     # 最大值所在半径 m
    t_bracket: tuple = (np.nan, np.nan)
    n_bisect: int = 0
    # 之后一个受控小增量处的核验
    t_check: float = np.nan
    C_max_check: float = np.nan
    still_below: bool = False
    # 首次 vs 持续
    # ⚠️ 持续性**不在本结构里**判定：需逐点核验，由 sustained_check() 返回。
    #    曾留过一个 sustained_ok 字段却从未赋值，在 json 里恒为 false，
    #    容易被误读成"未持续达标" → 已删除（权威字段是 sustained_check 的 sustained）。
    t_first_below: float = np.nan
    # 误差分解
    err_event: float = np.nan
    err_space: float = np.nan
    err_time: float = np.nan
    # 追踪曲线
    trace_t: np.ndarray = field(default_factory=lambda: np.array([]))
    trace_Cmax: np.ndarray = field(default_factory=lambda: np.array([]))
    bracket_src: str = ""

    def to_dict(self) -> dict:
        return {
            "t_star_s": self.t_star, "found": bool(self.found),
            "t_star_h": self.t_star / 3600.0 if self.found else float("nan"),
            "C_max_at_tstar": self.C_max_at, "r_argmax_m": self.r_argmax,
            "bracket": tuple(float(x) for x in self.t_bracket),
            "bracket_src": self.bracket_src,
            "n_bisect": self.n_bisect,
            "t_check_s": self.t_check, "C_max_at_check": self.C_max_check,
            "still_below_after": bool(self.still_below),
            "t_first_below_s": self.t_first_below,
            "err_event_s": self.err_event,
            "err_space_s": self.err_space, "err_time_s": self.err_time,
        }


# ==========================================================================
def c_max_of(C_cells, grid) -> tuple:
    """
    全域最大值及其位置。

    取值点 = 全部控制体中心 + r=0 处的二次外推点值
    （输出第 0 列报告的正是后者，判据必须与交付物一致）。
    """
    Cm_cell = float(np.max(C_cells))
    Cm_ctr = float(center_value(C_cells, grid))
    # 容差判定：初期剖面几乎平坦（C≡2.55）时，中心外推值与单元最大值
    # 只差浮点噪声，argmax 会落在任意单元上 —— 那并不表示"最大值不在中心"。
    if Cm_ctr >= Cm_cell - 1e-12 * max(1.0, abs(Cm_cell)):
        return max(Cm_ctr, Cm_cell), 0.0
    return Cm_cell, float(grid.rc[int(np.argmax(C_cells))])


def _segment_Cmax(setup, t_a, T_a, C_a, t_b):
    """从 (t_a, T_a, C_a) 推进到 t_b，返回终点的 (C_max, r_argmax)。"""
    sub = Problem2Setup(**{**setup.__dict__, "t_end": t_b})
    res, _ = solve_problem2(sub, t_output=np.array([t_b]), t0=t_a,
                            state0=(T_a, C_a), collect_diag=False)
    return c_max_of(res.C[-1], setup.grid)


# ==========================================================================
def locate_threshold(setup: Problem2Setup, dt_out: float = 60.0,
                     max_horizon: float = 6.0 * 86400.0,
                     target: float = C_TARGET,
                     bisect_tol: float = 1.0e-3,
                     check_dt: float = 600.0,
                     record_stride: int = 1) -> tuple:
    """
    定位 C_max(t) 首次跌破 target 的时刻。

    两阶段
    ------
    ① **粗扫**：从 0 以 dt_out 为输出间隔推进，记录 C_max(t) 曲线，
       **达标即停**（stop_fn），否则会跑满 max_horizon。
    ② **二分**：在锚点与上界之间二分。每次试算都**从同一个固定锚点**
       积分到试探时刻（solve_problem2 的 t0/state0），避免为试一个时刻重算整段。

    返回 (ThresholdResult, IntegrateResult粗扫结果, extra诊断)
     —— extra 里带每个输出时刻的 T_surf/C_surf，供 result3 写盘直接复用，
        避免为拿表面重构值再跑一遍全程。
    """
    g = setup.grid
    t_out = np.arange(dt_out, max_horizon + dt_out * 0.5, dt_out)

    # ★ 达标即停：不早停就要跑满 max_horizon（默认 6 天），
    #   而真实 t* 约 2.4 天 —— 白白多算一倍以上。
    #   多留一个输出间隔，保证"跌破点"及其前一点都在结果里。
    def _stop(t, state):
        return c_max_of(state["C"], g)[0] < target

    res, ex = solve_problem2(setup, t_output=t_out, stop_fn=_stop)
    pairs = [c_max_of(res.C[k], g) for k in range(len(res.times))]
    Cmax = np.array([p[0] for p in pairs])
    r_arg = np.array([p[1] for p in pairs])

    below = Cmax < target
    if np.any(below):
        k_b = int(np.argmax(below))          # 首次跌破的**输出序号**
        t_hi = float(res.times[k_b])
        t_anchor = 0.0 if k_b == 0 else float(res.times[k_b - 1])
        T_anchor = res.T[0] if k_b == 0 else res.T[k_b - 1]
        C_anchor = res.C[0] if k_b == 0 else res.C[k_b - 1]
        bracket_src = "两个相邻输出时刻"
    elif res.stop_reason == "event":
        # ★ 早停点落在两个输出时刻之间：跨越发生在 (末输出, 早停点] 上。
        #   若只认"输出里有跌破点"，这里会误判成"不可达标"
        #   （实测在 t*≈57.5 h 处踩过一次）。
        t_hi = float(res.t_final)
        t_anchor = float(res.times[-1])
        T_anchor, C_anchor = res.T[-1], res.C[-1]
        bracket_src = "早停点（落在输出间隔内）"
    else:
        return ThresholdResult(
            t_star=float("nan"), found=False, C_max_at=float(Cmax[-1]),
            r_argmax=float(r_arg[-1]), t_bracket=(float(res.times[-1]), np.nan),
            trace_t=res.times[::record_stride],
            trace_Cmax=Cmax[::record_stride]), res, ex

    # ---------------- 二分 ----------------
    # 🔴 **锚点必须固定**：每次试算都从 (t_anchor, T_anchor, C_anchor) 积分到试探时刻。
    #    早期实现把 t_a 往前挪却不更新锚点状态，等于"从新时刻、旧状态"出发，
    #    轨迹完全不对 —— 收敛后给出的 t* 处 C_max 竟然仍 > 0.15（实测 0.15009），
    #    是一个不报错但结果错的 bug。
    #    括号总宽 ≤ 两个输出间隔（约 600 s），从锚点重积的代价很小。
    t_lo = t_anchor
    n_bi = 0
    while (t_hi - t_lo) > bisect_tol:
        t_m = 0.5 * (t_lo + t_hi)
        Cm, _ = _segment_Cmax(setup, t_anchor, T_anchor, C_anchor, t_m)
        n_bi += 1
        if Cm < target:
            t_hi = t_m
        else:
            t_lo = t_m
        if n_bi > 200:
            raise RuntimeError("二分未在 200 次内收敛，检查实现")

    t_star = t_hi
    Cm_star, r_star = _segment_Cmax(setup, t_anchor, T_anchor, C_anchor, t_star)

    # 受控小增量处的复核（清单要求"单独验证其后一个受控小时间增量"）
    t_chk = t_star + check_dt
    Cm_chk, _ = _segment_Cmax(setup, t_anchor, T_anchor, C_anchor, t_chk)

    tr = ThresholdResult(
        t_star=t_star, found=True, C_max_at=Cm_star, r_argmax=r_star,
        t_bracket=(t_anchor, t_hi), n_bisect=n_bi,
        bracket_src=bracket_src,
        t_check=t_chk, C_max_check=Cm_chk,
        still_below=bool(Cm_chk < target),
        t_first_below=t_star,
        err_event=bisect_tol / 2.0,
        trace_t=res.times[::record_stride],
        trace_Cmax=Cmax[::record_stride],
    )
    return tr, res, ex


# ==========================================================================
def sustained_check(tr: ThresholdResult, setup: Problem2Setup,
                    n_probe: int = 6, span: float = 3600.0,
                    restart=None) -> dict:
    """
    「首次达标」vs「持续达标」核验。

    在 [t_* , t_* + span] 内均匀取 n_probe 个点，确认 C_max 始终 < target。
    ⚠️ 只有**逐点核验过**才能声称"持续达标"，不得由"驱动势始终为正"推断。

    实现：**一次积分**到 t_* + span，输出恰好落在探针时刻上
    （自适应步长会被裁剪到输出时刻，故探针处不需要插值）。

    restart=(t0, T0, C0)：由调用方给出一个 ≤ t_* 的已存状态，
    从那里续算即可，**不必从 0 重算**。
    """
    g = setup.grid
    ts = np.linspace(tr.t_star, tr.t_star + span, n_probe)
    sub = Problem2Setup(**{**setup.__dict__, "t_end": float(ts[-1])})
    t0, st0 = (0.0, None) if restart is None else (float(restart[0]),
                                                   (restart[1], restart[2]))
    res, _ = solve_problem2(sub, t_output=ts, t0=t0, state0=st0,
                            collect_diag=False)
    vals = [c_max_of(res.C[k], g)[0] for k in range(len(res.times))]
    ok = all(v < C_TARGET for v in vals)
    return {"t_probe": [float(x) for x in res.times],
            "C_max_probe": [float(v) for v in vals],
            "sustained": bool(ok),
            "note": "逐点核验；不由驱动势方向推断"}


# ==========================================================================
def integer_second_rule(t_star: float) -> int:
    """
    若输出规定"最早整数秒达标"，采用上取整并**在论文中注明该规则**。
    （连续模型的交点通常不落在整数秒上。）
    """
    return int(np.ceil(t_star - 1e-9))
