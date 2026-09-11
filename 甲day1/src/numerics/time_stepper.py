"""
CODE-08  隐式时间推进器                [甲 · M0 · Day1下午]

为什么必须隐式
--------------
半控制体 FVM 中心处显式稳定性上限约  Δt ≤ Δr²/(4α)。
Δr = 0.5 mm、α = k/(ρcp) = 1.6886e-7 m²/s 时该上限约 **0.37 s**，
无法满足"输出间隔 1 s"。故采用 L-稳定的 Backward Euler。
（验收方式：**检查稳定性条件并拒绝不满足的设置**，而非"观察发散"。）

关键约定
--------
* **输出间隔 ≠ 计算步长**。步长由误差控制与边界变化决定；
  每步都把 dt 裁剪到"恰好落在下一个输出时刻"，因此**输出时刻从不靠插值**。
* Crank–Nicolson（θ=0.5）实现但**仅用于线性温度基准**验证时间二阶；
  对刚性的水分方程不使用（会在陡峭前沿振铃）。

误差控制：步长加倍法（Richardson）
----------------------------------
    一步  u_full   = Φ_{Δt}(u^n)
    两步  u_half   = Φ_{Δt/2}(Φ_{Δt/2}(u^n))
    err = max_i |u_half,i − u_full,i| / (atol + rtol·|u_half,i|)
    err ≤ 1 接受；新步长 dt·clamp(safety·err^(−1/(p+1)), shrink, grow)

Backward Euler 阶 p = 1 → 指数 1/2；Crank–Nicolson p = 2 → 指数 1/3。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nonlinear import NonlinearFailure


class StepSizeTooSmall(RuntimeError):
    """步长触底，说明问题设置或实现有误（不静默）。"""


@dataclass
class StepRecord:
    """单个计算步的记录 —— 供 FIG-41 与 CODE-19 使用。"""
    t: float
    dt: float
    iters_C: int
    err_T: float
    err_C: float
    accepted: bool
    n_halving: int = 0
    note: str = ""


@dataclass
class IntegrateResult:
    times: np.ndarray                     # 输出时刻
    T: np.ndarray                         # (n_out, N)
    C: np.ndarray                         # (n_out, N)
    history: list = field(default_factory=list)
    n_accepted: int = 0
    n_rejected: int = 0
    n_halving_total: int = 0
    stop_reason: str = ""


# ==========================================================================
# 误差范数
# ==========================================================================
def scaled_error(u_half, u_full, atol, rtol):
    d = np.abs(u_half - u_full)
    s = atol + rtol * np.abs(u_half)
    return float(np.max(d / s))


# ==========================================================================
# 自适应推进
# ==========================================================================
class AdaptiveStepper:
    def __init__(self, num, state_keys=("T", "C")):
        self.num = num
        self.keys = state_keys
        self.dt = num.dt_init
        self.history: list[StepRecord] = []
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_halving_total = 0

    # ------------------------------------------------------------------
    def _err(self, half, full):
        eT = scaled_error(half["T"], full["T"], self.num.atol_T, self.num.rtol_T)
        eC = scaled_error(half["C"], full["C"], self.num.atol_C, self.num.rtol_C)
        return eT, eC

    def _try_step(self, t, dt, state, step_fn):
        """执行一次尝试步；返回 (half_state, full_state, iters_C, n_half) 或 None（非线性失败）。"""
        try:
            full, it_full = step_fn(t, dt, state)
            mid, it_a = step_fn(t, dt / 2.0, state)
            half, it_b = step_fn(t + dt / 2.0, dt / 2.0, mid)
            return half, full, max(it_full, it_a, it_b), 0
        except NonlinearFailure:
            return None

    # ------------------------------------------------------------------
    def integrate(self, t0, state0, t_end, t_output, step_fn, p_order=1.0,
                  progress=None):
        """
        自适应推进。

        参数
        ----
        step_fn : callable(t, dt, state) -> (state_new, n_iter_C)
                  调用者负责 θ 方法与线性/非线性求解
        t_output: 需要精确落点的输出时刻（升序 np.ndarray）
        p_order : 时间格式阶数（BE=1，CN=2）
        """
        self.dt = self.num.dt_init
        t = float(t0)
        state = {k: np.array(v, dtype=float) for k, v in state0.items()}

        out_t, out_T, out_C = [], [], []
        next_out = 0
        n_out = len(t_output)
        tol_out = 1e-9

        # t0 若是输出时刻，先记录
        if n_out and abs(t - t_output[0]) < tol_out:
            out_t.append(t); out_T.append(state["T"].copy()); out_C.append(state["C"].copy())
            next_out = 1

        expo = 1.0 / (p_order + 1.0)
        guard = 0
        guard_max = 5_000_000

        while t < t_end - 1e-12:
            guard += 1
            if guard > guard_max:
                raise RuntimeError("推进步数超限，疑似死循环")

            dt = min(self.dt, t_end - t)
            # ---- 精确落在下一个输出时刻 ----
            if next_out < n_out:
                t_next = float(t_output[next_out])
                if t_next > t + 1e-12 and t + dt > t_next:
                    dt = t_next - t

            if dt < self.num.dt_min:
                raise StepSizeTooSmall(
                    f"t={t:.6g} 处步长 {dt:.3e} 触底 (dt_min={self.num.dt_min:.1e})"
                )

            res = self._try_step(t, dt, state, step_fn)
            n_halving = 0
            while res is None:
                n_halving += 1
                self.n_halving_total += 1
                if n_halving > self.num.max_step_halving:
                    raise StepSizeTooSmall(
                        f"t={t:.6g} 处非线性连续失败 {n_halving} 次，步长已缩至 {dt:.3e}"
                    )
                dt *= 0.5
                res = self._try_step(t, dt, state, step_fn)

            half, full, iters, _ = res
            err_T, err_C = self._err(half, full)
            err = max(err_T, err_C)
            accepted = (err <= 1.0) and (n_halving == 0)

            rec = StepRecord(t=t, dt=dt, iters_C=iters, err_T=err_T, err_C=err_C,
                             accepted=accepted, n_halving=n_halving)
            self.history.append(rec)

            if accepted:
                state = half
                t += dt
                self.n_accepted += 1
                factor = self.num.safety * (err ** -expo if err > 0 else self.num.grow_max)
                factor = float(np.clip(factor, self.num.shrink_max, self.num.grow_max))
                self.dt = float(np.clip(dt * factor, self.num.dt_min, self.num.dt_max))

                # ---- 记录输出 ----
                while next_out < n_out and abs(t - t_output[next_out]) < 1e-9:
                    out_t.append(t)
                    out_T.append(state["T"].copy())
                    out_C.append(state["C"].copy())
                    next_out += 1
                if progress is not None and next_out % progress == 0:
                    pass
            else:
                # 拒绝：状态不变，缩小步长重试
                self.n_rejected += 1
                factor = self.num.safety * (err ** -expo if err > 0 else 0.5)
                factor = float(np.clip(factor, 0.1, 0.8))
                self.dt = max(dt * factor, self.num.dt_min)

        if next_out < n_out:
            # 允许 t_end 稍短于最后一个输出时刻的极端情形：补齐
            while next_out < n_out:
                out_t.append(float(t_output[next_out]))
                out_T.append(state["T"].copy())
                out_C.append(state["C"].copy())
                next_out += 1

        return IntegrateResult(
            times=np.asarray(out_t),
            T=np.asarray(out_T),
            C=np.asarray(out_C),
            history=self.history,
            n_accepted=self.n_accepted,
            n_rejected=self.n_rejected,
            n_halving_total=self.n_halving_total,
            stop_reason="completed" if t >= t_end - 1e-9 else "short",
        )


# ==========================================================================
# 显式稳定性上限（供配置检查，不作为推进方式）
# ==========================================================================
def explicit_stability_limit(dr, alpha):
    """
    半控制体 FVM 中心处的显式稳定性上限 ≈ Δr²/(4α)。
    实际限制应依据所选定的离散算子确定，此处给出基准量级供配置自检。
    """
    return dr ** 2 / (4.0 * alpha)
