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
    n_warmup: int = 0                     # 起始层强制等步长的步数
    t_final: float = 0.0                  # 实际推进到的时刻（早停时 < t_end）
    stop_reason: str = ""                 # completed | short | event


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
                  progress=None, warmup_steps=0, warmup_dt=1.0e-3,
                  stop_fn=None):
        """
        自适应推进。

        参数
        ----
        step_fn : callable(t, dt, state) -> (state_new, n_iter_C)
                  调用者负责 θ 方法与线性/非线性求解
        t_output: 需要精确落点的输出时刻（升序 np.ndarray）
        p_order : 时间格式阶数（BE=1，CN=2）
        warmup_steps, warmup_dt :
                  ★ **起始层处理**（Day2 新增，M1 必需）。见下方说明。
        stop_fn  : callable(t, state) -> bool。在每个**被接受**的步之后询问，
                   返回 True 即停止（`stop_reason="event"`）。
                   ★ **事件早停**（Day2 新增）：问题3 的阈值搜索若不定早停，
                     会一路跑满 6 天视界，白白多算 2—3 倍；
                     情景扫描（CODE-25）里这个浪费会放大 20 倍。

        ★ 为什么需要 warmup —— 步长加倍法在 t=0 处会失效
        ---------------------------------------------------
        若边界条件在 t=0 处有**阶跃**（M1 的蒸发潜热正是如此：
        t=0⁺ 瞬间 J_w = 5.6e-4 kg/(m²·s)，等效环境温度被压到 −22 °C），
        则真解在表面附近按 **√t** 规律演化 —— 解析解的时间导数在 t=0 发散。

        此时 Backward Euler 跨在 t=0 上的**首个**步的局部误差不是 O(Δt²)，
        而是 **O(Δt^{1/2})**；步长加倍法量到的 |u_full − u_half| 也按
        Δt^{1/2} 衰减。于是"缩小步长使 err ≤ 1"永远无法达成：
        实测 err 在 Δt = 1e-4 s 时仍为 161.6，且**与 Δt 无关地保持同一数值**，
        最终触发 StepSizeTooSmall。**这是估计器的适用条件问题，不是物理发散。**

        处置：在 [t0, t0 + warmup_steps·warmup_dt] 内**强制等步长、不做误差控制**，
        把 √t 起始层走完；之后解已光滑，估计器恢复正常，再交回自适应控制。
        ⚠️ 起始层内的误差必须**单独量化并在论文中如实报告**
        （做法：改变 warmup_dt 重跑，比较关心量的差异），
        不得因为"后面看不出差别"就默认它不存在。
        """
        # 有起始层时从 warmup_dt 起步（而非 dt_init），
        # 使自适应控制从起始层结束处**平滑接管**，避免一上来就连续拒绝大步长
        self.dt = float(warmup_dt) if warmup_steps > 0 else self.num.dt_init
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
        n_warm = 0

        def _record_out():
            nonlocal next_out
            while next_out < n_out and abs(t - t_output[next_out]) < 1e-9:
                out_t.append(t)
                out_T.append(state["T"].copy())
                out_C.append(state["C"].copy())
                next_out += 1

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

            in_warmup = n_warm < warmup_steps
            if in_warmup:
                dt = min(warmup_dt, dt)

            if dt < self.num.dt_min:
                raise StepSizeTooSmall(
                    f"t={t:.6g} 处步长 {dt:.3e} 触底 (dt_min={self.num.dt_min:.1e})"
                )

            if in_warmup:
                # ---- 起始层：单步推进，不做误差估计 ----
                out = step_fn(t, dt, state)
                state = {k: np.array(v, dtype=float) for k, v in out[0].items()}
                t += dt
                n_warm += 1
                self.n_warmup = n_warm
                self.history.append(StepRecord(
                    t=t - dt, dt=dt, iters_C=out[1], err_T=-1.0, err_C=-1.0,
                    accepted=True, n_halving=0, note="warmup"))
                _record_out()
                continue

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
                _record_out()
                # ---- 事件早停（在输出已记录之后判定，保证停机点可复现）----
                if stop_fn is not None and stop_fn(t, state):
                    self.n_early_stop = True
                    break
            else:
                # 拒绝：状态不变，缩小步长重试
                self.n_rejected += 1
                factor = self.num.safety * (err ** -expo if err > 0 else 0.5)
                factor = float(np.clip(factor, 0.1, 0.8))
                self.dt = max(dt * factor, self.num.dt_min)

        early = bool(getattr(self, "n_early_stop", False))
        if next_out < n_out and not early:
            # 允许 t_end 稍短于最后一个输出时刻的极端情形：补齐
            while next_out < n_out:
                out_t.append(float(t_output[next_out]))
                out_T.append(state["T"].copy())
                out_C.append(state["C"].copy())
                next_out += 1
        elif next_out < n_out and early:
            # 🔴 **事件早停时绝不能补齐**。
            # 早停点的状态只在 t* 时刻有效，把它复制到之后几千个输出时刻
            # 会伪造出一整段"含水率不再变化"的假结果
            # （实测：t*=57.6 h 早停后仍补出 8640 行、末行到 518400 s）。
            # 未到达的输出时刻**就是没有结果**，如实截断。
            pass

        self.n_warmup = n_warm
        if getattr(self, "n_early_stop", False):
            reason = "event"
        elif t >= t_end - 1e-9:
            reason = "completed"
        else:
            reason = "short"
        return IntegrateResult(
            times=np.asarray(out_t),
            T=np.asarray(out_T),
            C=np.asarray(out_C),
            history=self.history,
            n_accepted=self.n_accepted,
            n_rejected=self.n_rejected,
            n_halving_total=self.n_halving_total,
            n_warmup=n_warm,
            t_final=float(t),
            stop_reason=reason,
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
