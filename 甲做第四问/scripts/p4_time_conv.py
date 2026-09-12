"""
问题4 时间收敛：Δt = 30 / 10 / 3 / 1 s          [甲 · 验证脚本]

为什么必须做
------------
上一轮实测：Δt=30 与 Δt=10 给出的 t_dry 相差 **1203 s（0.66%）** ——
比空间离散误差还大。故必须把时间方向也钉住。

做法（省机时）
--------------
不做二分，改为**固定探测时刻法**：积分到 t_q = 184200 s，量 C_max(t_q)，
再由局部斜率 dC_max/dt 换算 t_dry：

        t_dry(Δt) ≈ t_q − (C_max(t_q) − 0.15) / (dC_max/dt)

每次只跑**一次**积分，比二分省 30 倍。

用法：python scripts/p4_time_conv.py
"""
from __future__ import annotations
import dataclasses, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import NUMERICS
from src.data_prep.env_extrap import build_env
from src.data_prep.env_interp import load_env
from src.models.problem4 import GridXiN, load_radius, solve_p4

P = lambda *a: print(*a, flush=True)
TQ = 184200.0
TARGET = 0.15
REF_H = 51.0823


def main():
    env = build_env(load_env(), mode="faithful", t_pre=14400.0)
    rad = load_radius()
    g = GridXiN(N=200, grading=1.5)
    P("=" * 84)
    P("  问题4 时间收敛（网格固定 N=200/γ=1.5，探测时刻 184200 s）")
    P("=" * 84)
    P(f"  {'Δt_max':>8} | {'C_max(t_q)':>14} {'dC/dt':>12} | {'t_dry / h':>11} {'相对Δt=1':>10} {'耗时':>7}")
    P("  " + "-" * 76)
    ref = None
    rows = []
    for dtm in (30.0, 10.0, 3.0, 1.0):
        num = dataclasses.replace(NUMERICS, dt_max=dtm, dt_init=dtm, dt_min=1e-4)
        t0 = time.time()
        t_out = np.array([TQ - 600.0, TQ])
        r = solve_p4(env, rad, g, TQ, t_out, num=num)
        el = time.time() - t0
        Cm1 = float(r["C"][-1].max()); Cm0 = float(r["C"][-2].max())
        dCdt = (Cm1 - Cm0) / 600.0
        td = (TQ - (Cm1 - TARGET) / dCdt) if dCdt < 0 else float("nan")
        if dtm == 1.0:
            ref = td
        rel = (td - ref) / ref * 100 if ref else float("nan")
        P(f"  {dtm:8.0f} | {Cm1:14.8f} {dCdt:12.4e} | {td/3600:11.6f} {rel:+9.3f}% {el:6.0f}s")
        rows.append({"dt_max": dtm, "C_max": Cm1, "dCdt": dCdt,
                     "t_dry_s": td, "t_dry_h": td / 3600.0, "rel_pct": rel})
    P("")
    P(f"  参考答案 = {REF_H} h")
    best = rows[-1]["t_dry_h"]
    P(f"  本项目收敛值 = {best:.6f} h   ⇒ 差 {best-REF_H:+.6f} h "
      f"= {(best-REF_H)*3600:+.1f} s")
    import json
    (ROOT / "docs" / "p4_time.json").write_text(
        json.dumps({"探测时刻": TQ, "参考_h": REF_H, "行": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    P("  → docs/p4_time.json")


if __name__ == "__main__":
    main()
