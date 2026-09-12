"""
CODE-26  端面效应：二维轴对称有限圆柱 vs 一维径向无限长圆柱    [甲 · M0 · Day3 追加]

要回答的问题
------------
问题1—3 一律用**无限长圆柱**的一维径向模型。题面药材长 25 cm、半径 2 cm，
$AR = L/(2R_0) = 6.25$，端面面积占侧面的 8% —— 端面到底带走多少水分？

做法
----
1. **内建回归**：端面 h=0 时，二维必须逐点退化为一维（不通过则一切作废）
2. **正式对照**：端面与侧面用**同一组** h、h_m（题设只给了一组系数），
   分别算二维有限圆柱与一维无限长圆柱的 $t_*$，给出端面效应

文献对照（见 docs/端面与结壳_文献与实现.md）
--------------------------------------------
[R4-F20] 声称忽略端面会使全域平均含水率偏差 3.5%~6.0%。本脚本用**自算**检验该区间。

用法：python scripts/run_day3_endo.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DOC_DIR, NUMERICS, T_END
from src.data_prep.env_extrap import build_env
from src.data_prep.env_interp import load_env
from src.models.boundary import BoundaryConfig
from src.models.problem2 import Problem2Setup, solve_problem2
from src.models.problem3 import C_TARGET
from src.models.problem3_2d import Grid2D, check_insulated_end, solve_2d
from src.numerics.fvm_cyl import Grid


def P(*a):
    print(*a, flush=True)


def tstar_1d(env, N=200, grading=1.5, t_horizon=4.0 * 86400.0, dt_out=600.0):
    """一维径向的 t*（C_max 首次跌破 0.15）。用固定输出间隔粗定位。"""
    g = Grid(N=N, R0=0.02, grading=grading)
    st = Problem2Setup(grid=g, env=env, bc=BoundaryConfig(latent=False),
                       t_end=t_horizon)
    t_out = np.arange(dt_out, t_horizon + dt_out, dt_out)
    res, _ = solve_problem2(st, t_output=t_out)
    Cm = res.C.max(axis=1)
    below = Cm < C_TARGET
    if not np.any(below):
        return float("nan"), Cm, res.times
    k = int(np.argmax(below))
    t_lo = 0.0 if k == 0 else float(res.times[k - 1])
    t_hi = float(res.times[k])
    return t_hi, Cm, res.times


def tstar_2d(env, g2, h_end_scale=1.0, t_horizon=4.0 * 86400.0,
             dt_out=900.0, dt_max=10.0, verbose=True):
    """二维轴对称的 t*（C_max 首次跌破 0.15），粗定位到输出间隔。"""
    num = dataclasses.replace(NUMERICS, dt_max=dt_max, dt_init=0.05)
    t_out = np.arange(dt_out, t_horizon + dt_out, dt_out)
    t0 = time.time()
    r = solve_2d(env, g2, t_end=t_horizon, t_output=t_out,
                 h_end_scale=h_end_scale, num=num, verbose=verbose)
    Cm = r["C_max"]
    below = Cm < C_TARGET
    if not np.any(below):
        return float("nan"), r, time.time() - t0
    k = int(np.argmax(below))
    return float(r["times"][k]), r, time.time() - t0


def main():
    P("=" * 92)
    P("  CODE-26  端面效应：二维轴对称有限圆柱 vs 一维径向")
    P("=" * 92)
    env = build_env(load_env(), mode="faithful", t_pre=14400.0)
    out = {}

    # ---------------- ① 内建回归 ----------------
    P("\n① 内建回归：端面绝热时 2D 必须等于 1D（判据 2e-3）")
    reg = check_insulated_end(Nr=20, Nz=12, t_end=3600.0, dt_max=1.0)
    out["回归"] = reg
    if not reg["通过"]:
        P("  ❌ 回归未通过 —— 后续端面结果不可用，停止。")
        return out

    # ---------------- ② 一维参照 ----------------
    P("\n② 一维径向参照（无限长圆柱，N=200/γ=1.5）")
    t0 = time.time()
    ts1, Cm1, tt1 = tstar_1d(env)
    P(f"   t*_1D = {ts1:.1f} s = {ts1/3600:.4f} h   （{time.time()-t0:.0f}s）")
    out["t_star_1D_s"] = float(ts1)

    # ---------------- ③ 二维有限圆柱 ----------------
    g2 = Grid2D(Nr=30, Nz=20, grading_r=1.5, grading_z=1.5)
    P(f"\n③ 二维轴对称有限圆柱（端面与侧面同系数）")
    P(f"   {g2.summary()}")
    ts2, r2, el = tstar_2d(env, g2, h_end_scale=1.0, dt_max=10.0)
    P(f"   t*_2D = {ts2:.1f} s = {ts2/3600:.4f} h   （{el:.0f}s，"
      f"{r2['n_accepted']} 步）")
    out.update({"t_star_2D_s": float(ts2), "t_star_2D_h": ts2 / 3600.0,
                "grid2d": g2.summary(), "n_steps": int(r2["n_accepted"]),
                "wall_s": el})

    # ---------------- ④ 端面效应 ----------------
    dt_s = ts1 - ts2
    P("\n④ 端面效应")
    P(f"   一维（无限长）  t* = {ts1/3600:8.4f} h")
    P(f"   二维（有限长）  t* = {ts2/3600:8.4f} h")
    P(f"   ⇒ 忽略端面使 t* **偏长 {dt_s:.1f} s = {dt_s/3600:.4f} h "
      f"（{dt_s/ts2*100:+.2f}%）**")
    P(f"   即：一维模型把干燥时间高估了 {dt_s/ts2*100:.2f}%")
    out.update({"delta_t_s": float(dt_s), "delta_t_h": float(dt_s / 3600.0),
                "rel_pct": float(dt_s / ts2 * 100.0)})

    # ---------------- ⑤ 与文献区间对照 ----------------
    P("\n⑤ 与 [R4-F20] 的文献区间对照")
    P(f"   文献称：忽略端面使全域平均含水率偏差 3.5%~6.0%")
    P(f"   自算  ：t* 偏差 {dt_s/ts2*100:.2f}%  ← 注意口径不同"
      f"（文献是**含水率**偏差，此处是**时间**偏差）")

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "day3_endo.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    P(f"\n   → docs/day3_endo.json")
    P("=" * 92)
    return out


if __name__ == "__main__":
    main()
