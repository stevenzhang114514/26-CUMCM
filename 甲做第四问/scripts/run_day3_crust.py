"""
结壳 / 玻璃化层对 t* 的影响（参数化扩展）        [甲 · M1 扩展 · Day3 追加]

物理图像
--------
表面先干 → 表层进入玻璃态 / 形成低渗透硬壳 → 该层的有效扩散系数比芯部低若干个数量级
→ 内部水分被"锁"住 → 干燥末期显著变慢。

模型（**一个参数都不改附录3，只在壳层上乘一个降幅因子**）
----------------------------------------------------------
        D_eff(C,T) = D_附录3(C,T) · β(C),      β(C) = β_c  当 C < C_g
                                                    1     当 C ≥ C_g

* 判据用**含水率**（等价于玻璃化判据：低含水率 → T_g 升高 → 进入玻璃态）
* `C_g` 为壳层临界含水率，`β_c` 为壳层扩散系数降幅
* 结壳与玻璃化在本模型里是**同一件事的两种叫法** —— 这一点在论文里必须说明

🔴 分层纪律
-----------
这是 **M1 扩展**，与 M0 基线**分开存放**。
它**不修订**问题3/4 的答案；只用于回答"如果结壳成立，答案会怎么变"。

🔴 参数没有文献标定值（见 docs/端面与结壳_文献与实现.md §二）
--------------------------------------------------------------
Gulati & Datta (2015) 等给出了**机理与建模框架**，但**没有给出中药材的
壳层扩散系数比**。研报 [R3-F5] 写的 "D_crust = 10⁻²~10⁻⁴ D_core" 的脚注
指向一个 ResearchGate 页面，**溯源自查失败**（见 `参考文献原文/引用核实与影响分析.md`）。
故本脚本把 β_c 当作**自由参数扫描**，只报告敏感性，不声称标定值。

用法：python scripts/run_day3_crust.py
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

import src.models.problem2 as P2          # noqa: E402
from src.config import DOC_DIR, NUMERICS  # noqa: E402
from src.data_prep.env_extrap import build_env   # noqa: E402
from src.data_prep.env_interp import load_env    # noqa: E402
from src.models.boundary import BoundaryConfig   # noqa: E402
from src.models.problem3 import C_TARGET         # noqa: E402
from src.numerics.fvm_cyl import Grid            # noqa: E402

T_PROBE = 206960.0
DT_OUT = 600.0
_ORIG_D = P2.D_app3


def P(*a):
    print(*a, flush=True)


def make_beta(C_g: float, beta_c: float, width: float = 0.02):
    """
    返回一个 D_app3 的包装：C < C_g 的单元乘 beta_c。

    🔴 用**平滑阶跃**而不是硬 if
    ----------------------------
    硬阶跃会让 D 对 C 不连续，自适应步长控制器在判定处反复缩步，
    实测在 t=73200 s 处触底报 `StepSizeTooSmall`（dt_min=1e-4）。
    改用 logistic 过渡（宽度 width）后 D 连续可导，控制器正常工作。

        β(C) = 1 − (1−β_c)·σ((C_g − C)/width)
    """
    def D_mod(C, T_celsius):
        D = np.atleast_1d(np.asarray(_ORIG_D(C, T_celsius), dtype=float))
        Cv = np.atleast_1d(np.asarray(C, dtype=float))
        sig = 1.0 / (1.0 + np.exp(-np.clip((C_g - Cv) / width, -60, 60)))
        return D * (1.0 - (1.0 - beta_c) * sig)
    return D_mod


def run(env, C_g, beta_c, N=200, grading=1.5):
    """积分到 T_PROBE，由 C_max 与局部斜率换算 t*。"""
    P2.D_app3 = make_beta(C_g, beta_c)
    try:
        grid = Grid(N=N, R0=0.02, grading=grading)
        st = P2.Problem2Setup(grid=grid, env=env, bc=BoundaryConfig(latent=False),
                              t_end=T_PROBE, label="M1-crust")
        t_out = np.arange(DT_OUT, T_PROBE + DT_OUT, DT_OUT)
        res, ex = P2.solve_problem2(st, t_output=t_out)
    finally:
        P2.D_app3 = _ORIG_D
    Cm = np.asarray(ex["C_max"], dtype=float)
    dCdt = (Cm[-1] - Cm[-2]) / (res.times[-1] - res.times[-2])
    t_star = res.times[-1] - (Cm[-1] - C_TARGET) / dCdt if dCdt != 0 else np.nan
    return float(t_star), float(Cm.min()), res.n_accepted


def main():
    P("=" * 96)
    P("  结壳 / 玻璃化层对 t* 的影响（参数化扫描，M1 扩展，不改题设）")
    P("=" * 96)
    env = build_env(load_env(), mode="faithful", t_pre=14400.0)
    out = {"说明": "D_eff = D_附录3 · β(C)，β=β_c 当 C<C_g；M1 扩展，不修订 M0 答案",
           "T_PROBE": T_PROBE, "runs": []}

    # 基线（无结壳）
    t0 = time.time()
    tb, _, nb = run(env, C_g=0.0, beta_c=1.0)      # C_g=0 → 永不触发
    P(f"\n  基线（无结壳）              t* = {tb:11.3f} s = {tb/3600:7.4f} h"
      f"   {time.time()-t0:4.0f}s")
    out["baseline_s"] = tb
    out["runs"].append({"C_g": 0.0, "beta_c": 1.0, "t_star_s": tb,
                        "dt_s": 0.0, "label": "基线"})

    P(f"\n  {'C_g':>6} {'β_c':>8} | {'t* / s':>12} {'t* / h':>9} "
      f"{'Δt* / s':>10} {'Δt* / h':>9} {'相对':>8}")
    P("  " + "-" * 74)
    for C_g in (0.15, 0.25, 0.35, 0.50):
        for beta_c in (0.1, 0.01, 0.001):
            t0 = time.time()
            ts, cmin, n = run(env, C_g, beta_c)
            dt = ts - tb
            P(f"  {C_g:6.2f} {beta_c:8.0e} | {ts:12.3f} {ts/3600:9.4f} "
              f"{dt:10.1f} {dt/3600:9.4f} {dt/tb*100:+7.2f}%   "
              f"({time.time()-t0:.0f}s)")
            out["runs"].append({"C_g": C_g, "beta_c": beta_c,
                                "t_star_s": float(ts), "dt_s": float(dt),
                                "rel_pct": float(dt / tb * 100),
                                "C_min": cmin})

    # 汇总：只看 β_c=0.01（研报区间的中档）
    sel = [r for r in out["runs"] if r.get("beta_c") == 0.01]
    if sel:
        P(f"\n  β_c=0.01 时的 Δt*（随 C_g 变化）：")
        for r in sel:
            P(f"    C_g={r['C_g']:.2f}  →  Δt* = {r['dt_s']/3600:+.4f} h "
              f"（{r['rel_pct']:+.2f}%）")
        out["beta001_spread_h"] = float(
            (max(r["t_star_s"] for r in sel) - min(r["t_star_s"] for r in sel)) / 3600)

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "day3_crust.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    P(f"\n  → docs/day3_crust.json")
    P("=" * 96)
    return out


if __name__ == "__main__":
    main()
