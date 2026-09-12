"""
差异归因：是不是因为潜热 / 端面 / 结壳？          [甲 · 验证脚本]

结论（先说）
------------
**不是。** 三个因素在本项目交付的答案里**本来就不存在**：

| 因素 | 在 M0 交付答案里的状态 | 若计入会怎样 |
|---|---|---|
| 蒸发潜热 | **关闭**（`BoundaryConfig(latent=False)`） | M1 实测 $t_*$ = 60.33 h，**比参考值更远** |
| 端面效应 | 只作**验证**用；交付值取自一维径向 | 对 $t_*$ 影响 < 1e-5（低于输出分辨率） |
| 结壳 | **不入基线**（M1 扩展） | 实测 +42%~+970%，方向与量级都不符 |

本脚本做**正向**验证：把候选原因逐一带进来，看谁能让 $t_*$ 移动 ~80 s。

用法：python scripts/explain_reference_gap.py
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import NUMERICS                       # noqa: E402
from src.data_prep.env_interp import load_env         # noqa: E402
from src.data_prep.env_extrap import PlateauEnv       # noqa: E402
from src.models.boundary import BoundaryConfig        # noqa: E402
from src.models.problem2 import Problem2Setup         # noqa: E402
from src.models.problem3 import locate_threshold      # noqa: E402
from src.numerics.fvm_cyl import Grid                 # noqa: E402

P = lambda *a: print(*a, flush=True)
REF_P3_H = 57.46681156
MY_P3_H = 57.4889


def env_with(data, T_plat, C_plat, mode="faithful", t_pre=14400.0):
    # 🔴 必须用**真正的** EnvInterpolator：PlateauEnv.__post_init__ 会调用
    #    它的 plateau_stats()。最初用鸭子类型的空对象冒充，直接 AttributeError。
    from src.data_prep.env_interp import EnvInterpolator
    interp = EnvInterpolator(data, mode=mode)
    return PlateauEnv(interp, t_pre=t_pre, T_plateau=T_plat,
                      C_plateau=C_plat, ramp=0.0)


def tstar(env, latent=False, N=200, grading=1.5, label=""):
    g = Grid(N=N, R0=0.02, grading=grading)
    bc = BoundaryConfig(latent=latent, rho_d=275.0422535211268)
    km = dict(warmup_steps=1000, warmup_dt=1.0e-3) if latent else {}
    st = Problem2Setup(grid=g, env=env, bc=bc, t_end=6.0 * 86400.0,
                       label="M1" if latent else "M0", **km)
    t0 = time.time()
    tr, _, _ = locate_threshold(st, dt_out=60.0, max_horizon=6.0 * 86400.0)
    el = time.time() - t0
    P(f"   {label:42s} t* = {tr.t_star:11.3f} s = {tr.t_star/3600:9.6f} h"
      f"   ({el:.0f}s)")
    return tr.t_star / 3600.0


def main():
    d = load_env()
    P("=" * 94)
    P("  差异归因：候选原因逐个代入，看谁能让 t* 移动 ~80 s")
    P("=" * 94)
    P(f"  参考 {REF_P3_H} h   本项目 M0 {MY_P3_H} h   差 {MY_P3_H-REF_P3_H:+.6f} h"
      f" = {(MY_P3_H-REF_P3_H)*3600:+.1f} s")
    P("")

    P("① 基准（交付口径）：平台 = t≥9600 s 实测均值")
    m = d.t >= 9600.0
    base = tstar(env_with(d, float(d.T[m].mean()), float(d.C[m].mean())),
                 latent=False, label="M0，实测平台 (50.0017/0.049998)")
    P("")

    P("② 换平台窗口（同一个模型，只是'恒温段环境取哪个值'不同）")
    for tag, Tp, Cp in (
            ("整数平台 (50.0/0.05)", 50.0, 0.05),
            ("t≥12400 s 均值 (50.0092/0.04998)", 50.0092, 0.04998),
            ("只取末点 (50.1650/0.049860)", 50.1650, 0.049860)):
        tstar(env_with(d, Tp, Cp), latent=False, label=tag)
    P("")

    P("③ 计入蒸发潜热（M1）—— 检验'是不是潜热造成的'")
    tstar(env_with(d, float(d.T[m].mean()), float(d.C[m].mean())),
          latent=True, label="M1，潜热开启")
    P("")

    P("=" * 94)
    P("  判读")
    P("=" * 94)
    P("  · 若②的平台变动就能让 t* 移动 ~80 s ⇒ 差异来自**长时环境口径**，不是模型")
    P("  · 若③把 t* 推向 60 h 量级 ⇒ 潜热方向**相反且量大 100 倍**，")
    P("    若计入只会**离参考值更远**")
    P("  · 端面效应对 t* < 1e-5（见 docs/03），**量级不够**")
    P("  · 结壳不入基线，且 +42%~+970%，**量级与方向都不符**")


if __name__ == "__main__":
    main()
