"""
Day1 总入口（甲）                        [M0 · 题设基线]

执行链
------
    单位自检 → 环境数据装载 → 问题1 求解 → 解析基准 → 守恒/合理性检验
    → 重采样 → 写 result1.xlsx → 回读自检 → 出图 → 写运行日志

用法
----
    python scripts/run_day1.py [--N 200] [--grading 1.5] [--theta 1.0]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (DOC_DIR, H_COEF, HM_COEF, NUMERICS, PROPS_APP2, RESULT_DIR,
                        T_END, T_START, check_units, derived_numbers)
from src.data_prep.env_interp import EnvInterpolator, load_env
from src.io.excel_writer import print_validation, validate_output, write_result1
from src.io.resample import check_no_overshoot, output_radii_m, sample_field
from src.models.problem1 import Problem1Setup, solve_problem1
from src.numerics.fvm_cyl import Grid, assemble
from src.numerics.properties import D_app2
from src.validation import conservation, sanity
from src.validation.bessel_bench import run_benchmark


def _utf8():
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)
    except Exception:
        pass


def banner(s):
    print("\n" + "=" * 74)
    print(f"  {s}")
    print("=" * 74)


def main():
    _utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=NUMERICS.N)
    ap.add_argument("--grading", type=float, default=NUMERICS.grading)
    ap.add_argument("--theta", type=float, default=NUMERICS.theta)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    t_wall = time.time()
    log = {"run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
           "N": args.N, "grading": args.grading, "theta": args.theta, "layer": "M0"}

    # ================= 1. 单位与物性自检 =================
    banner("STEP 1  单位约定与物性自检  (CODE-01)")
    dcheck = check_units()
    log["d_checkpoints"] = {k: v["computed"] for k, v in dcheck.items()}
    der = derived_numbers()
    log["derived"] = der
    print(f"    α = {der['alpha']:.6e} m²/s      Bi_T = {der['Bi_T']:.4f}")
    print(f"    传热特征时间 R₀²/α = {der['tau_heat_s']:.1f} s")
    print(f"    显式稳定性上限 Δr²/(4α) = {der['explicit_dt_limit_s']:.4f} s"
          f"   ← 远小于 1 s 输出间隔，故必须隐式")
    print(f"    端面/径向时间尺度比 (L/2 / R₀)² = {der['end_face_scale_ratio']:.4f}")

    # ================= 2. 环境数据 =================
    banner("STEP 2  附件1 环境数据装载与插值  (CODE-02)")
    env_data = load_env()
    print(env_data.summary())
    env = EnvInterpolator(env_data, mode="faithful")
    env_sm = EnvInterpolator(env_data, mode="smoothed")
    print(f"    {env}")
    print(f"    平台段统计 (t≥9600 s): T∞={env.plateau_stats()['T_mean']:.4f} °C, "
          f"C∞={env.plateau_stats()['C_mean']:.6f} kg/kg")
    log["env"] = {"n_points": int(len(env_data.t)),
                  "t_span": env_data.t_span,
                  "T_range": [float(env_data.T.min()), float(env_data.T.max())],
                  "C_range": [float(env_data.C.min()), float(env_data.C.max())],
                  "plateau": env.plateau_stats()}

    # 两版本差异（对照用）
    tt = np.linspace(0, 1800, 101)
    dT = float(np.max(np.abs(env.T_inf(tt) - env_sm.T_inf(tt))))
    dC = float(np.max(np.abs(env.C_inf(tt) - env_sm.C_inf(tt))))
    print(f"    忠实插值 vs 受控平滑（0–1800 s）最大差: ΔT={dT:.4f} °C, ΔC={dC:.6f}")
    print(f"    ⚠ 实测波动遍布全程（温度差分变号 {int((np.diff(np.sign(np.diff(env_data.T)))!=0).sum())} 次），"
          f"默认为忠实插值；不推断控制器类型。")
    log["interp_compare"] = {"dT_max": dT, "dC_max": dC}

    # ================= 3. 问题1 求解 =================
    banner("STEP 3  问题1 求解（先水后温，单向耦合）  (CODE-10/11)")
    grid = Grid(N=args.N, R0=0.02, grading=args.grading)
    setup = Problem1Setup(grid=grid, env=env, theta=args.theta, t_end=float(T_END))
    t_out_all = np.concatenate(([0.0], np.arange(T_START, T_END + 1, dtype=float)))

    t0 = time.time()
    res, extra = solve_problem1(setup, t_output=t_out_all)
    wall = time.time() - t0
    print(f"    网格 {grid.summary()}   θ={args.theta}")
    print(f"    接受步 {res.n_accepted}，拒绝步 {res.n_rejected}，"
          f"额外折半 {res.n_halving_total} 次，耗时 {wall:.2f} s")
    print(f"    停止原因: {res.stop_reason}；输出时刻数 {len(res.times)}")
    print(f"    终态: T ∈ [{res.T[-1].min():.4f}, {res.T[-1].max():.4f}] °C, "
          f"C ∈ [{res.C[-1].min():.6f}, {res.C[-1].max():.6f}] kg/kg")
    log["solve"] = {"n_accepted": res.n_accepted, "n_rejected": res.n_rejected,
                    "n_halving": res.n_halving_total, "wall_s": wall,
                    "T_final_range": [float(res.T[-1].min()), float(res.T[-1].max())],
                    "C_final_range": [float(res.C[-1].min()), float(res.C[-1].max())]}

    # ================= 4. 解析基准 =================
    banner("STEP 4  常系数解析基准（Bessel 级数）  (CODE-18 甲自用)")
    bench = run_benchmark()
    log["benchmark"] = {"Bi": bench["Bi"],
                        "trunc_err": bench["trunc_err"],
                        "max_err_per_t": {str(t): float(e) for t, e in
                                          zip(bench["t_out"], bench["max_err_per_t"])}}

    # ================= 5. 守恒与合理性 =================
    banner("STEP 5  守恒与物理合理性检验  (CODE-20 / CODE-21)")
    D_final = np.atleast_1d(D_app2(res.C[-1])).astype(float)
    C_inf_final = float(env.C_inf(res.times[-1]))
    op_ref = assemble(D_final, HM_COEF, C_inf_final, grid)
    cons = conservation.flux_cancellation(res.C[-1], D_final, HM_COEF,
                                          C_inf_final, grid, op=op_ref)
    mass = conservation.mass_budget(res.times, extra["surface_flux_w"],
                                    res.C, grid)
    # 温度侧：向外的对流通量 q = h(Ts − T∞)，W/m²
    T_inf_hist = np.asarray(env.T_inf(res.times), dtype=float)
    q_conv = H_COEF * (extra["surface_T"] - T_inf_hist)
    heat = conservation.heat_budget(res.times, q_conv, res.T, grid,
                                    PROPS_APP2.rho, PROPS_APP2.cp)
    cons_rep = conservation.report(cons, mass, heat)
    log["conservation"] = cons_rep

    san = sanity.run_all(res.T, res.C, T_inf_hist,
                         np.asarray(env.C_inf(res.times), dtype=float), grid)
    log["sanity"] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                         float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in san.items()}

    # ================= 6. 重采样与写出 =================
    banner("STEP 6  重采样与写 result1.xlsx  (CODE-16 / CODE-17)")
    r_out = output_radii_m()
    sel = res.times >= T_START - 1e-9
    T_out = sample_field(res.T[sel], grid, extra["surface_T"][sel], r_out)
    C_out = sample_field(res.C[sel], grid, extra["surface_C"][sel], r_out)
    t_int = np.arange(T_START, T_END + 1, dtype=int)

    ok_c, exc_c, info_c = check_no_overshoot(C_out, res.C[sel], r_out_m=r_out)
    ok_t, exc_t, info_t = check_no_overshoot(T_out, res.T[sel], r_out_m=r_out)
    print(f"    采样形状 T{T_out.shape}  C{C_out.shape}  期望 ({len(t_int)}, 21)")
    print(f"    插值列无过冲（已排除 r=0 外推列与 r=R₀ 边界重构列）:")
    print(f"        温度 {'通过' if ok_t else '**未通过**'} (超出 {exc_t:.2e} °C)，"
          f"单元值域 {info_t['cell_range'][0]:.4f}~{info_t['cell_range'][1]:.4f}")
    print(f"        含水率 {'通过' if ok_c else '**未通过**'} (超出 {exc_c:.2e})，"
          f"单元值域 {info_c['cell_range'][0]:.6f}~{info_c['cell_range'][1]:.6f}")
    print(f"    输出范围: T ∈ [{T_out.min():.4f}, {T_out.max():.4f}] °C, "
          f"C ∈ [{C_out.min():.6f}, {C_out.max():.6f}] kg/kg")
    log["resample"] = {"no_overshoot_T": bool(ok_t), "no_overshoot_C": bool(ok_c),
                       "excess_T": float(exc_t), "excess_C": float(exc_c)}
    assert T_out.shape == (len(t_int), 21), "输出矩阵形状不符"

    meta = {
        "运行标识": log["run_id"], "模型层次": "M0（题设基线）",
        "问题": "问题1 预热平衡阶段（0–1800 s）",
        "物性": "附录2 常物性；M0 不含蒸发潜热项",
        "网格": grid.summary(),
        "时间格式": f"θ={args.theta} (Backward Euler)" if args.theta == 1.0
                    else f"θ={args.theta} (Crank–Nicolson)",
        "边界": "附件1 忠实 PCHIP 插值；C∞ 按题设等效约定读作等效平衡含水率",
        "单位": "t: s, r: m, T: °C（D 公式内转 K）",
        "接受步/拒绝步": f"{res.n_accepted} / {res.n_rejected}",
    }
    out_path = RESULT_DIR / "result1.xlsx"
    write_result1(out_path, t_int, T_out, C_out, meta=meta)
    print(f"    已写出 {out_path}")

    rep = validate_output(out_path, expect_t=t_int)
    ok_out = print_validation(rep)
    log["output_validation"] = {"ok": ok_out, "path": str(out_path)}

    # ================= 7. 出图 =================
    if not args.no_fig:
        banner("STEP 7  出图  (FIG-08 / FIG-41)")
        from src.figures import fig08_fvm, fig41_history
        # FIG-08 为示意图，控制体数目固定（不随生产网格 N 变化）
        p8 = fig08_fvm.make()
        print(f"    [FIG-08] " + "\n             ".join(str(p) for p in p8))
        p41, h41 = fig41_history.make(res.history)
        print(f"    [FIG-41] " + "\n             ".join(str(p) for p in p41))
        print(f"             接受 {h41['n_accepted']} / 拒绝 {h41['n_rejected']}，"
              f"平均 Δt={h41['dt_mean']:.3f} s，平均迭代 {h41['iters_mean']:.2f} 次")
        log["figures"] = {"fig08": [str(p) for p in p8],
                          "fig41": [str(p) for p in p41], "fig41_stats": h41}

    # ================= 8. 运行日志 =================
    log["wall_total_s"] = time.time() - t_wall
    log["status"] = "OK" if ok_out else "OUTPUT_VALIDATION_FAILED"
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DOC_DIR / f"run_log_{log['run_id']}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n运行日志: {log_path}")
    print(f"总耗时 {log['wall_total_s']:.2f} s，状态 {log['status']}")
    return 0 if ok_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
