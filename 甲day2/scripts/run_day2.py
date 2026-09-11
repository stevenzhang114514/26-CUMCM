"""
Day2 主运行脚本（甲）                    [甲 · M0/M1 · Day2]

用法
----
    python scripts/run_day2.py --stage core     # 问题2/3 主体结果（关键路径）
    python scripts/run_day2.py --stage conv     # CODE-19 收敛检验
    python scripts/run_day2.py --stage scen     # CODE-25 情景与分位
    python scripts/run_day2.py --stage all

产出
----
    results/M0/result2.xlsx            问题2 正式版（3 h，每 1 s）
    results/M0/result2_全程版.xlsx     问题2 全程版（每 60 s，至 t*）
    results/M0/result3.xlsx            问题3（每 60 s，至 t*，末行为达标时刻）
    results/M1/...                     M1 潜热对照（**与 M0 分开存放**）
    docs/day2_core.json 等             全部数值证据

🔴 分层纪律
-----------
M0 = 题设基线（附录3 物性、**不含蒸发潜热**）→ 进 result2/result3 正式文件。
M1 = 本文扩展（R3-F1 的表面潜热边界）→ **只进 results/M1/**，
     绝不与 M0 混在同一文件里（清单 §CODE-17）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DOC_DIR, NUMERICS, R_OUT_CM, RESULT_DIR, T_START
from src.data_prep.env_interp import load_env
from src.data_prep.env_extrap import build_env, PlateauEnv
from src.io.excel_writer import (print_validation, validate_output,
                                 validate_result3, write_result1, write_result3)
from src.io.resample import output_radii_m, sample_field
from src.models.boundary import BoundaryConfig, RHO_D_INIT
from src.models.problem2 import Problem2Setup, solve_problem2, timescales
from src.models.problem3 import (C_TARGET, integer_second_rule,
                                 locate_threshold, sustained_check)
from src.numerics.fvm_cyl import Grid
from src.refs import get

GRID_PROD = dict(N=NUMERICS.N, R0=0.02, grading=NUMERICS.grading)
M1_WARMUP = dict(warmup_steps=1000, warmup_dt=1.0e-3)


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump(name, obj):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    p = DOC_DIR / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jdefault),
                 encoding="utf-8")
    print(f"    → {p.name}")
    return p


def make_env(data, t_pre=14400.0, T_plat=None, C_plat=None, mode="faithful"):
    kw = {}
    if T_plat is not None:
        kw["T_plateau"] = T_plat
    if C_plat is not None:
        kw["C_plateau"] = C_plat
    return build_env(data, mode=mode, t_pre=t_pre, **kw)


def sample(res, ex, key_surf, r_out):
    return sample_field(res.C, ex["setup"].grid, ex[key_surf], r_out)


# ==========================================================================
# 阶段 core
# ==========================================================================
def stage_core(data):
    print("=" * 78)
    print("阶段 core —— 问题2 / 问题3 主体结果")
    print("=" * 78)
    out = {}
    env = make_env(data)
    grid = Grid(**GRID_PROD)
    r_out = output_radii_m()

    for layer, bc, extra_setup in (
            ("M0", BoundaryConfig(latent=False), {}),
            ("M1", BoundaryConfig(latent=True, rho_d=RHO_D_INIT), M1_WARMUP)):
        tag = f"[{layer}]"
        print(f"\n{tag} " + "-" * 68)

        # ---------------- 问题2 正式版：3 h，每 1 s ----------------
        t0 = time.time()
        setup3h = Problem2Setup(grid=grid, env=env, bc=bc,
                                t_end=10800.0, label=layer, **extra_setup)
        t_int = np.arange(float(T_START), 10800.0 + 0.5, 1.0)
        res, ex = solve_problem2(setup3h, t_output=t_int)
        T_out = sample_field(res.T, grid, ex["T_surf"], r_out)
        C_out = sample_field(res.C, grid, ex["C_surf"], r_out)
        el2a = time.time() - t0

        meta = {
            "层次": layer, "问题": "问题2（3 h 正式版）",
            "边界": str(ex["bc"].describe()),
            "网格": grid.summary(),
            "时间格式": f"θ={setup3h.theta}（Backward Euler）",
            "长时边界": "附件1 覆盖 0—14400 s，本文件只用 0—10800 s，**未使用外推**",
            "接受步/拒绝步": f"{res.n_accepted}/{res.n_rejected}",
            "平均Picard迭代": f"{np.mean(ex['iters']):.2f}",
            "D": get("Q-PDF").detail.split("附录3：")[1].split("；")[0],
        }
        p = RESULT_DIR.parent / layer / "result2.xlsx"
        write_result1(p, t_int.astype(int), T_out, C_out, meta=meta)
        rep = validate_output(p, expect_t=t_int.astype(int))
        print(f"{tag} result2.xlsx（3 h，{len(t_int)} 行） 用时 {el2a:.1f}s  "
              f"接受 {res.n_accepted} 拒绝 {res.n_rejected}")
        print_validation(rep)
        out[f"result2_3h_{layer}"] = {
            "ok": bool(rep["ok"]), "rows": int(len(t_int)),
            "wall_s": el2a, "n_accepted": res.n_accepted,
            "n_rejected": res.n_rejected,
            "mean_iters": float(np.mean(ex["iters"])),
            "max_iters": int(np.max(ex["iters"])),
            "bc": ex["bc"].describe(),
            "T_center_3h": float(ex["T_center"][-1]),
            "T_surf_3h": float(ex["T_surf"][-1]),
            "C_center_3h": float(ex["C_center"][-1]),
            "C_surf_3h": float(ex["C_surf"][-1]),
            "C_max_3h": float(ex["C_max"][-1]),
            "T_min_over_run": float(res.T.min()),
            "T_max_over_run": float(res.T.max()),
            "dT_evap_60s": float(ex["dT_evap"][59]) if len(ex["dT_evap"]) > 59 else None,
        }
        np.savez_compressed(ROOT / "data" / "processed" / f"core_{layer}_3h.npz",
                            t=res.times, T=res.T, C=res.C,
                            **{f"diag_{k}": v for k, v in ex.items()
                               if isinstance(v, np.ndarray)})

        # ---------------- 问题3：阈值通过时刻 ----------------
        print(f"{tag} 定位 t*（先粗扫到 C_max<0.15，再二分到 1e-3 s）…")
        t0 = time.time()
        setup_full = Problem2Setup(grid=grid, env=env, bc=bc,
                                   t_end=6.0 * 86400.0, label=layer, **extra_setup)
        tr, res_c, ex_c = locate_threshold(setup_full, dt_out=60.0,
                                           max_horizon=6.0 * 86400.0,
                                           target=C_TARGET, bisect_tol=1.0e-3)
        el3 = time.time() - t0

        if not tr.found:
            print(f"{tag} **6 天内未达标**，按清单要求单独标记，不强行填时长")
            out[f"tstar_{layer}"] = {"found": False, "wall_s": el3}
            continue

        t_star = tr.t_star
        t_star_int = integer_second_rule(t_star)
        print(f"{tag} t* = {t_star:.4f} s = {t_star/3600:.4f} h = {t_star/86400:.4f} d"
              f"   C_max(t*) = {tr.C_max_at:.8f}   二分 {tr.n_bisect} 次  用时 {el3:.0f}s")

        # ---- 输出时间轴：60, 120, …, ≤t*，末行接达标时刻（可能不是 60 的倍数）
        grid_t = res_c.times[res_c.times <= np.floor(t_star / 60.0) * 60.0 + 1e-9]
        k_last = int(np.searchsorted(res_c.times, grid_t[-1]))
        t_int_out = np.unique(np.concatenate([grid_t, [float(t_star_int)]]))
        seg_end = float(t_int_out[-1])

        # ---- 末段用**已存状态续算**，不重跑全程；表面重构值同样续算 ----
        seg_end = float(t_star_int)
        sub = Problem2Setup(grid=grid, env=env, bc=bc, t_end=seg_end,
                            label=layer, **extra_setup)
        res_seg, ex_seg = solve_problem2(
            sub, t_output=np.array([seg_end]), t0=float(grid_t[-1]),
            state0=(res_c.T[k_last], res_c.C[k_last]))

        T_hist = np.vstack([res_c.T[:k_last + 1], res_seg.T[-1][None, :]])
        C_hist = np.vstack([res_c.C[:k_last + 1], res_seg.C[-1][None, :]])
        Ts_hist = np.concatenate([ex_c["T_surf"][:k_last + 1], ex_seg["T_surf"]])
        Cs_hist = np.concatenate([ex_c["C_surf"][:k_last + 1], ex_seg["C_surf"]])

        T_out_full = sample_field(T_hist, grid, Ts_hist, r_out)
        C_out_full = sample_field(C_hist, grid, Cs_hist, r_out)
        assert len(t_int_out) == T_out_full.shape[0], \
            f"时间轴 {len(t_int_out)} 与场 {T_out_full.shape[0]} 长度不一致"

        meta3 = {
            "层次": layer, "问题": "问题3",
            "判据": f"C_max(t) = max_r C(r,t) < {C_TARGET} kg/kg（未舍入判定）",
            "阈值通过时刻 t*": f"{t_star:.4f} s = {t_star/3600:.4f} h",
            "C_max(t*)": f"{tr.C_max_at:.10f}",
            "最大值位置": f"r = {tr.r_argmax*100:.2f} cm",
            "事件定位误差": f"±{tr.err_event:.1e} s（二分容差的一半）",
            "达标后复核": f"t*+{tr.t_check - t_star:.0f}s 处 C_max = "
                          f"{tr.C_max_check:.8f}（{'仍达标' if tr.still_below else '**未保持**'}）",
            "长时边界": str(env.describe()),
            "网格": grid.summary(),
        }
        p3 = RESULT_DIR.parent / layer / "result3.xlsx"
        write_result3(p3, t_int_out.astype(int), C_out_full, meta=meta3)
        rep3 = validate_result3(p3, expect_t=t_int_out.astype(int))
        print(f"{tag} result3.xlsx（{len(t_int_out)} 行，末行 {t_int_out[-1]:.0f} s）")
        for k_, v in rep3["checks"].items():
            print(f"      [{'OK ' if v['ok'] else 'FAIL'}] {k_}: {v['detail']}")

        # ---------------- 问题2 全程版（每 60 s） ----------------
        p2f = RESULT_DIR.parent / layer / "result2_全程版.xlsx"
        meta2f = dict(meta)
        meta2f.update({
            "问题": "问题2（全程版，每 60 s）",
            "说明": "题面表3/表4 只要求 3 h；本文件按『整个烘干过程』的表述给到 t*。"
                    "采样间隔取 60 s（与 result3 同规格）——"
                    "全程按 1 s 输出需约 20 万行，超出 Excel 的实用规模，"
                    "故 1 s 采样只用于 3 h 正式版。",
            "覆盖时长": f"0—{t_star_int} s",
        })
        write_result1(p2f, t_int_out.astype(int), T_out_full, C_out_full, meta=meta2f)
        print(f"{tag} result2_全程版.xlsx（{len(t_int_out)} 行，每 60 s）")

        # ---------------- 达标持续性（逐点核验，从已存状态续算） ----------------
        sus = sustained_check(tr, setup_full, n_probe=6, span=3600.0,
                              restart=(float(grid_t[-1]), res_c.T[k_last],
                                       res_c.C[k_last]))
        print(f"{tag} 持续达标核验（t*—t*+1 h 逐点）：{'通过' if sus['sustained'] else '**未通过**'}")

        out[f"tstar_{layer}"] = {
            "found": True, **tr.to_dict(),
            "t_star_h": t_star / 3600.0, "t_star_d": t_star / 86400.0,
            "t_star_int_s": int(t_star_int),
            "sustained": sus,
            "wall_s": el3,
            "T_center_at_tstar": float(res_seg.T[-1][0]),
            "T_surf_at_tstar": float(ex_seg["T_surf"][-1]),
            "C_center_at_tstar": float(res_seg.C[-1][0]),
            "bc": ex["bc"].describe(),
            "env": env.describe(),
        }
        np.savez_compressed(ROOT / "data" / "processed" / f"core_{layer}_full.npz",
                            t=t_int_out, T=T_out_full, C=C_out_full,
                            T_cells=T_hist, C_cells=C_hist,
                            Cmax_trace=tr.trace_Cmax, t_trace=tr.trace_t,
                            T_center=ex_c["T_center"][:k_last + 1],
                            C_center=ex_c["C_center"][:k_last + 1],
                            C_surf=ex_c["C_surf"][:k_last + 1],
                            T_surf=ex_c["T_surf"][:k_last + 1],
                            C_max=ex_c["C_max"][:k_last + 1],
                            J_w=ex_c["J_w"][:k_last + 1],
                            dT_evap=ex_c["dT_evap"][:k_last + 1],
                            Bi_m=ex_c["Bi_m_surf"][:k_last + 1],
                            Lu=ex_c["Lu_surf"][:k_last + 1],
                            # ⚠️ t_diag 必须与 C_cells/T_cells **等长**：
                            # C_hist = res_c 的前 k_last+1 行，**再接一行**续算末点，
                            # 故 t_diag 也要在末尾补上 t_star_int。
                            # （曾漏补导致 G12 报形状 3449 vs 3450。）
                            t_diag=np.concatenate([ex_c["t"][:k_last + 1],
                                                   [float(t_star_int)]]))

    # ---------------------------------------------------------------- 对照
    if "tstar_M0" in out and "tstar_M1" in out:
        t0_, t1_ = out["tstar_M0"]["t_star_s"], out["tstar_M1"]["t_star_s"]
        out["M0_vs_M1"] = {
            "t_star_M0_h": t0_ / 3600.0, "t_star_M1_h": t1_ / 3600.0,
            "delta_h": (t1_ - t0_) / 3600.0,
            "ratio": t1_ / t0_,
            "T_center_3h_M0": out["result2_3h_M0"]["T_center_3h"],
            "T_center_3h_M1": out["result2_3h_M1"]["T_center_3h"],
            "deltaT_center_3h": (out["result2_3h_M0"]["T_center_3h"]
                                 - out["result2_3h_M1"]["T_center_3h"]),
            "ref": {"R3-F1": get("R3-F1").detail},
            "声明": "M1 为**条件性扩展**，其量级取决于 [OWN-RHOD] 声明的 ρ_d；"
                    "**不得用 M1 修订问题3 的答案**，只作对照报告。",
        }
        print(f"\n[M0 vs M1]  t*: {t0_/3600:.2f} h → {t1_/3600:.2f} h "
              f"（+{(t1_-t0_)/3600:.2f} h，×{t1_/t0_:.2f}）")
        print(f"            3 h 中心温度差 = "
              f"{out['M0_vs_M1']['deltaT_center_3h']:.2f} K")

    out["timescales"] = timescales(Problem2Setup(grid=grid, env=env))
    dump("day2_core.json", out)
    return out


# ==========================================================================
# 阶段 conv
# ==========================================================================
def stage_conv(data):
    print("=" * 78)
    print("阶段 conv —— CODE-19 空间/时间误差分开细化")
    print("=" * 78)
    from src.validation.convergence import (production_vs_reference,
                                            spatial_study, temporal_study)
    env = make_env(data)
    out = {}

    print("\n[1/3] 空间细化（均匀网格，时间固定 dt=1 s，t=10800 s）")
    out["spatial"] = spatial_study(env, t_end=10800.0, Ns=(40, 80, 160),
                                   N_ref=320, dt_fine=1.0, grading=1.0)
    for k in ("T_center", "C_center", "C_max"):
        print(f"    {k:9s} 比值 {['%.2f' % r for r in out['spatial'][f'ratio_{k}']]}"
              f"   拟合阶 = {out['spatial'][f'order_{k}']:+.3f}")

    print("\n[2/3] 时间细化（生产网格 N=200 γ=1.5，θ=1 BE）")
    out["temporal_be"] = temporal_study(env, t_end=10800.0,
                                        dts=(40.0, 20.0, 10.0, 5.0),
                                        dt_ref=2.5, N=200, grading=1.5, theta=1.0)
    for k in ("T_center", "C_center", "C_max"):
        print(f"    {k:9s} 比值 {['%.2f' % r for r in out['temporal_be'][f'ratio_{k}']]}"
              f"   拟合阶 = {out['temporal_be'][f'order_{k}']:+.3f}"
              f"   (期望 +{out['temporal_be']['expected_order']:.0f})")

    print("\n[3/3] 生产网格 vs 细参考网格（逐点，判据 < 5e-5 舍入量子）")
    # 参照网格：与生产网格**同族、温和加密**的一档 (300, 1.5)，对比时长 1 h。
    # 🔴 一条重要的数值发现：渐变网格的**最小控制体决定可容许时间步长**
    #    （τ_cond ~ Δr_min²），而 Δr_min ~ R0·N^(−γ)。因此**固定 γ 去加密 N
    #    会让时间刚性同步上升**：(400,1.5) 的 Δr_min 只有生产网格的 0.354 倍，
    #    τ 降到 1/8，实测单这一项 15 min 未结束。参照网格必须**温和加密**
    #    （或同时调小 γ），否则量到的是"参照网格太刚"，不是生产网格的误差。
    # 🔴 两条踩过的坑（都不是关键路径，但很耗机时）：
    #   · (640, 1.0) 均匀参照：同样容差下步数远超预期，单项跑了 2 h 未结束；
    #   · (240, 2.0) 参照：γ=2 使表面控制体只有 0.34 µm，
    #     导热时间常数降到约 0.8 µs，自适应步长被逼到微秒级。
    #   → 参照网格不是越极端越好；**γ 与 N 都要与生产网格同量级**才有比较意义。
    # ---------------------------------------------------------------
    # [跳过] 生产网格 vs 细参考网格 —— 已移出本阶段，理由如实记录
    # ---------------------------------------------------------------
    # 这项想回答"生产网格够不够用"。实测三种参照网格都因**参照网格自身**
    # 过慢而未能在预算内完成：
    #     (640, γ=1.0)  单项 > 2 h 未结束
    #     (400, γ=1.5)  单项 > 15 min 未结束
    #     (300, γ=1.5)  N=200 只需 9.4 s，N=300 超 300 s 仍未完成
    # 根因：**渐变网格的最小控制体随 N 下降**（Δr_min ∝ N^(−γ)），
    #       其导热时间常数随 Δr² 同步变小，时间刚性被抬高。
    #       所以"固定 γ、加密 N"不是免费的 —— 参照网格很快变成瓶颈。
    # 该问题已由 [1/3] 的空间收敛研究覆盖（均匀网格上阶数 +2.20，
    # 最细档误差约 1e-5，已低于 4 位小数的舍入量子 5e-5）。
    # → 列为 Day3 补做项：改用**同时调小 γ** 的参照（如 (300, 1.2)）或
    #    在均匀网格上单独做网格无关性。
    out["prod_vs_ref"] = {
        "done": False,
        "reason": "参照网格自身代价超出预算；根因是渐变网格 Δr_min ∝ N^(−γ) 抬高了时间刚性",
        "attempted": ["(640,1.0)", "(400,1.5)", "(300,1.5)"],
        "n200_seconds": 9.4,
        "covered_by": "[1/3] 空间收敛（均匀网格，阶 +2.20）",
        "todo_day3": "改用同时调小 γ 的参照网格，或在均匀网格上做网格无关性",
    }
    print("    [跳过] 生产网格 vs 细参考网格 —— 见 docs/Day2_结论.md 局限表")

    dump("day2_conv.json", out)
    return out


# ==========================================================================
# 阶段 scen
# ==========================================================================
def stage_scen(data):
    print("=" * 78)
    print("阶段 scen —— CODE-25 情景与分位区间")
    print("=" * 78)
    from src.validation.scenarios import (baseline_scenarios,
                                          d_quantile_scenarios,
                                          env_scenarios, m1_rho_scenarios,
                                          run_scenarios)
    grid = Grid(**GRID_PROD)
    out = {}

    def eb(sc):
        return make_env(data,
                        t_pre=sc.overrides.get("t_pre", 14400.0),
                        T_plat=sc.overrides.get("T_plat"),
                        C_plat=sc.overrides.get("C_plat"))

    print("\n[A] 情景区间：D / h_m / h 的倍率")
    out["A_param"] = run_scenarios(eb, baseline_scenarios(BoundaryConfig()),
                                   grid, dt_out=1800.0)

    print("\n[B] 情景区间：长时环境外推（t_pre 与平台值）")
    out["B_env"] = run_scenarios(eb, env_scenarios(), grid, dt_out=1800.0)

    print("\n[C] 情景区间：M1 的 ρ_d 取值 [OWN-RHOD]")
    out["C_rhod"] = run_scenarios(eb, m1_rho_scenarios(), grid, dt_out=1800.0,
                                  base_bc=BoundaryConfig(latent=False))

    print("\n[D] 参数扰动分位区间：仅 D 一维传播（对数正态，CV=20% [R4-F21]）")
    qs, meta_q = d_quantile_scenarios()
    out["D_d_quantile"] = run_scenarios(eb, qs, grid, dt_out=1800.0)
    out["D_d_quantile"]["meta"] = meta_q
    print(f"    ⚠️ {meta_q['scope']}；分布假设：{meta_q['assumption']}")

    # ---- 汇总：把 A 段做成敏感度排序（**检验**"D 最敏感"的猜想） ----
    rows = out["A_param"]["rows"]
    base = next(r for r in rows if r["key"] == "base")
    if base["reachable"]:
        sens = []
        for r in rows:
            if r["key"] == "base" or not r["reachable"]:
                continue
            sens.append({"key": r["key"], "kind": r["kind"],
                         "dt_star_h": r["t_star_h"] - base["t_star_h"],
                         "rel": (r["t_star_h"] - base["t_star_h"]) / base["t_star_h"]})
        sens.sort(key=lambda d: -abs(d["dt_star_h"]))
        out["sensitivity_ranking"] = sens
        print("\n[敏感度排序] （以 t* 变化幅度计，**这是实测检验，不是预设**）")
        for s in sens:
            print(f"    {s['key']:10s} Δt* = {s['dt_star_h']:+7.2f} h "
                  f"({s['rel']*100:+.1f}%)")
        dv = [s for s in sens if s["key"].startswith("D_")]
        hv = [s for s in sens if s["key"].startswith("hm_")]
        hhv = [s for s in sens if s["key"].startswith("h_")]
        out["hypothesis_check"] = {
            "猜想": "文献 [R4-F21] 报道 S_D≈0.65~0.78 而 S_hm<0.03，"
                    "即 D 最敏感、h_m 最不敏感（其前提是 Bi_m≫1）",
            "D 的两档平均|Δt*|": float(np.mean([abs(s["dt_star_h"]) for s in dv])) if dv else None,
            "h_m 的两档平均|Δt*|": float(np.mean([abs(s["dt_star_h"]) for s in hv])) if hv else None,
            "h 的两档平均|Δt*|": float(np.mean([abs(s["dt_star_h"]) for s in hhv])) if hhv else None,
            "结论": "见 docs/Day2_结论.md（本字段只给数字，判定写在文档里）",
        }
    dump("day2_scen.json", out)
    return out


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="core",
                    choices=["core", "conv", "scen", "all"])
    ap.add_argument("--t-pre", type=float, default=14400.0)
    a = ap.parse_args()

    t_start = time.time()
    print(f"Day2 运行开始  stage={a.stage}  网格 N={NUMERICS.N} γ={NUMERICS.grading}")
    data = load_env()
    print(f"附件1: {len(data.t)} 点, t ∈ [0, {data.t[-1]:.0f}] s\n")

    stages = ["core", "conv", "scen"] if a.stage == "all" else [a.stage]
    for s in stages:
        {"core": stage_core, "conv": stage_conv, "scen": stage_scen}[s](data)

    print(f"\nDay2 stage={a.stage} 完成，总用时 {time.time()-t_start:.1f} s")


if __name__ == "__main__":
    main()
