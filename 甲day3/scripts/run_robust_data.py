"""
审稿意见补做 ② — 稳健性检验：随机删数据 / 加噪声 / 换插值算法
                                                          [甲 · 验证脚本]

审稿人的原话要求
----------------
    "随机删除 10% 数据、加噪声、不同算法对比、K-fold 交叉验证"

本脚本做其中**三项**，并说明第四项为何不适用（见文末 §K-fold）：

    A  随机删除 10% 的附件1 数据点（24/241），重插值重求解   —— Monte Carlo n 次
    B  加噪声：B1 量化噪声（附件自带末位精度）；B2 假定传感器精度（题面未给）
    C  不同算法/插值对比：PCHIP(基准) / Savitzky–Golay 平滑 / 分段线性

🔴 **本节的区间不是"统计置信区间"** —— 必须写进论文
-----------------------------------------------------
附件1 是**每 60 s 一条的确定性采样时间表**，不是从某个总体里随机抽的样本。
"随机删掉 10%" 模拟的是**记录缺失**，不是"重做一次实验"。
因此 A/B 得到的散布刻画的是「**结果对数据点集合的敏感度**」，
**不能**当作烘干时间的置信区间，也不能与批次间真实变差混为一谈
（真实批次变差本文没有数据，[R4-F21] 也只给了 D 的 CV）。
论文里这个名字叫 **「数据扰动分位区间」**，与已有的
「情景区间」「参数扰动分位区间」并列，三者不得混称。

用法
----
    py scripts/run_robust_data.py --stage mc --n 20 --workers 6
    py scripts/run_robust_data.py --stage algo
    py scripts/run_robust_data.py --stage all
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VERSION = "A-2026-M0-r4"
T_PRE = 14400.0
N_GRID, GRADING = 200, 1.5
MAX_HORIZON = 6.0 * 86400.0
DT_OUT = 60.0
DROP_FRAC = 0.10
BASE_SEED = 20260912            # 固定种子 → 可复现

DOC = ROOT / "docs"
JSON_NAME = "day3_robust_data.json"
P = lambda *a: print(*a, flush=True)

# 附件1 的**实测**量化精度（不是假设）：T 3 位小数、C 5 位小数
QUANT_T = 0.5e-3
QUANT_C = 0.5e-5
# B2 假定的仪表精度（题面**未给**，属声明性情景，论文中必须写明"假定"）
SENS_T_SIGMA = 0.10             # °C —— 常用热电偶/铂电阻在 50 ℃ 附近
SENS_C_SIGMA = 1.0e-3           # kg/kg —— 约为 0.05 kg/kg 平台的 2%


# ==========================================================================
# 线性插值器（"不同算法对比"的第三个算法）
# ==========================================================================
class _Lin:
    """与 PCHIP(extrapolate=False) 同语义的分段线性插值（域外给 NaN）。"""

    def __init__(self, t, v):
        self.t, self.v = np.asarray(t, float), np.asarray(v, float)

    def __call__(self, x):
        xa = np.asarray(x, dtype=float)
        out = np.interp(xa, self.t, self.v)
        return np.where((xa < self.t[0]) | (xa > self.t[-1]), np.nan, out)


def make_linear_interp(data):
    from src.data_prep.env_interp import EnvInterpolator
    it = EnvInterpolator(data, mode="faithful")
    it.mode = "linear"
    it._T = _Lin(data.t, data.T)
    it._C = _Lin(data.t, data.C)
    return it


# ==========================================================================
# 扰动器
# ==========================================================================
def perturb(base, kind: str, seed: int):
    """
    返回 (EnvData, 说明 dict)。base 是原始 EnvData（**绝不被就地修改**）。
    """
    from src.data_prep.env_interp import EnvData
    rng = np.random.default_rng(seed)
    t, T, C = base.t.copy(), base.T.copy(), base.C.copy()
    info = {}

    if kind == "del10":
        k = int(round(DROP_FRAC * len(t)))
        # 🔴 **端点固定**：候选集只含内部点 1…n−2。
        #    为什么：末点 t=14400 s 是"实测段 → 恒温段平台值"的交接锚点，
        #    首点 t=0 是初值。删掉末点会把插值定义域缩到 14280 s，
        #    于是 PlateauEnv 的 t_pre=14400 越界——这不是"数据少了"，而是
        #    **建模约定被改了**。本节要测的是采样密度，不是换约定，故把端点钉住。
        #    （首次运行正是踩了这个：20 个样本里有 2 个抽到末点，直接抛
        #      ValueError: t_pre=14400.0 超出附件1 覆盖范围 (0, 14280.0]，整批崩掉。）
        cand = np.arange(1, len(t) - 1)
        idx = np.sort(rng.choice(cand, size=k, replace=False))
        keep = np.setdiff1d(np.arange(len(t)), idx)
        assert keep[0] == 0 and keep[-1] == len(t) - 1, "端点必须保留"
        t, T, C = t[keep], T[keep], C[keep]
        info = {"删除点数": k, "原点数": int(len(base.t)),
                "剩余点数": int(len(t)),
                "候选集": "内部点 1…n−2（**两端端点固定保留**）",
                "删除比例": f"{k}/{len(base.t)} = {k/len(base.t)*100:.2f}%",
                "方式": "无放回均匀随机，种子 {}".format(seed)}
    elif kind == "quant":
        T = T + rng.uniform(-QUANT_T, QUANT_T, size=T.shape)
        C = C + rng.uniform(-QUANT_C, QUANT_C, size=C.shape)
        info = {"噪声": "均匀分布", "T 半宽 / ℃": QUANT_T,
                "C 半宽 / (kg/kg)": QUANT_C,
                "来源": "附件1 自身末位精度（T 3 位、C 5 位小数）——**实测**，非假设",
                "种子": seed}
    elif kind == "sensor":
        T = T + rng.normal(0.0, SENS_T_SIGMA, size=T.shape)
        C = C + rng.normal(0.0, SENS_C_SIGMA, size=C.shape)
        info = {"噪声": "高斯独立同分布", "T σ / ℃": SENS_T_SIGMA,
                "C σ / (kg/kg)": SENS_C_SIGMA,
                "来源": "**假定**的仪表精度，题面未给（声明性情景，非实测）",
                "种子": seed}
    elif kind == "base":
        info = {"说明": "原始附件1，未扰动"}
    else:
        raise ValueError(kind)

    return EnvData(t=t, T=T, C=C, source=f"{base.source}#{kind}#{seed}"), info


# ==========================================================================
# 单次运行（模块级：Windows spawn 需可 pickle）
# ==========================================================================
def _run_one(job: dict) -> dict:
    # 🔴 子进程是**新解释器**（Windows spawn），不会继承父进程在 main() 里做的
    #    stdout.reconfigure，默认仍是 GBK → 打印"平台 ℃"会抛 UnicodeEncodeError，
    #    日志也会变成半 GBK 半 UTF-8 的混合体。故在**每个子进程**里再设一次。
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    from src.config import NUMERICS
    from src.data_prep.env_interp import load_env
    from src.models.boundary import BoundaryConfig
    from src.models.problem2 import Problem2Setup
    from src.models.problem3 import locate_threshold
    from src.numerics.fvm_cyl import Grid

    import dataclasses
    import src.models.problem2 as _P2
    from src.data_prep.env_extrap import PlateauEnv, build_env
    from src.data_prep.env_interp import EnvInterpolator

    # 允许单样本放宽步长下限（抢救 StepSizeTooSmall 的样本；默认不放松）。
    # 必须打到 problem2 的**模块级** NUMERICS 上，与 run_day3.run_config 同一做法。
    if job.get("dt_min"):
        from src.config import NUMERICS as _N
        _P2.NUMERICS = dataclasses.replace(_N, dt_min=float(job["dt_min"]))

    base = load_env()
    if job["kind"] in ("sg", "linear"):
        # 换插值算法：数据不动，只换插值/平滑方式
        info = {"说明": "原始附件1，仅更换插值算法"}
        interp = (make_linear_interp(base) if job["kind"] == "linear"
                  else EnvInterpolator(base, mode="smoothed"))
        env = PlateauEnv(interp, t_pre=T_PRE, ramp=0.0)
    else:
        data, info = perturb(base, job["kind"], job["seed"])
        env = build_env(data, mode="faithful", t_pre=T_PRE)

    grid = Grid(N=N_GRID, R0=0.02, grading=GRADING)
    setup = Problem2Setup(grid=grid, env=env, bc=BoundaryConfig(latent=False),
                          t_end=MAX_HORIZON, label=job["key"])
    t0 = time.time()
    tr, _, _ = locate_threshold(setup, dt_out=DT_OUT, max_horizon=MAX_HORIZON)
    wall = time.time() - t0

    rec = dict(job)
    rec.update({
        "found": bool(tr.found),
        "t_star_s": float(tr.t_star),
        "t_star_h": float(tr.t_star) / 3600.0 if tr.found else float("nan"),
        "T_plateau": float(env.T_plateau), "C_plateau": float(env.C_plateau),
        "n_plateau": int(env._plateau_stats["n"]),
        "wall_s": wall, "info": info, "atol_T": NUMERICS.atol_T,
        "N": N_GRID, "grading": GRADING, "version": VERSION,
        "sig": job_sig(job), "python": platform.python_version(),
        "dt_min": job.get("dt_min") or "prod",
    })
    P(f"    [{job['key']:>14s}] t* = {rec['t_star_h']:9.6f} h   "
      f"平台 {rec['T_plateau']:.4f} ℃ / {rec['C_plateau']:.6f}   ({wall:.0f}s)")
    return rec


# ==========================================================================
def job_sig(job: dict) -> str:
    """
    扰动定义的签名。**只有签名一致的旧结果才允许复用** ——
    否则改一次 perturb 的定义就会悄悄沿用按旧定义算出的数（这正是本次的教训：
    del10 的候选集从"全部点"改成"内部点"后，旧的 del10 结果必须全部作废重跑）。
    """
    k = job["kind"]
    if k == "del10":
        return (f"del10|interior-only|frac={DROP_FRAC}|seed={job['seed']}|"
                f"pin_endpoints=1|dt_min={job.get('dt_min') or 'prod'}")
    if k == "quant":
        return f"quant|U(±{QUANT_T},{QUANT_C})|seed={job['seed']}"
    if k == "sensor":
        return f"sensor|N(0,{SENS_T_SIGMA},{SENS_C_SIGMA})|seed={job['seed']}"
    return f"{k}|v1"


def build_jobs(n: int, keys=None, dt_min=None):
    jobs = [dict(key="base", kind="base", seed=BASE_SEED)]
    for k in range(n):
        jobs.append(dict(key=f"del10_{k:02d}", kind="del10", seed=BASE_SEED + 100 + k))
    for k in range(n):
        jobs.append(dict(key=f"quant_{k:02d}", kind="quant", seed=BASE_SEED + 200 + k))
    for k in range(n):
        jobs.append(dict(key=f"sensor_{k:02d}", kind="sensor", seed=BASE_SEED + 300 + k))
    for kind in ("sg", "linear"):
        jobs.append(dict(key=f"algo_{kind}", kind=kind, seed=BASE_SEED))
    if dt_min:
        for j in jobs:
            j["dt_min"] = dt_min
    if keys:
        # 🔴 base 必须**始终保留**：summarize() 要拿它当基准，
        #    过滤掉它会直接 StopIteration（抢救那 3 个样本时就踩过一次）。
        jobs = [j for j in jobs if j["key"] == "base" or j["key"] in keys]
    return jobs


_KIND_ORDER = {"base": 0, "del10": 1, "quant": 2, "sensor": 3, "sg": 4,
               "linear": 5, "pchip": 6}


def _sort_key(key: str):
    """稳定、可读的台账排序：先按扰动类别，再按 key 字面序。"""
    kind = key.split("_")[0] if not key.startswith("algo") else \
        key.split("_", 1)[1]
    return (_KIND_ORDER.get(kind, 9), key)


def _stats(vals):
    v = np.asarray(vals, dtype=float)
    return {
        "n": int(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1)),
        "min": float(v.min()), "max": float(v.max()),
        "q05": float(np.percentile(v, 5)), "q50": float(np.percentile(v, 50)),
        "q95": float(np.percentile(v, 95)),
        "range": float(v.max() - v.min()),
    }


def summarize(recs):
    base = next(r for r in recs if r["key"] == "base")
    t0 = base["t_star_h"]
    groups = {}
    for r in recs:
        k = r["kind"]
        if k in ("base", "pchip"):
            continue
        groups.setdefault(k, []).append(r)
    out = {"base": base, "runs": recs, "groups": {}}
    for k, rs in groups.items():
        ts = [x["t_star_h"] for x in rs if x["found"]]
        st = _stats(ts)
        st.update({
            "dt_star_s": (st["mean"] - t0) * 3600.0,
            "rel_vs_base": st["mean"] / t0 - 1.0,
            "half_width_h": 0.5 * st["range"],
            "info": rs[0]["info"],
            "keys": [x["key"] for x in rs],
        })
        out["groups"][k] = st
    return out


def print_summary(out):
    t0 = out["base"]["t_star_h"]
    P("")
    P("=" * 92)
    P("  数据扰动稳健性")
    P("=" * 92)
    P(f"  基准（原始附件1，PCHIP）  t* = {t0:.6f} h")
    name = {"del10": "A 随机删 10% 数据", "quant": "B1 量化噪声（附件末位精度）",
            "sensor": "B2 传感器噪声（假定 σ_T=0.1 ℃）",
            "sg": "C 换 SG 平滑插值", "linear": "C 换分段线性插值"}
    P(f"  {'组':<30s}{'n':>4s}{'均值 t*/h':>13s}{'σ/h':>10s}"
      f"{'极差/h':>10s}{'5%':>11s}{'95%':>11s}")
    P("  " + "-" * 88)
    for k, g in out["groups"].items():
        P(f"  {name.get(k, k):<30s}{g['n']:>4d}{g['mean']:>13.6f}{g['std']:>10.5f}"
          f"{g['range']:>10.5f}{g['q05']:>11.6f}{g['q95']:>11.6f}")
    P("")
    P("  ⚠️ 以上是**数据扰动分位区间**，不是统计置信区间（附件1 非随机抽样）。")


# ==========================================================================
def dump(name, obj):
    DOC.mkdir(parents=True, exist_ok=True)

    def jd(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    p = DOC / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jd),
                 encoding="utf-8")
    P(f"\n  → {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["mc", "algo", "all"])
    ap.add_argument("--n", type=int, default=20, help="每组 Monte Carlo 样本数")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rescue", default="", help="逗号分隔的 key，只跑这些样本")
    ap.add_argument("--dt-min", type=float, default=None,
                    help="放宽步长下限（默认用生产值 1e-4），用于抢救 "
                         "StepSizeTooSmall 的样本")
    a = ap.parse_args()

    # Windows 控制台默认 GBK，打印 U+2212/⚠ 会抛 UnicodeEncodeError → 统一改 UTF-8
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t0 = time.time()
    keys = [k.strip() for k in a.rescue.split(",") if k.strip()] or None
    if a.stage == "mc":
        jobs = build_jobs(a.n, keys=keys, dt_min=a.dt_min)
    elif a.stage == "algo":
        jobs = build_jobs(0)
        jobs = [j for j in jobs if j["kind"] in ("base", "sg", "linear")]
    else:
        jobs = build_jobs(a.n, keys=keys, dt_min=a.dt_min)

    P("=" * 92)
    P(f"  审稿补做 ② 数据扰动稳健性 / 不同算法     版本 {VERSION}")
    P("=" * 92)
    P(f"  作业数 {len(jobs)}   并行度 {a.workers}   网格 N={N_GRID}/γ={GRADING}"
      f"   dt_min={a.dt_min if a.dt_min else '生产值'}")
    P("")
    # ★ 增量执行：签名一致的旧结果直接复用（同一批计算，不重复烧机时）
    done = {}
    p_json = DOC / JSON_NAME
    if p_json.exists():
        old = json.loads(p_json.read_text(encoding="utf-8"))
        if (old.get("meta", {}).get("N") == N_GRID
                and old.get("meta", {}).get("grading") == GRADING):
            done = {r["key"]: r for r in old.get("runs", [])
                    if r.get("found") and r.get("sig")}
    todo = [j for j in jobs if done.get(j["key"], {}).get("sig") != job_sig(j)]
    P(f"  作业共 {len(jobs)}，签名一致可复用 {len(jobs)-len(todo)}，"
      f"本次实跑 {len(todo)}")
    for j in todo[:6]:
        P(f"    待跑 {j['key']}")
    if len(todo) > 6:
        P(f"    … 共 {len(todo)} 个")
    P("")

    # 🔴 **按 key 并集合并，不按"本次作业表"过滤**。
    #    早期实现写成 `recs = [r for r in recs if r["key"] in order]`，
    #    于是 `--rescue`（只跑 3 个 key）会把 json 里其余 60 条**全部丢掉** ——
    #    实测踩过一次，整批 63 次运行的结果被覆盖成 4 条。
    #    正确语义：json 是该实验的**累积台账**，本次只更新它算过的 key。
    merged = dict(done)
    if todo:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_run_one, j): j["key"] for j in todo}
            for f in as_completed(futs):
                try:
                    r = f.result()
                    merged[r["key"]] = r
                except Exception as e:          # 单个样本失败不得拖垮整批
                    P(f"    !! 样本失败 {futs[f]}: {type(e).__name__}: {e}")
    recs = sorted(merged.values(), key=lambda r: _sort_key(r["key"]))
    P(f"\n  完成，本批耗时 {time.time()-t0:.0f} s    "
      f"台账累计 {len(recs)} 条")
    missing = [j["key"] for j in jobs if j["key"] not in merged]
    if missing:
        P(f"  ⚠️ 本次作业中**未取得结果**的样本 {len(missing)} 个：{missing}")

    out = summarize(recs)
    out["meta"] = {
        "version": VERSION, "N": N_GRID, "grading": GRADING,
        "drop_frac": DROP_FRAC, "n_mc": a.n, "base_seed": BASE_SEED,
        "method": "problem3.locate_threshold（与交付值同源）",
        "not_a_ci": "数据扰动分位区间 ≠ 统计置信区间；附件1 为 60 s 确定性采样，非随机样本",
    }
    dump(JSON_NAME, out)
    print_summary(out)


if __name__ == "__main__":
    main()
