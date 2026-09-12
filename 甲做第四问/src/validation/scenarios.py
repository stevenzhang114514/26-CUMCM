"""
CODE-25  关键参数与环境情景区间         [甲主 / 乙佐 · M1 · Day2晚间]

🔴 三种区间**必须分开命名**，不得混为一谈（v3 清单 §CODE-25）
-------------------------------------------------------------
| 类型             | 含义                         | 何时可用        |
|------------------|------------------------------|-----------------|
| **情景区间**     | 人为设定参数范围 → 结果范围  | 随时            |
| **参数扰动分位区间** | 由**参数分布**传播得到   | 需明确分布假设  |
| **统计置信区间** | 由**实测样本**得到           | **需批次测量数据** |

本模块输出前两类，且**逐条标注属于哪一类**。
🔴 本工况**没有**批次测量数据 → **不作统计置信区间**，也不得把情景区间说成置信区间。

❌ 删除预设"D 最敏感、h 最不敏感"
----------------------------------
清单要求把"D 最敏感"当作**待检验猜想**。文献 [R4-F21] 报道
S_D ≈ 0.65~0.78、S_h_m < 0.03，但该结论的前提是 **Bi_m ≫ 1**；
本工况自算初期 Bi_m ≈ 1.2（见 FIG-16），前提未必成立 → 必须实测检验。

扰动优先级（清单指定）
----------------------
    D 的倍率 → h_m → 长时环境外推 → （半径拟合方式属问题4，此处不涉及）→ h、初始半径
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..models.problem2 import Problem2Setup, solve_problem2
from ..models.problem3 import c_max_of
from ..models.boundary import BoundaryConfig, RHO_D_INIT
from ..data_prep.env_extrap import PlateauEnv
from ..refs import get


# ==========================================================================
@dataclass
class Scenario:
    """一个情景：人为设定的参数取值 + **所属区间类型**。"""
    key: str
    kind: str                    # "情景" 或 "参数扰动分位"
    desc: str
    overrides: dict
    ref_keys: tuple = ()         # 支撑该情景的引用键


def _run_to_target(setup, target=0.15, dt_out=900.0, max_horizon=8.0 * 86400.0):
    """
    跑一个情景并给出：t*（粗定位）、是否可达标、C_max(t) 曲线。
    用粗输出间隔即可 —— 情景研究关心的是**趋势与量级**，
    精确的 t* 只对基线（CODE-15）做二分定位。

    ★ 达标即停（stop_fn）。20 个情景若每个都跑满 8 天视界，
      机时会放大一个数量级。
    """
    t_out = np.arange(dt_out, max_horizon + dt_out * 0.5, dt_out)
    g = setup.grid

    def _stop(t, state):
        return c_max_of(state["C"], g)[0] < target

    res, _ = solve_problem2(setup, t_output=t_out, stop_fn=_stop)
    Cmax = np.array([c_max_of(res.C[k], g)[0] for k in range(len(res.times))])
    below = Cmax < target
    if np.any(below):
        k = int(np.argmax(below))
        t_lo = 0.0 if k == 0 else float(res.times[k - 1])
        c_lo = Cmax[k - 1] if k > 0 else float(setup.C_init)
        t_hi, c_hi = float(res.times[k]), float(Cmax[k])
    elif res.stop_reason == "event":
        # ★ 与 CODE-15 同一类陷阱：早停点落在两个输出时刻**之间**，
        #   输出数组里根本看不到"跌破"。若只认输出，会把可达标的情景
        #   误判成不可达标（实测基线情景就这样被误判过）。
        t_lo, c_lo = float(res.times[-1]), float(Cmax[-1])
        t_hi, c_hi = float(res.t_final), target      # t_final 处已知 < target
    else:
        return {"t_star_s": float("nan"), "reachable": False,
                "t_coarse_s": float("nan"), "n_accepted": res.n_accepted,
                "t": res.times, "Cmax": Cmax,
                "T_center_end": float(res.T[-1][0]),
                "C_center_end": float(res.C[-1][0])}
    # 线性插值给出粗定位（量级用；精确值由 CODE-15 二分给）
    w = (c_lo - target) / (c_lo - c_hi) if c_lo > c_hi else 0.0
    t_coarse = t_lo + w * (t_hi - t_lo)
    # 用**末个已记录输出**处的场作为"达标时刻附近"的场
    #（不是 res.T[-1]：早停时最后一行的时刻与输出网格不重合，语义含糊）
    kk = len(res.times) - 1
    return {"t_star_s": t_coarse, "reachable": True, "t_coarse_s": t_coarse,
            "n_accepted": res.n_accepted, "t": res.times, "Cmax": Cmax,
            "T_center_end": float(res.T[kk][0]),
            "C_center_end": float(res.C[kk][0])}


# ==========================================================================
# 情景集
# ==========================================================================
def baseline_scenarios(bc: BoundaryConfig) -> list:
    """情景区间（一维、逐参数）—— D、h_m、h、环境外推。"""
    return [
        Scenario("base", "情景", "题设基线：D、h_m、h 均取题面值，t_pre=14400 s",
                 {}, ("Q-PDF",)),
        # ---- D 的倍率 ----
        Scenario("D_x0.5", "情景", "D 整体减半", {"x_D": 0.5}, ("R4-F21",)),
        Scenario("D_x2", "情景", "D 整体加倍", {"x_D": 2.0}, ("R4-F21",)),
        # ---- h_m ----
        Scenario("hm_x0.5", "情景", "h_m 减半（外阻力加倍）", {"x_hm": 0.5}, ("R4-F21",)),
        Scenario("hm_x2", "情景", "h_m 加倍", {"x_hm": 2.0}, ("R4-F21",)),
        # ---- h ----
        Scenario("h_x0.5", "情景", "h 减半", {"x_h": 0.5}, ("Q-PDF",)),
        Scenario("h_x2", "情景", "h 加倍", {"x_h": 2.0}, ("Q-PDF",)),
    ]


def env_scenarios() -> list:
    """长时环境外推的情景（框架 §5.3 要求 t_pre 与稳态值都扰动）。"""
    return [
        Scenario("tpre_9600", "情景", "恒温段起点取 9600 s（而非 14400 s）",
                 {"t_pre": 9600.0}, ("Q-A1",)),
        Scenario("Tinf_49.8", "情景", "恒温段 T∞ 取 49.8 °C", {"T_plat": 49.8},
                 ("Q-A1",)),
        Scenario("Tinf_50.3", "情景", "恒温段 T∞ 取 50.3 °C", {"T_plat": 50.3},
                 ("Q-A1",)),
        Scenario("Cinf_0.0495", "情景", "恒温段 C∞ᵉᑫ 取 0.0495", {"C_plat": 0.0495},
                 ("Q-A1",)),
        Scenario("Cinf_0.0505", "情景", "恒温段 C∞ᵉᑫ 取 0.0505", {"C_plat": 0.0505},
                 ("Q-A1",)),
    ]


def m1_rho_scenarios() -> list:
    """
    M1 潜热项对干基密度 ρ_d 的敏感性。

    [OWN-RHOD] 已声明：附录3 的 ρ(C) 与干物质守恒不兼容，
    ρ_d = ρ(C)/(1+C) 在 C=0 给 650、在 C=2.55 给 275.0，**不是常数**。
    本项目取初始状态基准 275.04 kg/m³。这里扫描它的影响，
    **目的是量化该不确定度，不是为挑一个让答案好看的取值**。
    """
    return [
        Scenario("M1_rhod_x1", "情景", "M1，ρ_d = 275.04（初始基准，主用）",
                 {"bc_mult": 1.0}, ("OWN-RHOD",)),
        Scenario("M1_rhod_x0.5", "情景", "M1，ρ_d 减半 = 137.52", {"bc_mult": 0.5},
                 ("OWN-RHOD",)),
        Scenario("M1_rhod_x0.25", "情景", "M1，ρ_d 取 1/4 = 68.76", {"bc_mult": 0.25},
                 ("OWN-RHOD",)),
        Scenario("M1_rhod_x2", "情景", "M1，ρ_d 加倍 = 550.08", {"bc_mult": 2.0},
                 ("OWN-RHOD",)),
    ]


# ==========================================================================
def run_scenarios(env_builder, scenarios, grid, t_end=None, target=0.15,
                  dt_out=900.0, base_bc=None, verbose=True) -> dict:
    """
    逐情景运行。env_builder(sc) 返回该情景的环境对象（可为 None，表示用默认）。
    """
    base_bc = base_bc or BoundaryConfig()
    rows = []
    curves = {}
    for sc in scenarios:
        bc = base_bc
        if "bc_mult" in sc.overrides:
            m = sc.overrides["bc_mult"]
            bc = replace(base_bc, latent=True, rho_d=RHO_D_INIT * m)
        env = env_builder(sc)
        horizon = t_end or 8.0 * 86400.0
        setup = Problem2Setup(
            grid=grid, env=env, bc=bc,
            # 🔴 t_end 必须显式给成**视界**。不给就落到 Problem2Setup 的默认
            #    10800 s（问题2 的 3 h），积分 3 h 就结束，永远到不了 t* ——
            #    表现是每个情景都被误报成"不可达标"（实测接受步 6659 ≈ 3 h）。
            t_end=horizon,
            x_D=sc.overrides.get("x_D", 1.0),
            x_hm=sc.overrides.get("x_hm", 1.0),
            x_h=sc.overrides.get("x_h", 1.0),
            warmup_steps=1000 if bc.latent else 0,
            warmup_dt=1.0e-3,
            label=bc.layer(),
        )
        r = _run_to_target(setup, target=target, dt_out=dt_out,
                           max_horizon=horizon)
        curves[sc.key] = (r["t"], r["Cmax"])
        row = {"key": sc.key, "kind": sc.kind, "desc": sc.desc,
               "refs": list(sc.ref_keys),
               "t_star_h": r["t_star_s"] / 3600.0 if r["reachable"] else None,
               "reachable": r["reachable"],
               "n_accepted": r["n_accepted"],
               "T_center_end": r["T_center_end"],
               "C_center_end": r["C_center_end"]}
        rows.append(row)
        if verbose:
            ts = f"{row['t_star_h']:.2f} h" if row["reachable"] else "不可达标"
            print(f"  [{sc.kind}] {sc.key:14s} t* = {ts:>10s}   "
                  f"(接受 {r['n_accepted']} 步)")
    return {"rows": rows, "curves": curves,
            "reference": {"R4-F21": get("R4-F21").detail},
            "interval_naming": {
                "情景区间": "人为设定参数范围对应的结果范围；随时可用",
                "参数扰动分位区间": "由参数分布传播得到；需明确分布假设",
                "统计置信区间": "由实测样本得到；**本工况无批次测量数据，不给出**",
            }}


# ==========================================================================
# D 的一维分位传播（参数扰动分位区间）
# ==========================================================================
D_CV_REPORTED = 0.20      # 文献 [R4-F21] 报道 D_eff 的对数正态 CV 约 15%~35%


def d_quantile_scenarios(cv: float = D_CV_REPORTED, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """
    **仅 D 一维**的分位传播。

    🔴 命名与适用范围（不得含糊）
      * 属于 **参数扰动分位区间**，不是统计置信区间；
      * 分布假设：D 的倍率 ~ 对数正态，CV 取文献报道值 20%（[R4-F21]，
        该值为**文献假设**，非本工况实测）→ 对数标准差 σ = √ln(1+cv²)；
      * **只传播 D 一个参数**，其余全部固定在题设值 —— 因此它给出的是
        **D 单独贡献**的分位区间，**不是**全参数联合不确定度。
    """
    sigma = float(np.sqrt(np.log(1.0 + cv ** 2)))
    out = []
    for q in qs:
        from scipy.stats import norm
        z = norm.ppf(q)
        m = float(np.exp(z * sigma))          # 对数正态中位数倍率为 1
        out.append(Scenario(f"D_q{q:.2f}", "参数扰动分位",
                            f"D 倍率取对数正态 {q:.0%} 分位（倍率 {m:.4f}）",
                            {"x_D": m}, ("R4-F21",)))
    return out, {"cv": cv, "sigma_lognormal": sigma, "quantiles": list(qs),
                 "assumption": "D 倍率~对数正态，CV 取文献报道 20%（非本工况实测）",
                 "scope": "仅 D 一维传播，其余参数固定在题设值"}
