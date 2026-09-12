"""
论文出图的数据层                         [论文出图 · 公共模块]

**唯一职责**：把散在各成员目录里的权威数据读进来，给 `fig_*.py` 用。

铁律
----
🔴 **只读甲的权威结果**，不读任何副本：

    问题1  ../甲day1/results/M0/result1.xlsx
    问题2  ../甲day2/results/M0/result2.xlsx
    问题3  ../甲day2/results/M0/result3.xlsx
    问题4  ../甲做第四问/results/result4.xlsx
    附件1/2 ../A题题目和附件/附件/
    诊断量 ../甲day3/docs/day3_*.json、day2_*.json

⚠️ `丙day1/_superseded/M0结果副本/result4.xlsx` 是**乙的旧版**（73.03 h，网格欠分辨
   偏大 43%），**绝不可用**。本模块根本不提供指向那里的路径。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # .../26-CUMCM
FIGS = Path(__file__).resolve().parents[1] / "figs"

ATTACH = ROOT / "A题题目和附件" / "附件"
R1 = ROOT / "甲day1" / "results" / "M0" / "result1.xlsx"
R2 = ROOT / "甲day2" / "results" / "M0" / "result2.xlsx"
R3 = ROOT / "甲day2" / "results" / "M0" / "result3.xlsx"
R4 = ROOT / "甲做第四问" / "results" / "result4.xlsx"
D2 = ROOT / "甲day2" / "docs"
D3 = ROOT / "甲day3" / "docs"
D4 = ROOT / "甲做第四问" / "docs"

T_TARGET = 0.15          # kg/kg 题面阈值
R0_CM = 2.0              # 初始半径 cm


# ==========================================================================
def load_json(p) -> dict:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"缺数据文件：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _read_xlsx(path, sheet=None):
    """
    返回 (header, data ndarray)。空单元格为 NaN。

    ⚠️ 必须用 `iter_rows(values_only=True)`（按行流式取），
    不要用 `ws.cell(r, c)` 逐格取 —— result2.xlsx 是 10801×22 ≈ 24 万个格子，
    逐格访问要几十秒，流式取不到 1 秒。
    """
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet] if sheet else wb[wb.sheetnames[0]]
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    rows = [list(r) for r in it]
    wb.close()
    data = np.array([[np.nan if v is None else float(v) for v in row]
                     for row in rows], dtype=float)
    return hdr, data


# ==========================================================================
# 题面结果
# ==========================================================================
def result1():
    """
    问题1。返回 dict：t(1800,), r_cm(21,), T(1800,21), C(1800,21)。
    sheet「温度」与「水分浓度」。
    """
    hT, T = _read_xlsx(R1, "温度")
    hC, C = _read_xlsx(R3 if False else R1, "水分浓度")
    r_cm = np.array([float(v) for v in hT[1:]], float)
    return {"t": T[:, 0], "r_cm": r_cm, "T": T[:, 1:], "C": C[:, 1:],
            "sheet_T": hT, "sheet_C": hC}


def result2():
    """
    问题2（3 h 正式版，1 s × 0.1 cm）。结构同 result1。
    """
    hT, T = _read_xlsx(R2, "温度")
    hC, C = _read_xlsx(R2, "水分浓度")
    r_cm = np.array([float(v) for v in hT[1:]], float)
    return {"t": T[:, 0], "r_cm": r_cm, "T": T[:, 1:], "C": C[:, 1:],
            "sheet_T": hT, "sheet_C": hC}


def result3():
    """
    问题3。单 sheet，只含水率；末行是达标时刻（不必落在 60 s 网格上）。
    返回 t(s)、r_cm、C，(n,21)。
    """
    hdr, D = _read_xlsx(R3, "Sheet1")
    r_cm = np.array([float(v) for v in hdr[1:]], float)
    return {"t": D[:, 0], "r_cm": r_cm, "C": D[:, 1:], "sheet": hdr}


def result4():
    """
    问题4。单 sheet，末列是**字符串**「药材表面」，域外为空 → NaN。
    返回 t(s)、r_cm（20 个，0–1.9）、C（域外 NaN）、C_surf、以及逐行半径
    （半径不在文件里，由 C 的有效列数推不出；调用方需另用 radius()）。
    """
    hdr, D = _read_xlsx(R4, "Sheet1")
    # 末列是「药材表面」字符串 → 读进来是 NaN，单独取
    r_cm = np.array([float(v) for v in hdr[1:-1]], float)
    C = D[:, 1:1 + len(r_cm)]
    C_surf = D[:, 1 + len(r_cm)]
    return {"t": D[:, 0], "r_cm": r_cm, "C": C, "C_surf": C_surf, "sheet": hdr}


# ==========================================================================
# 题目附件
# ==========================================================================
def attach1():
    """附件1：环境温度与水分浓度，241 点，0–14400 s，步长 60 s。"""
    import pandas as pd
    df = pd.read_excel(ATTACH / "附件1.xlsx")
    cols = {c.strip(): c for c in df.columns}
    return {"t": df[cols["时间"]].to_numpy(float),
            "T": df[cols["温度"]].to_numpy(float),
            "C": df[cols["水分浓度"]].to_numpy(float)}


def attach2():
    """附件2：半径随时间，145 点，0–259200 s，R 由 2.000 单调降到 1.198 cm。"""
    import pandas as pd
    df = pd.read_excel(ATTACH / "附件2.xlsx")
    cols = {c.strip(): c for c in df.columns}
    tk = [c for c in cols if "时间" in c][0]
    rk = [c for c in cols if "半径" in c][0]
    return {"t": df[cols[tk]].to_numpy(float), "R_cm": df[cols[rk]].to_numpy(float)}


def radius_interp():
    """附件2 的 PCHIP 插值器（与问题4 求解器同款）。"""
    from scipy.interpolate import PchipInterpolator
    a = attach2()
    return PchipInterpolator(a["t"], a["R_cm"] / 100.0, extrapolate=True)


# ==========================================================================
# 环境插值双版本（丙的产物：faithful / smoothed）
# ==========================================================================
def env_interp_csv():
    """
    `丙day1/data/env_interp.csv` 与 `env_interp_smooth.csv`。
    列：时间, 温度, 水分浓度（由 code02 生成）。缺文件时返回 None。
    """
    import pandas as pd
    p1 = ROOT / "丙day1" / "data" / "env_interp.csv"
    p2 = ROOT / "丙day1" / "data" / "env_interp_smooth.csv"
    if not (p1.exists() and p2.exists()):
        return None
    d1, d2 = pd.read_csv(p1), pd.read_csv(p2)
    return {"raw": d1, "smooth": d2,
            "cols": list(d1.columns)}


# ==========================================================================
# 诊断量
# ==========================================================================
def day3_conv():
    """CODE-19 收敛阶（空间/时间分列）。"""
    return load_json(D3 / "day3_conv.json")


def day2_conv():
    return load_json(D2 / "day2_conv.json")


def day2_core():
    return load_json(D2 / "day2_core.json")


def day2_scen():
    return load_json(D2 / "day2_scen.json")


def sens_param():
    """参数灵敏度 ±5/10/20% + 大扰动档（28 次运行）。"""
    return load_json(D3 / "day3_sens_param.json")


def robust_data():
    """数据扰动稳健性（63 次运行）。"""
    return load_json(D3 / "day3_robust_data.json")


def p4_paper_table6():
    return load_json(D4 / "p4_paper_table6.json")


def p4_paper_fixb():
    return load_json(D4 / "p4_paper_fixB.json")


def p4_time():
    """问题4 时间收敛（Δt=30/10/3/1 s）。"""
    return load_json(D4 / "p4_time.json")


def core_npz(layer="M0", full=True):
    """甲day2/甲day3 的核心场 npz（含 C_center / C_surf / C_max / t_diag）。"""
    p = ROOT / "甲day3" / "data" / "processed" / \
        f"core_{layer}_{'full' if full else '3h'}.npz"
    return np.load(p)


# ==========================================================================
# 常量（论文正文引用的口径）
# ==========================================================================
TSTAR_H = 57.4889          # 问题3 达标时间 h（冻结值）
TSTAR_S = 206959.9521      # s（未取整）
TDRY_H = 51.090506         # 问题4 t_dry h（Δt=1 s 收敛口径）
TDRY_S = 51.090506 * 3600.0
TSTAR_H_NOTE = "57.49 h（论文报告 2 位小数）"
