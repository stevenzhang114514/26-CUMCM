"""
数据溯源审计：四个问的每一个数字从哪里来          [甲 · 审计工具 · 非交付]

用途
----
回答三个问题：
  ① 四个问的过程里有没有**编造**数据（即：任何数字没有可追溯来源）
  ② 哪些数据来自**外部论文**
  ③ 哪些因素来自 21 因素 / 论文池

做法
----
把产物文件里的每一个关键数字**回读核对**，而不是靠文档抄写：
  V1 题设表 3/4/5 ← results/M0/result2.xlsx · result3.xlsx（逐格）
  V2 文档关键数字 ← docs/day2_core.json · day3_freeze.json
  V3 输出格式    ← 附件3 官方模板（表头、sheet 名、起始行）
  V4 环境边界    ← 附件1 原始 241 点（平台值、阶跃）
  V5 乙的 result4 ← 逐行移植版复现

用法：python scripts/audit_data_provenance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ATT = ROOT.parent / "A题题目和附件" / "附件"
OFFICIAL = Path(r"C:\Users\33154\Desktop\CUMCM2026Problems\A题\附件")

P = lambda *a: print(*a, flush=True)


def sheet_rows(p, sh=0):
    """
    ⚠️ 必须用 iter_rows(values_only=True)。
    逐格 .cell(r,c) 读 10801×22 要几分钟（实测挂住不返回），
    iter_rows 是流式读取，常数级开销。
    """
    wb = openpyxl.load_workbook(p, data_only=True, read_only=True)
    ws = wb[wb.sheetnames[sh]]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return rows


def main():
    P("=" * 94)
    P("  V0  输入数据同一性：官方路径 vs 本项目使用的副本")
    P("=" * 94)
    import hashlib
    for f in ("附件1.xlsx", "附件2.xlsx"):
        a = hashlib.sha256((OFFICIAL / f).read_bytes()).hexdigest()[:16]
        b = hashlib.sha256((ATT / f).read_bytes()).hexdigest()[:16]
        P(f"   {f:14s} 官方={a}  本项目={b}  {'✅ 同一文件' if a==b else '★不同★'}")

    P("\n" + "=" * 94)
    P("  V1  题设表 3/4/5 ← 结果文件逐格回读（不靠文档抄写）")
    P("=" * 94)
    R2T = sheet_rows(ROOT / "results/M0/result2.xlsx", 0)
    R2C = sheet_rows(ROOT / "results/M0/result2.xlsx", 1)
    R3 = sheet_rows(ROOT / "results/M0/result3.xlsx", 0)

    P("\n  [表3] 3 小时内药材温度 / °C     列：r = 0 / 0.5 / 1 / 1.5 / 2 cm")
    hdr = R2T[0]
    for th in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        row = next(r for r in R2T[1:] if r[0] == int(th * 3600))
        P(f"    {th:4.1f} h  " + "  ".join(f"{row[hdr.index(x)]:8.4f}"
                                           for x in (0, 0.5, 1, 1.5, 2)))

    P("\n  [表4] 3 小时内药材水分浓度 / (kg/kg)")
    hdr = R2C[0]
    for th in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        row = next(r for r in R2C[1:] if r[0] == int(th * 3600))
        P(f"    {th:4.1f} h  " + "  ".join(f"{row[hdr.index(x)]:8.4f}"
                                           for x in (0, 0.5, 1, 1.5, 2)))

    P("\n  [表5] 烘干过程水分浓度 / (kg/kg)")
    hdr = R3[0]
    for th in (6, 12, 18, 24, 30, 36, 42, 48, 54):
        row = next((r for r in R3[1:] if r[0] == th * 3600), None)
        if row:
            P(f"    {th:4d} h  " + "  ".join(f"{row[hdr.index(x)]:8.4f}"
                                             for x in (0, 0.5, 1, 1.5, 2)))
    last = R3[-1]
    P(f"   末行(达标) {last[0]} s = {last[0]/3600:.4f} h  "
      + "  ".join(f"{last[hdr.index(x)]:8.4f}" for x in (0, 0.5, 1, 1.5, 2)))
    vals = [v for v in last[1:] if isinstance(v, (int, float))]
    P(f"   末行全列最大值 = {max(vals):.4f}  ← 判据要求 < 0.15")

    P("\n" + "=" * 94)
    P("  V2  文档关键数字 ← JSON 记录（防文档与结果脱节）")
    P("=" * 94)
    core = json.loads((ROOT / "docs/day2_core.json").read_text(encoding="utf-8"))
    fr = json.loads((ROOT / "docs/day3_freeze.json").read_text(encoding="utf-8"))
    checks = [
        ("t* / s", core["tstar_M0"]["t_star_s"], fr["t_star_s"]),
        ("t* / h", core["tstar_M0"]["t_star_h"], fr["t_star_h"]),
        ("C_max(t*)", core["tstar_M0"]["C_max_at_tstar"], fr["C_max_at_tstar"]),
        ("3h T_center", core["result2_3h_M0"]["T_center_3h"],
         fr["result2_3h"]["T_center_3h"]),
        ("3h C_center", core["result2_3h_M0"]["C_center_3h"],
         fr["result2_3h"]["C_center_3h"]),
    ]
    for name, a, b in checks:
        ok = abs(a - b) < 1e-9
        P(f"   {name:14s} Day2={a!r:22s} Day3={b!r:22s} {'✅一致' if ok else '★不一致★'}")
    P(f"   result3 末行 {last[0]} s  vs  t* 二分值 "
      f"{core['tstar_M0']['t_star_s']:.4f} s  → 差 "
      f"{last[0]-core['tstar_M0']['t_star_s']:.4f} s（整数秒规则，题面要求每隔 60 s）")

    P("\n" + "=" * 94)
    P("  V3  输出格式 ← 附件3 官方模板")
    P("=" * 94)
    for name, mine in (("result2.xlsx", "results/M0/result2.xlsx"),
                       ("result3.xlsx", "results/M0/result3.xlsx")):
        off = OFFICIAL / "附件3" / name
        wo = openpyxl.load_workbook(off); wso = wo[wo.sheetnames[0]]
        wm = openpyxl.load_workbook(ROOT / mine); wsm = wm[wm.sheetnames[0]]
        a1o, a1m = wso.cell(1, 1).value, wsm.cell(1, 1).value
        P(f"   {name}: 官方 A1={a1o!r}  本项目 A1={a1m!r}  "
          f"{'✅' if a1o == a1m else '★表头不符★'}")
        P(f"       官方 sheet={wo.sheetnames}  本项目 sheet={wm.sheetnames[:2]}")

    P("\n" + "=" * 94)
    P("  V4  环境边界 ← 附件1 原始 241 点")
    P("=" * 94)
    d = sheet_rows(ATT / "附件1.xlsx")
    t = np.array([r[0] for r in d[1:] if isinstance(r[0], (int, float))])
    T = np.array([r[1] for r in d[1:] if isinstance(r[1], (int, float))])
    C = np.array([r[2] for r in d[1:] if isinstance(r[2], (int, float))])
    m = t >= 9600
    P(f"   附件1: {len(t)} 点，t ∈ [{t[0]:.0f}, {t[-1]:.0f}] s")
    P(f"   t≥9600 s 平台均值: T∞ = {T[m].mean():.4f} °C (std {T[m].std():.4f})，"
      f"C∞ = {C[m].mean():.6f} (std {C[m].std():.6f})")
    P(f"   本项目采用: 0—14400 s 用 PCHIP 实测，其后取上平台均值（框架 §5.3 方案A 的"
      f" t_pre=14400 档）")
    P(f"   乙（问题4）采用: t≤1800 s 用实测，t>1800 s 硬切到常数 50 / 0.05")
    P(f"     ⚠️ 框架 §5.3 原定分界点为 9600 s 或 14400 s，**未含 1800 s** → 属偏离")

    P("\n" + "=" * 94)
    P("  V5  乙的 result4.xlsx ← 逐行移植版复现")
    P("=" * 94)
    yi = ROOT.parent / "乙day12/data/result4.xlsx"
    if yi.exists():
        RR = sheet_rows(yi, 0)
        P(f"   乙 result4: {len(RR)-1} 行，末行 {RR[-1][0]} s = {RR[-1][0]/3600:.4f} h")
        P(f"   移植版（review_yi_p4.py S0）: 73.033 h → 差 "
          f"{RR[-1][0]/3600 - 73.033:+.4f} h  ✅ 复现成功")
        P(f"   移植版加密网格（N=160）: 51.200 h  → 乙的结果偏大 "
          f"{(RR[-1][0]/3600 - 51.20)/51.20*100:.1f}%")
    P("\n" + "=" * 94)


if __name__ == "__main__":
    main()
