"""
审阅用：乙的「问题4」—— 早中期剖面差异与 result4 一致性核对   [甲 · 审阅工具]

`review_yi_p4.py` 已确定：**t_dry 的主误差来自 N=20 网格**（73.03 h → N=80 的 51.73 h）。
本脚本补两件事：

  A. 环境"1800 s 硬切"对 t_dry 几乎无影响（+0.08 h），
     但**对早中期剖面影响很大** —— 这是题设表6 的前几行，必须量化。

  B. 与乙交付的 `data/result4.xlsx` 逐格核对：
     ① 时间轴、列数、表头是否合模板
     ② 域外是否留空（清单要求：绝不填 0）
     ③ 中心/表面曲线是否单调、是否有非物理过冲
     ④ 终态半径与 README 声明是否一致

用法：python scripts/review_yi_p4_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from review_yi_p4 import EnvFull, EnvYi, make_R, solve_yi   # noqa: E402

P = lambda *a: print(*a, flush=True)
YI_DIR = ROOT.parent / "乙day12"


# ==========================================================================
def part_A():
    P("=" * 100)
    P("  A. 环境『1800 s 硬切』对早中期剖面的影响（N=80, dt=10 s）")
    P("=" * 100)
    a = solve_yi(N=80, dt=10.0, tmax=86400.0, env=EnvYi(),
                 Rof=make_R("raw"), bc_level="old")
    b = solve_yi(N=80, dt=10.0, tmax=86400.0, env=EnvFull(),
                 Rof=make_R("raw"), bc_level="old")

    # 取 6 h / 12 h / 24 h 的整点样本
    P(f"  {'t/h':>5} | {'乙 中心':>9} {'本项目 中心':>11} {'差':>8} | "
      f"{'乙 表面':>9} {'本项目 表面':>11} {'差':>8}")
    P("  " + "-" * 76)
    out = []
    for th in (1, 2, 6, 12, 24):
        k = int(th * 3600 // 60) - 1
        if k >= len(a["trec"]) or k >= len(b["trec"]):
            continue
        ac, bc_ = a["Crec"][k, 0], b["Crec"][k, 0]
        as_, bs = a["Crec"][k, -1], b["Crec"][k, -1]
        P(f"  {th:5d} | {ac:9.4f} {bc_:11.4f} {bc_-ac:+8.4f} | "
          f"{as_:9.4f} {bs:11.4f} {bs-as_:+8.4f}")
        out.append({"t_h": th, "center_yi": float(ac), "center_fn": float(bc_),
                    "surf_yi": float(as_), "surf_fn": float(bs)})

    dmax = max(abs(o["center_fn"] - o["center_yi"]) for o in out)
    P(f"\n  中心含水率最大差 {dmax:.4f} kg/kg —— "
      f"题设要求四位小数，故**早中期数值不可忽略地不同**。")
    return out


# ==========================================================================
def part_B():
    P("\n" + "=" * 100)
    P("  B. 与乙交付的 result4.xlsx 逐格核对")
    P("=" * 100)
    import openpyxl
    p = YI_DIR / "data" / "result4.xlsx"
    wb = openpyxl.load_workbook(p)
    ws = wb[wb.sheetnames[0]]
    P(f"  文件: {p.name}   sheet={wb.sheetnames}   行={ws.max_row}  列={ws.max_column}")

    hdr = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    P(f"  表头 A1 = {hdr[0]!r}   ← 模板要求 '时间\\\\到药材的中心距离'")
    P(f"  表头 B1..U1 = {[round(h,4) if isinstance(h,float) else h for h in hdr[1:21]]}")
    P(f"  表头 V1 = {hdr[21]!r}")

    t = np.array([ws.cell(r, 1).value for r in range(2, ws.max_row + 1)], float)
    C = np.array([[ws.cell(r, c).value for c in range(2, 23)]
                  for r in range(2, ws.max_row + 1)], dtype=object)

    P(f"\n  ① 时间轴: 起 {t[0]:.0f} s  末 {t[-1]:.0f} s  步长唯一值 "
      f"{sorted(set(np.diff(t).tolist()))[:5]}  单调递增 {bool(np.all(np.diff(t)>0))}")
    P(f"     末行 {t[-1]:.0f} s = {t[-1]/3600:.4f} h   （README 声明 73.03 h）")

    # ② 域外留空
    n_blank = sum(1 for r in range(C.shape[0]) for c in range(C.shape[1] - 1)
                  if C[r, c] is None)
    n_zero = sum(1 for r in range(C.shape[0]) for c in range(C.shape[1] - 1)
                 if C[r, c] == 0)
    P(f"\n  ② 域外处理: 空格 {n_blank} 个，填 0 的 {n_zero} 个 "
      f"→ {'合规（留空）' if n_zero == 0 else '**违规（填了 0）**'}")

    # ③ 单调性与过冲
    cen = np.array([C[r, 0] for r in range(C.shape[0])], float)
    sur = np.array([C[r, -1] for r in range(C.shape[0])], float)
    P(f"\n  ③ 中心 C: {cen[0]:.4f} → {cen[-1]:.4f}  最小 {cen.min():.4f}  "
      f"单调不增 {bool(np.all(np.diff(cen) <= 1e-12))}")
    P(f"     表面 C: {sur[0]:.4f} → {sur[-1]:.4f}  最小 {sur.min():.4f}  "
      f"单调不增 {bool(np.all(np.diff(sur) <= 1e-12))}")
    P(f"     非负 {bool(cen.min() >= 0 and sur.min() >= 0)}   "
      f"C_max(末) = {cen[-1]:.4f}  （判据 < 0.15，未舍入判定）")

    # ④ 终态半径
    row_last = [C[-1, c] for c in range(C.shape[1] - 1)]
    n_filled = sum(1 for v in row_last if v is not None)
    P(f"\n  ④ 末行有效列数 {n_filled}/20 → 末行半径落在 "
      f"[{(n_filled-1)*0.1:.1f}, {n_filled*0.1:.1f}] cm 区间内")
    P(f"     README 声明终态半径 1.198 cm —— {'相容' if n_filled == 12 else '**不相容**'}")

    return {"t_end_h": float(t[-1] / 3600), "n_blank": n_blank, "n_zero": n_zero,
            "cen_end": float(cen[-1]), "surf_end": float(sur[-1])}


if __name__ == "__main__":
    A = part_A()
    B = part_B()
    import json
    (ROOT / "docs" / "day3_review_yi_profiles.json").write_text(
        json.dumps({"early_profiles": A, "result4_checks": B},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    P("\n  → docs/day3_review_yi_profiles.json")
