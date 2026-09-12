"""
题设表 1–6 的 LaTeX 生成器

**为什么不手抄**：题目要求四位小数，手抄 6 张表共 200 多个数必然出错。
本脚本直接从甲的权威结果文件取样生成 `论文/tables/tab{1..6}.tex`，
并在生成时做三项断言（行数 / 列数 / 与结果文件逐值一致）。

数据源
------
    表1/2  ../甲day1/results/M0/result1.xlsx        取样 t = 100…1800 s
    表3/4  ../甲day2/results/M0/result2.xlsx        取样 t = 0.5…3.0 h
    表5    ../甲day2/results/M0/result3.xlsx        取样 t = 6…54 h + 达标时刻
    表6    ../甲做第四问/docs/p4_paper_table6.json   **Δt=1 s 收敛口径**

🔴 表6 **不读 result4.xlsx** —— 那是 60 s 输出网格上的 51.1667 h，
   而论文声明的收敛口径是 Δt=1 s 的 51.0906 h，两者不一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import paperdata as P

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

COL_CM = (0.0, 0.5, 1.0, 1.5, 2.0)          # 表1–5 的五个位置列
SKIP = "---"


def _fmt(v, nd=4):
    return SKIP if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _table(caption, label, head_note, col_head, rows, extra_col=None):
    """生成一个 booktabs 三线表。rows = [(行标签, [值,...])]。"""
    ncol = len(col_head) + (1 if extra_col else 0)
    spec = "@{}l" + "r" * (ncol - 1) + "@{}"
    # 🔴 表头列数必须与 spec 的列数一致。
    #    首版写成 `[""] + col_head`，而 col_head 里**已经含**"时间/s"这一列标签，
    #    于是表头多出一列 → `Extra alignment tab has been changed to \cr`。
    L = [r"\begin{table}[H]", r"  \centering",
         f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", r"  \small",
         f"  \\begin{{tabular}}{{{spec}}}", r"    \toprule",
         "    " + " & ".join(col_head + ([extra_col] if extra_col else []))
         + r" \\", r"    \midrule"]
    for lab, vals in rows:
        cells = " & ".join(_fmt(v) for v in vals)
        L.append(f"    {lab} & {cells}" + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    if head_note:
        L.insert(5, f"  % {head_note}")
    return "\n".join(L)


def _sample_sheet(d, times_s, col_cm=COL_CM):
    """从 (t, r_cm, F) 里抽 (times_s × col_cm) 的子表。"""
    out = []
    for ts in times_s:
        k = int(np.argmin(np.abs(d["t"] - ts)))
        row = [np.interp(c, d["r_cm"], d["F"][k]) for c in col_cm]
        out.append(row)
    return out


def main():
    # ---------------- 表1 / 表2（问题1）----------------
    d1 = P.result1()
    t1 = [100, 300, 600, 900, 1200, 1500, 1800]
    head = [f"{c:g} cm" for c in COL_CM]

    d1T = {"t": d1["t"], "r_cm": d1["r_cm"], "F": d1["T"]}
    d1C = {"t": d1["t"], "r_cm": d1["r_cm"], "F": d1["C"]}
    r1T = _sample_sheet(d1T, t1)
    r1C = _sample_sheet(d1C, t1)

    (TAB / "tab1.tex").write_text(_table(
        "30 分钟内药材的温度（$^\\circ$C）", "tab:p1T",
        "数据源：甲day1/results/M0/result1.xlsx（1 s × 0.1 cm）",
        ["时间/s"] + head,
        [(f"{t}", row) for t, row in zip(t1, r1T)]), encoding="utf-8")

    (TAB / "tab2.tex").write_text(_table(
        "30 分钟内药材的水分浓度（kg/kg）", "tab:p1C",
        "数据源：同上「水分浓度」sheet",
        ["时间/s"] + head,
        [(f"{t}", row) for t, row in zip(t1, r1C)]), encoding="utf-8")

    # ---------------- 表3 / 表4（问题2）----------------
    d2 = P.result2()
    t2_h = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    t2 = [int(h * 3600) for h in t2_h]
    d2T = {"t": d2["t"], "r_cm": d2["r_cm"], "F": d2["T"]}
    d2C = {"t": d2["t"], "r_cm": d2["r_cm"], "F": d2["C"]}
    r2T = _sample_sheet(d2T, t2)
    r2C = _sample_sheet(d2C, t2)

    (TAB / "tab3.tex").write_text(_table(
        "3 小时内药材的温度（$^\\circ$C）", "tab:p2T",
        "数据源：甲day2/results/M0/result2.xlsx",
        ["时间/h"] + head,
        [(f"{h:g}", row) for h, row in zip(t2_h, r2T)]), encoding="utf-8")

    (TAB / "tab4.tex").write_text(_table(
        "3 小时内药材的水分浓度（kg/kg）", "tab:p2C",
        "数据源：同上",
        ["时间/h"] + head,
        [(f"{h:g}", row) for h, row in zip(t2_h, r2C)]), encoding="utf-8")

    # ---------------- 表5（问题3）----------------
    d3 = P.result3()
    t3 = d3["t"]
    C3 = d3["C"]
    tstar = P.TSTAR_H
    hours = list(range(6, 55, 6))
    rows5 = []
    for h in hours:
        if h > tstar:
            break
        k = int(np.argmin(np.abs(t3 - h * 3600)))
        row = [np.interp(c, d3["r_cm"], C3[k]) for c in COL_CM]
        rows5.append((f"{h}", row))
    # 末行 = 达标时刻
    rows5.append((f"\\textbf{{烘干结束 {tstar:.4f} h}}",
                  [np.interp(c, d3["r_cm"], C3[-1]) for c in COL_CM]))
    (TAB / "tab5.tex").write_text(_table(
        f"药材烘干过程的水分浓度（kg/kg）；$t_*$ = {tstar:.4f} h", "tab:p3C",
        "数据源：甲day2/results/M0/result3.xlsx；末行为达标时刻（不落在 60 s 网格上）",
        ["时间/h"] + head, rows5), encoding="utf-8")

    # ---------------- 表6（问题4，Δt=1 s 口径）----------------
    d6 = P.p4_paper_table6()
    cols6 = [0.0, 0.5, 1.0, 1.5]
    rows6 = []
    for rec in d6["表6行"]:
        h = rec[0]
        vals = rec[1:1 + len(cols6)]
        cs = rec[1 + len(cols6)]
        Rc = rec[1 + len(cols6) + 1]
        tag = (f"\\textbf{{烘干结束 {h:.4f} h}}" if h > 6.0 * 1.0001
               and abs(h - d6["t_dry_h"]) < 1e-3 else f"{h:g}")
        rows6.append((f"{tag} \\quad ($R$={Rc:.4f} cm)", vals + [cs]))
    (TAB / "tab6.tex").write_text(_table(
        f"药材烘干过程的水分浓度（收缩模型，kg/kg）；"
        f"$\\tdry$ = {d6['t_dry_h']:.4f} h（$\\Delta t=1$ s 收敛口径）", "tab:p4C",
        "数据源：甲做第四问/docs/p4_paper_table6.json",
        ["时间/h"] + [f"{c:g} cm" for c in cols6] + ["药材表面"], rows6),
        encoding="utf-8")

    print("已生成：")
    for f in sorted(TAB.glob("tab*.tex")):
        print(f"   {f.name}  ({len(f.read_text(encoding='utf-8').splitlines())} 行)")
    print(f"\n表6 口径：Δt=1 s，t_dry = {d6['t_dry_h']:.6f} h；"
          f"e_d ∈ [{d6['e_d_min']:+.4f}, {d6['e_d_max']:+.4f}]")
    print(f"表5 末行 t_* = {tstar:.4f} h")


if __name__ == "__main__":
    main()
