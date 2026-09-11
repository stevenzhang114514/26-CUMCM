"""
CODE-17  结果导出器（result1.xlsx）  [丙 · M0 · Day1下午]

严格按附件3 模板实测规格写出
----------------------------
    result1.xlsx
        sheet 仅两个：「温度」与「水分浓度」
        A1 = 字符串 "时间\\到药材中心的距离"（含反斜杠，与模板逐字符一致）
        B1:U1 = 0.0, 0.1, 0.2, …, 2.0（21 个**数值**）
        A2:A1801 = 时间 1, 2, …, 1800（**整数，从 1 起，不含 0**）
        正文 = 4 位小数（存储舍入到 4 位 + 单元格数字格式 0.0000，两者都要）

    运行信息（meta）**不得**放进 result1.xlsx，用 write_meta() 单独存文件。

**M0/M1/M2 分层**：本写入器只写 M0（题设基线）。扩展结果另存，不得混入。

rounding 只在写盘时发生；主数组保持未舍入，供后续阈值判定（CODE-15）使用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl

from ..config import (HEADER_A1, N_DECIMALS, R_OUT_CM, SHEET_C, SHEET_T,
                      T_START)


def write_result1(path, t_int, T_out, C_out, sheet_T=SHEET_T, sheet_C=SHEET_C,
                  decimals=N_DECIMALS):
    """
    写 result1.xlsx（**仅「温度」「水分浓度」两个 sheet**）。

    参数
    ----
    t_int : (n_t,) 整数时间（秒），必须等于 T_START..T_END
    T_out : (n_t, 21) 温度
    C_out : (n_t, 21) 含水率
    """
    t_int = np.asarray(t_int)
    T_out = np.asarray(T_out, dtype=float)
    C_out = np.asarray(C_out, dtype=float)
    assert T_out.shape == C_out.shape, "温度与含水率矩阵形状不一致"
    assert T_out.shape[1] == len(R_OUT_CM), f"列数应为 {len(R_OUT_CM)}"
    assert np.all(np.diff(t_int) > 0), "时间列必须递增"

    wb = openpyxl.Workbook()
    ws_T = wb.active
    ws_T.title = sheet_T
    ws_C = wb.create_sheet(sheet_C)

    fmt = "0." + "0" * decimals
    for ws, data in ((ws_T, T_out), (ws_C, C_out)):
        ws.cell(1, 1, HEADER_A1)
        for j, r_cm in enumerate(R_OUT_CM):
            ws.cell(1, 2 + j, float(r_cm))
        for i, ti in enumerate(t_int):
            ws.cell(2 + i, 1, int(ti))
            row = data[i]
            for j in range(len(R_OUT_CM)):
                cell = ws.cell(2 + i, 2 + j, float(np.round(row[j], decimals)))
                cell.number_format = fmt   # 存储舍入之外，显示也固定 4 位小数

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wb.save(p)
    return p


def write_meta(path, meta):
    """运行信息单独存盘（不得混入 result1.xlsx）。"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "运行信息"
    ws.cell(1, 1, "项")
    ws.cell(1, 2, "值")
    for i, (k, v) in enumerate(meta.items(), start=2):
        ws.cell(i, 1, str(k))
        ws.cell(i, 2, str(v))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wb.save(p)
    return p


# ==========================================================================
# 回读自检
# ==========================================================================
def validate_output(path, expect_t=None, expect_radii=R_OUT_CM,
                    decimals=N_DECIMALS) -> dict:
    """
    重新打开写出的文件并逐项核对（**这一步不能省**）。
    返回检查报告 dict；任一项失败则 'ok' 为 False。
    """
    p = Path(path)
    wb = openpyxl.load_workbook(p, data_only=True)
    rep = {"path": str(p), "checks": {}, "ok": True}

    def chk(name, cond, detail=""):
        rep["checks"][name] = {"ok": bool(cond), "detail": detail}
        if not cond:
            rep["ok"] = False

    chk("sheets", wb.sheetnames == [SHEET_T, SHEET_C],
        f"实际 {wb.sheetnames}（必须恰好两个，运行信息须单独存）")

    for sname in (SHEET_T, SHEET_C):
        ws = wb[sname]
        n_row = ws.max_row
        n_col = ws.max_column

        hdr = ws.cell(1, 1).value
        chk(f"{sname}:表头", hdr == HEADER_A1, f"实际 {hdr!r}")

        radii = [ws.cell(1, 2 + j).value for j in range(len(expect_radii))]
        radii_ok = all(abs(float(a) - b) < 1e-12 for a, b in zip(radii, expect_radii))
        chk(f"{sname}:距离轴", radii_ok, f"前3个 {radii[:3]} 末个 {radii[-1]}")

        times = [ws.cell(2 + i, 1).value for i in range(min(n_row - 1, 5))]
        chk(f"{sname}:时间为整数", all(isinstance(t, int) for t in times),
            f"前5个 {times}")

        if expect_t is not None:
            chk(f"{sname}:时间起点", ws.cell(2, 1).value == int(expect_t[0]),
                f"实际 {ws.cell(2,1).value}")
            chk(f"{sname}:时间终点", ws.cell(1 + len(expect_t), 1).value == int(expect_t[-1]),
                f"实际 {ws.cell(1+len(expect_t),1).value}")
            chk(f"{sname}:行数", n_row == len(expect_t) + 1,
                f"实际 {n_row}，期望 {len(expect_t)+1}")
        chk(f"{sname}:列数", n_col == len(expect_radii) + 1,
            f"实际 {n_col}，期望 {len(expect_radii)+1}")

        # 数值范围与小数位
        vals = np.array([[ws.cell(2 + i, 2 + j).value for j in range(len(expect_radii))]
                         for i in range(n_row - 1)], dtype=float)
        chk(f"{sname}:无NaN", np.all(np.isfinite(vals)))
        lim = (0.0, 100.0) if sname == SHEET_T else (0.0, 3.0)
        chk(f"{sname}:数值范围", vals.min() >= lim[0] - 1e-9 and vals.max() <= lim[1] + 1e-9,
            f"[{vals.min():.4f}, {vals.max():.4f}]")
        # 小数位
        maxdec = 0
        for v in vals.ravel()[:5000]:
            s = repr(float(v))
            if "e" in s or "E" in s:
                maxdec = 99
                break
            if "." in s:
                maxdec = max(maxdec, len(s.split(".")[1]))
        chk(f"{sname}:小数位≤{decimals}", maxdec <= decimals, f"实际最多 {maxdec} 位")
        # 显示格式：题目要求"保留 4 位小数"，须固定数字格式（否则 1.5160 显示成 1.516）
        expect_fmt = "0." + "0" * decimals
        fmt_bad = sum(
            1 for i in range(0, n_row - 1, 97)
            for j in range(len(expect_radii))
            if ws.cell(2 + i, 2 + j).number_format != expect_fmt
        )
        chk(f"{sname}:数字格式{expect_fmt}", fmt_bad == 0,
            f"抽样 {len(range(0, n_row - 1, 97)) * len(expect_radii)} 格，不符 {fmt_bad} 格")
    return rep


def print_validation(rep):
    print(f"\n[CODE-17] 回读自检: {rep['path']}")
    for k, v in rep["checks"].items():
        mark = "OK " if v["ok"] else "FAIL"
        print(f"    [{mark}] {k:24s} {v['detail']}")
    print(f"    总体: {'通过' if rep['ok'] else '**未通过**'}")
    return rep["ok"]
