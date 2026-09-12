"""
CODE-17  结果导出器（result1.xlsx）  [丙 · M0 · Day1下午]

严格按附件3 模板实测规格写出
----------------------------
    result1.xlsx
        sheet 「温度」与「水分浓度」
        A1 = 字符串 "时间\\到药材中心的距离"（含反斜杠，已与附件3 模板逐字符比对一致）
        B1:U1 = 0.0, 0.1, 0.2, …, 2.0（21 个**数值**）
        A2:A1801 = 时间 1, 2, …, 1800（**整数，从 1 起，不含 0**）
        正文 = 4 位小数

**M0/M1/M2 分层**：本写入器只写 M0（题设基线）。扩展结果另存，不得混入。

rounding 只在写盘时发生；主数组保持未舍入，供后续阈值判定（CODE-15）使用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl

from ..config import (HEADER_A1, N_DECIMALS, R_OUT_CM, SHEET_C, SHEET_T,
                      T_START)


def _fill_sheet(ws, t_int, data, decimals):
    """写一个 sheet：A1 表头、B1:U1 距离轴、A 列时间、正文数值。"""
    ws.cell(1, 1, HEADER_A1)
    for j, r_cm in enumerate(R_OUT_CM):
        ws.cell(1, 2 + j, float(r_cm))
    for i, ti in enumerate(t_int):
        ws.cell(2 + i, 1, int(ti))
        row = data[i]
        for j in range(len(R_OUT_CM)):
            ws.cell(2 + i, 2 + j, float(np.round(row[j], decimals)))


def _write_meta(wb, meta):
    if not meta:
        return
    ws = wb.create_sheet("运行信息")
    ws.cell(1, 1, "项")
    ws.cell(1, 2, "值")
    for i, (k, v) in enumerate(meta.items(), start=2):
        ws.cell(i, 1, str(k))
        ws.cell(i, 2, str(v))


def _check(t_int, arr, other=None):
    t_int = np.asarray(t_int)
    arr = np.asarray(arr, dtype=float)
    assert arr.shape[1] == len(R_OUT_CM), f"列数应为 {len(R_OUT_CM)}"
    assert np.all(np.diff(t_int) > 0), "时间列必须递增"
    assert np.all(np.isfinite(arr)), "含非有限值"
    if other is not None:
        other = np.asarray(other, dtype=float)
        assert other.shape == arr.shape, "两个场的矩阵形状不一致"
        assert np.all(np.isfinite(other)), "含非有限值"
    return t_int, arr, other


def write_result1(path, t_int, T_out, C_out, sheet_T=SHEET_T, sheet_C=SHEET_C,
                  decimals=N_DECIMALS, meta=None):
    """
    写 result1.xlsx / result2.xlsx（**两个文件结构完全相同**）。

    规格（附件3 模板实测）
    ----------------------
        sheet 「温度」与「水分浓度」
        A1 = "时间\\到药材中心的距离"（含反斜杠，已与附件3 模板逐字符比对一致）
        B1:U1 = 0.0, 0.1, …, 2.0（21 个**数值**）
        A 列 = 整数秒，**从 1 起**
        正文 = 4 位小数

    参数
    ----
    t_int : (n_t,) 整数时间（秒），严格递增
    T_out : (n_t, 21) 温度
    C_out : (n_t, 21) 含水率
    """
    t_int, T_out, C_out = _check(t_int, T_out, C_out)
    wb = openpyxl.Workbook()
    ws_T = wb.active
    ws_T.title = sheet_T
    ws_C = wb.create_sheet(sheet_C)
    for ws, data in ((ws_T, T_out), (ws_C, C_out)):
        _fill_sheet(ws, t_int, data, decimals)
    _write_meta(wb, meta)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wb.save(p)
    return p


def write_result3(path, t_int, C_out, sheet="Sheet1",
                  decimals=N_DECIMALS, meta=None):
    """
    写 result3.xlsx（**单 sheet，只含水率**）。

    规格（附件3 模板实测）
    ----------------------
        sheet 名 `Sheet1`（不是「温度」/「水分浓度」）
        A1 = "时间\\到药材中心的距离"
        A 列 = 整数秒，**从 60 起**（题面：每隔 60 s）
        正文 = 4 位小数
    末行是**达标时刻**，**不一定落在 6 h 的整数倍上**（题面表5 最后一行为
    "烘干结束时间"）。
    """
    t_int, C_out, _ = _check(t_int, C_out, None)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    _fill_sheet(ws, t_int, C_out, decimals)
    _write_meta(wb, meta)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wb.save(p)
    return p


def validate_result3(path, expect_t=None, expect_radii=R_OUT_CM,
                     decimals=N_DECIMALS) -> dict:
    """result3 的回读自检（单 sheet 版）。"""
    p = Path(path)
    wb = openpyxl.load_workbook(p, data_only=True)
    rep = {"path": str(p), "checks": {}, "ok": True}

    def chk(name, cond, detail=""):
        rep["checks"][name] = {"ok": bool(cond), "detail": detail}
        if not cond:
            rep["ok"] = False

    chk("sheet名", wb.sheetnames[0] == "Sheet1", f"实际 {wb.sheetnames}")
    ws = wb[wb.sheetnames[0]]
    chk("表头", ws.cell(1, 1).value == HEADER_A1, f"实际 {ws.cell(1,1).value!r}")
    chk("列数", ws.max_column == len(expect_radii) + 1,
        f"实际 {ws.max_column}，期望 {len(expect_radii)+1}")
    t0 = ws.cell(2, 1).value
    chk("时间起点=60", t0 == 60, f"实际 {t0}")
    if expect_t is not None:
        chk("末行时间", ws.cell(1 + len(expect_t), 1).value == int(expect_t[-1]),
            f"实际 {ws.cell(1+len(expect_t),1).value}")
        chk("行数", ws.max_row == len(expect_t) + 1,
            f"实际 {ws.max_row}，期望 {len(expect_t)+1}")
    vals = np.array([[ws.cell(2 + i, 2 + j).value
                      for j in range(len(expect_radii))]
                     for i in range(ws.max_row - 1)], dtype=float)
    chk("无NaN", np.all(np.isfinite(vals)))
    chk("数值范围", vals.min() >= -1e-9 and vals.max() <= 3.0 + 1e-9,
        f"[{vals.min():.4f}, {vals.max():.4f}]")
    return rep


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

    chk("sheets", wb.sheetnames[:2] == [SHEET_T, SHEET_C],
        f"实际 {wb.sheetnames}")

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
    return rep


def print_validation(rep):
    print(f"\n[CODE-17] 回读自检: {rep['path']}")
    for k, v in rep["checks"].items():
        mark = "OK " if v["ok"] else "FAIL"
        print(f"    [{mark}] {k:24s} {v['detail']}")
    print(f"    总体: {'通过' if rep['ok'] else '**未通过**'}")
    return rep["ok"]


# ==========================================================================
# result4 —— 末列是字符串「药材表面」，域外位置**留空**
# ==========================================================================
def write_result4(path, t_int, C_mat, C_surf, sheet="Sheet1",
                  decimals=N_DECIMALS, meta=None):
    """
    写 result4.xlsx。规格（附件3 模板实测）：

        sheet 名 `Sheet1`；A1 = "时间\到药材中心的距离"
        B1:U1 = 0, 0.1, ..., 1.9（**注意：到 1.9 为止，共 20 列**）
        V1 = 字符串 "药材表面"      ← 第 22 列，是文字不是数字
        A 列 = 整数秒，从 60 起
        正文 = 4 位小数；**域外位置留空（绝不填 0）**

    🔴 与 result2/3 的关键差别
    --------------------------
    半径 R(t) 随时间缩小，到 t 时刻只有 r ≤ R(t) 的位置有物理意义。
    模板把「药材表面」单列成第 22 列，且**随 R(t) 变化**，
    不能固定取 2 cm（分工 v3 §五 明确要求）。
    """
    t_int = np.asarray(t_int)
    C_surf = np.asarray(C_surf, dtype=float)
    assert len(t_int) == len(C_surf), "时间轴与表面值长度不一致"
    assert np.all(np.diff(t_int) > 0), "时间列必须递增"

    # 🔴 result4 的距离轴只到 **1.9 cm**（20 列），第 22 列是字符串「药材表面」。
    #    与 result1/2/3 的 21 列（0—2.0）**不同** —— 这是附件3 模板的实测规格。
    #    本项目最初沿用 R_OUT_CM 的 21 列写出 23 列，回读自检直接报「列数=23、末列名=2」。
    #    首列时间 + 20 个距离 + 表面列 = **22 列**。
    R4_CM = R_OUT_CM[:20]                      # 0.0 … 1.9
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws.cell(1, 1, HEADER_A1)
    for j, r_cm in enumerate(R4_CM):
        ws.cell(1, 2 + j, float(r_cm))
    ws.cell(1, 2 + len(R4_CM), "药材表面")
    for i, ti in enumerate(t_int):
        ws.cell(2 + i, 1, int(ti))
        for j in range(len(R4_CM)):
            v = C_mat[i][j]
            if v is None or not np.isfinite(v):
                continue                      # ← 域外留空，不写任何值
            ws.cell(2 + i, 2 + j, float(np.round(v, decimals)))
        ws.cell(2 + i, 2 + len(R4_CM), float(np.round(C_surf[i], decimals)))
    _write_meta(wb, meta)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wb.save(p)
    return p


def validate_result4(path, expect_t=None, decimals=N_DECIMALS) -> dict:
    """result4 的回读自检：末列必须是字符串、域外必须为空。"""
    p = Path(path)
    wb = openpyxl.load_workbook(p, data_only=True)
    rep = {"path": str(p), "checks": {}, "ok": True}

    def chk(name, cond, detail=""):
        rep["checks"][name] = {"ok": bool(cond), "detail": detail}
        if not cond:
            rep["ok"] = False

    chk("sheet名", wb.sheetnames[0] == "Sheet1", f"实际 {wb.sheetnames}")
    ws = wb[wb.sheetnames[0]]
    chk("表头", ws.cell(1, 1).value == HEADER_A1, f"实际 {ws.cell(1,1).value!r}")
    chk("列数=22", ws.max_column == 22, f"实际 {ws.max_column}")
    chk("末列名=药材表面", ws.cell(1, 22).value == "药材表面",
        f"实际 {ws.cell(1,22).value!r}")
    chk("时间起点=60", ws.cell(2, 1).value == 60, f"实际 {ws.cell(2,1).value}")
    if expect_t is not None:
        chk("行数", ws.max_row == len(expect_t) + 1,
            f"实际 {ws.max_row}，期望 {len(expect_t)+1}")
        chk("末行时间", ws.cell(1 + len(expect_t), 1).value == int(expect_t[-1]),
            f"实际 {ws.cell(1+len(expect_t),1).value}")
    # 域外必须为空：末行必有一部分是 None
    nblank = sum(1 for j in range(2, 22) if ws.cell(ws.max_row, j).value is None)
    chk("域外留空", nblank > 0, f"末行空格数 {nblank}")
    nzero = sum(1 for j in range(2, 22) if ws.cell(ws.max_row, j).value == 0)
    chk("域外未填0", nzero == 0, f"末行填0个数 {nzero}")
    return rep
