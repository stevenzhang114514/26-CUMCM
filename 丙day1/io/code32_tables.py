"""
CODE-32 题设表 1-6 生成
- 表1/2：问题1（result1.xlsx）
- 表3/4：问题2（result2.xlsx）
- 表5：问题3（result3.xlsx）
- 表6：问题4（result4.xlsx，等乙）
输出：
    report/表1_温度.csv
    report/表2_水分浓度.csv
    report/表3_温度.csv
    report/表4_水分浓度.csv
    report/表5_水分浓度.csv
"""

from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M0_DIR = ROOT / "results" / "M0"
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 问题1：t = 100, 300, 600, 900, 1200, 1500, 1800 s
TABLE1_TIMES = [100, 300, 600, 900, 1200, 1500, 1800]
# 问题2：t = 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 h
TABLE34_TIMES_H = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# 问题3：每 6 h
TABLE5_TIMES_H = [6, 12, 18, 24, 30, 36, 42, 48, 54]

TABLE_DIST = [0.0, 0.5, 1.0, 1.5, 2.0]


def load_sheet(xlsx_path, sheet_name):
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]
    header = [ws.cell(row=1, column=c).value
              for c in range(1, ws.max_column + 1)]
    rows = []
    for r in range(2, ws.max_row + 1):
        row = [ws.cell(row=r, column=c).value
               for c in range(1, ws.max_column + 1)]
        rows.append(row)
    wb.close()
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={header[0]: "time_s"})
    return df


def build_table(df, times_s, name, time_col_name):
    t = df["time_s"].to_numpy(dtype=float)
    r_cols = [c for c in df.columns if c != "time_s"]

    rows = []
    for tt in times_s:
        idx = np.argmin(np.abs(t - tt))
        row = {time_col_name: tt}
        for dd in TABLE_DIST:
            col = min(r_cols, key=lambda c: abs(float(c) - dd))
            v = df.iloc[idx][col]
            row[f"{dd:.1f}cm"] = f"{float(v):.4f}"
        rows.append(row)

    out = pd.DataFrame(rows)
    print(f"\n=== {name} ===")
    print(out.to_string(index=False))
    return out

def build_table6(df, times_s, name, time_col_name):
    """表6 专用：末列是 '药材表面'，不是固定 2 cm。"""
    t = df["time_s"].to_numpy(dtype=float)
    r_cols = [c for c in df.columns if c != "time_s"]

    # 先分离：能转 float 的列 + 字符串列
    num_cols = []
    str_cols = []
    for c in r_cols:
        try:
            float(c)
            num_cols.append(c)
        except (ValueError, TypeError):
            str_cols.append(c)

    # 找 0, 0.5, 1, 1.5
    fixed_cols = []
    for dd in [0.0, 0.5, 1.0, 1.5]:
        col = min(num_cols, key=lambda c: abs(float(c) - dd))
        fixed_cols.append(col)

    # 最后一列是"药材表面"
    surface_col = None
    for c in str_cols:
        if str(c).strip() == "药材表面":
            surface_col = c
            break
    if surface_col is None:
        surface_col = r_cols[-1]

    rows = []
    for tt in times_s:
        idx = np.argmin(np.abs(t - tt))
        row = {time_col_name: tt}
        for dd, col in zip([0.0, 0.5, 1.0, 1.5], fixed_cols):
            v = df.iloc[idx][col]
            row[f"{dd:.1f}cm"] = f"{float(v):.4f}"
        v = df.iloc[idx][surface_col]
        row["药材表面"] = f"{float(v):.4f}"
        rows.append(row)

    out = pd.DataFrame(rows)
    print(f"\n=== {name} ===")
    print(out.to_string(index=False))
    return out


def main():
    # 表1/2：问题1
    result1 = M0_DIR / "result1.xlsx"
    if result1.exists():
        df_T = load_sheet(result1, "温度")
        t1 = build_table(df_T, TABLE1_TIMES, "表1 温度", "时间/s")
        t1.to_csv(REPORT_DIR / "表1_温度.csv", index=False, encoding="utf-8-sig")

        df_C = load_sheet(result1, "水分浓度")
        t2 = build_table(df_C, TABLE1_TIMES, "表2 水分浓度", "时间/s")
        t2.to_csv(REPORT_DIR / "表2_水分浓度.csv", index=False, encoding="utf-8-sig")

    # 表3/4：问题2
    result2 = M0_DIR / "result2.xlsx"
    if result2.exists():
        df_T = load_sheet(result2, "温度")
        t3 = build_table(df_T, [int(h * 3600) for h in TABLE34_TIMES_H],
                         "表3 温度", "时间/s")
        t3.to_csv(REPORT_DIR / "表3_温度.csv", index=False, encoding="utf-8-sig")

        df_C = load_sheet(result2, "水分浓度")
        t4 = build_table(df_C, [int(h * 3600) for h in TABLE34_TIMES_H],
                         "表4 水分浓度", "时间/s")
        t4.to_csv(REPORT_DIR / "表4_水分浓度.csv", index=False, encoding="utf-8-sig")

    # 表5：问题3
    result3 = M0_DIR / "result3.xlsx"
    if result3.exists():
        df_C = load_sheet(result3, "Sheet1")
        # 6, 12, ..., 54 h
        t5 = build_table(df_C, [int(h * 3600) for h in TABLE5_TIMES_H],
                         "表5 水分浓度", "时间/s")
        # 补最后一行：烘干结束时间
        t = df_C["time_s"].to_numpy(dtype=float)
        t_end = t[-1]
        idx_end = len(t) - 1
        row_end = {"时间/s": int(t_end)}
        for dd in TABLE_DIST:
            col = min([c for c in df_C.columns if c != "time_s"],
                      key=lambda c: abs(float(c) - dd))
            v = df_C.iloc[idx_end][col]
            row_end[f"{dd:.1f}cm"] = f"{float(v):.4f}"
        t5 = pd.concat([t5, pd.DataFrame([row_end])], ignore_index=True)
        t5.to_csv(REPORT_DIR / "表5_水分浓度.csv", index=False, encoding="utf-8-sig")
        print("表5 最后一行（烘干结束时间）:")
        print(t5.tail(1).to_string(index=False))

    # 表6：问题4
    result4 = M0_DIR / "result4.xlsx"
    if result4.exists():
        df_C = load_sheet(result4, "Sheet1")
        t6 = build_table6(df_C, [int(h * 3600) for h in TABLE5_TIMES_H],
                          "表6 水分浓度（收缩）", "时间/s")
        t6.to_csv(REPORT_DIR / "表6_水分浓度.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()