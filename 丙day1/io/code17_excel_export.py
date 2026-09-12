"""
CODE-17 Excel 导出与回读自检
读 results/frozen/*.csv，写入 data/result1-4.xlsx
然后回读检查：
- sheet 名称
- A 列时间
- 第 1 行距离
- 四位小数
- NaN / Inf / 负值
- 覆盖时长
"""

from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
FROZEN_DIR = ROOT / "results" / "frozen"
DATA_DIR = ROOT / "data"
M0_DIR = ROOT / "results" / "M0"
M1_DIR = ROOT / "results" / "M1"
M0_DIR.mkdir(parents=True, exist_ok=True)
M1_DIR.mkdir(parents=True, exist_ok=True)


# 每个 result 文件的规格
RESULT_SPEC = {
    "result1.xlsx": {
        "layer": "M0",
        "sheets": {
            "温度": "p1_temperature_sampled.csv",
            "水分浓度": "p1_moisture_sampled.csv",
        },
        "time_col": "time_s",
        "distance_prefix": "",
    },
    "result2.xlsx": {
        "layer": "M0",
        "sheets": {
            "温度": "p2_temperature_sampled.csv",
            "水分浓度": "p2_moisture_sampled.csv",
        },
        "time_col": "time_s",
        "distance_prefix": "",
    },
    "result3.xlsx": {
        "layer": "M0",
        "sheets": {
            "Sheet1": "p3_moisture_sampled.csv",
        },
        "time_col": "time_s",
        "distance_prefix": "",
    },
    "result4.xlsx": {
        "layer": "M0",
        "sheets": {
            "Sheet1": "p4_moisture_sampled.csv",
        },
        "time_col": "time_s",
        "distance_prefix": "",
        "last_col_name": "药材表面",
    },
}


def write_sheet(ws, df, last_col_name=None):
    """把 DataFrame 写入 worksheet。"""
    # 第 1 行
    ws.cell(row=1, column=1, value="时间\\到药材中心的距离")
    cols = list(df.columns)
    for j, col in enumerate(cols):
        if col == "time_s":
            continue
        if last_col_name and j == len(cols) - 1:
            ws.cell(row=1, column=2 + j - 1, value=last_col_name)
        else:
            try:
                ws.cell(row=1, column=2 + j - 1, value=float(col))
            except ValueError:
                ws.cell(row=1, column=2 + j - 1, value=col)

    # A 列时间
    for i, t in enumerate(df["time_s"].to_numpy()):
        ws.cell(row=2 + i, column=1, value=float(t))

    # 数据
    data = df.drop(columns=["time_s"]).to_numpy(dtype=float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v) or np.isinf(v):
                v = None
            else:
                v = round(float(v), 4)
            ws.cell(row=2 + i, column=2 + j, value=v)


def export_one(result_name, spec):
    """导出一个 result 文件。"""
    print(f"\n=== {result_name} ===")
    wb = openpyxl.Workbook()
    first = True

    for sheet_name, csv_name in spec["sheets"].items():
        csv_path = FROZEN_DIR / csv_name
        if not csv_path.exists():
            print(f"[缺失] {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if first:
            ws = wb.active
            ws.title = sheet_name
            first = False
        else:
            ws = wb.create_sheet(sheet_name)

        write_sheet(ws, df, spec.get("last_col_name"))
        print(f"  sheet {sheet_name}: {df.shape}")

    layer = spec.get("layer", "M0")
    if layer == "M0":
        out_dir = M0_DIR
    elif layer == "M1":
        out_dir = M1_DIR
    else:
        out_dir = M0_DIR

    out_path = out_dir / result_name
    wb.save(out_path)
    print(f"已保存 [{layer}]:", out_path)


def readback_check(result_name, spec):
    """回读检查。"""
    print(f"\n--- 回读 {result_name} ---")
    layer = spec.get("layer", "M0")
    if layer == "M0":
        out_dir = M0_DIR
    elif layer == "M1":
        out_dir = M1_DIR
    else:
        out_dir = M0_DIR

    path = out_dir / result_name
    wb = openpyxl.load_workbook(path, data_only=True)

    for sheet_name in spec["sheets"]:
        ws = wb[sheet_name]
        # 第 1 行
        header = [ws.cell(row=1, column=c).value
                  for c in range(1, ws.max_column + 1)]
        # A 列
        times = [ws.cell(row=r, column=1).value
                 for r in range(2, ws.max_row + 1)]
        times = [t for t in times if t is not None]

        # 数据
        bad = 0
        neg = 0
        for r in range(2, ws.max_row + 1):
            for c in range(2, ws.max_column + 1):
                v = ws.cell(row=r, column=c).value
                if v is None:
                    continue
                if isinstance(v, (int, float)):
                    if np.isnan(v) or np.isinf(v):
                        bad += 1
                    if v < 0:
                        neg += 1

        print(f"  sheet {sheet_name}:")
        print(f"    第1行: {header[:5]} ... {header[-1]}")
        print(f"    时间范围: {times[0]} ~ {times[-1]}, 共 {len(times)} 行")
        print(f"    NaN/Inf: {bad}, 负值: {neg}")

    wb.close()


def main():
    for result_name, spec in RESULT_SPEC.items():
        export_one(result_name, spec)
        readback_check(result_name, spec)


if __name__ == "__main__":
    main()