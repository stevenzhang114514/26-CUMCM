"""
检查 result1-4.xlsx 是否符合题目模板。
- sheet 名称
- A 列起点
- 第 1 行表头
- 数据行数
- 小数位
- NaN / Inf / 负值
"""

from pathlib import Path
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
M0_DIR = ROOT / "results" / "M0"

REQUIRED = {
    "result1.xlsx": {
        "sheets": ["温度", "水分浓度"],
        "time_start": 1,
        "last_col": 2.0,
        "last_col_name": None,
    },
    "result2.xlsx": {
        "sheets": ["温度", "水分浓度"],
        "time_start": 1,
        "last_col": 2.0,
        "last_col_name": None,
    },
    "result3.xlsx": {
        "sheets": ["Sheet1"],
        "time_start": 60,
        "last_col": 2.0,
        "last_col_name": None,
    },
    "result4.xlsx": {
        "sheets": ["Sheet1"],
        "time_start": 60,
        "last_col": 2.0,
        "last_col_name": "药材表面",
    },
}


def check_one(name, spec):
    path = M0_DIR / name
    print(f"\n=== {name} ===")
    if not path.exists():
        print(f"[缺失] {path}")
        return

    wb = openpyxl.load_workbook(path, data_only=True)

    for s in spec["sheets"]:
        if s not in wb.sheetnames:
            print(f"[错误] 缺 sheet: {s}")
            continue

        ws = wb[s]
        print(f"[OK] sheet {s}: {ws.max_row} 行 × {ws.max_column} 列")

        # A 列
        times = [ws.cell(row=r, column=1).value
                 for r in range(2, ws.max_row + 1)]
        times = [t for t in times if t is not None]
        if times:
            print(f"     A 列: {times[0]} → {times[-1]}, 共 {len(times)} 行")
            if times[0] != spec["time_start"]:
                print(f"     [警告] A 列起点应为 {spec['time_start']}，实际 {times[0]}")

        # 第 1 行
        header = [ws.cell(row=1, column=c).value
                  for c in range(1, ws.max_column + 1)]
        print(f"     第 1 行: {header[:5]} ... {header[-1]}")

        # 检查末列
        last = ws.cell(row=1, column=ws.max_column).value
        if spec["last_col_name"]:
            if last != spec["last_col_name"]:
                print(f"     [警告] 末列应为 '{spec['last_col_name']}'，实际 '{last}'")
        else:
            if last != spec["last_col"]:
                print(f"     [警告] 末列应为 {spec['last_col']}，实际 {last}")

        # 数据检查
        nan_count = 0
        neg_count = 0
        for r in range(2, ws.max_row + 1):
            for c in range(2, ws.max_column + 1):
                v = ws.cell(row=r, column=c).value
                if v is None:
                    continue
                if isinstance(v, (int, float)):
                    if np.isnan(v) or np.isinf(v):
                        nan_count += 1
                    if v < 0:
                        neg_count += 1
        print(f"     NaN/Inf: {nan_count}, 负值: {neg_count}")
        print(f"     [完成] sheet {s} 检查完毕")
    wb.close()


def main():
    for name, spec in REQUIRED.items():
        check_one(name, spec)


if __name__ == "__main__":
    main()