"""
CODE-31 输出模板解析器
读取 result1-4.xlsx，输出每个 sheet 的规格：
- sheet 名称
- 第 1 行（距离列表）
- A 列（时间列表）
- 数据区域形状
- 是否有预填值
- 是否含空值
结果保存到 report/template_spec.json
"""

from pathlib import Path
import json
import openpyxl

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILES = ["result1.xlsx", "result2.xlsx", "result3.xlsx", "result4.xlsx"]


def parse_sheet(ws):
    """解析一个 sheet，返回规格字典。"""
    # 第 1 行：距离
    header = []
    for cell in ws[1]:
        v = cell.value
        if v is None:
            break
        header.append(v)

    # A 列：时间
    time_col = []
    for row in ws.iter_rows(min_col=1, max_col=1, values_only=True):
        v = row[0]
        if v is None:
            continue
        time_col.append(v)

    # 数据区域：第 2 行起，第 2 列起
    data_rows = []
    n_rows = ws.max_row
    n_cols = ws.max_column
    for r in range(2, n_rows + 1):
        row_vals = []
        for c in range(2, n_cols + 1):
            row_vals.append(ws.cell(row=r, column=c).value)
        data_rows.append(row_vals)

    # 统计
    total_cells = len(data_rows) * (n_cols - 1) if n_cols > 1 else 0
    filled_cells = sum(
        1 for row in data_rows for v in row if v is not None
    )
    has_prefilled = filled_cells > 0

    return {
        "sheet_name": ws.title,
        "header_row": header,
        "header_count": len(header),
        "time_column": time_col[:10],   # 只存前 10 个，避免太长
        "time_count": len(time_col),
        "time_first": time_col[0] if time_col else None,
        "time_last": time_col[-1] if time_col else None,
        "max_row": n_rows,
        "max_col": n_cols,
        "data_shape": [len(data_rows), max(n_cols - 1, 0)],
        "total_cells": total_cells,
        "filled_cells": filled_cells,
        "has_prefilled": has_prefilled,
    }


def parse_file(path):
    """解析一个 xlsx 文件，返回所有 sheet 的规格。"""
    wb = openpyxl.load_workbook(path, data_only=True)
    result = {
        "file": path.name,
        "sheets": [],
    }
    for ws in wb.worksheets:
        result["sheets"].append(parse_sheet(ws))
    wb.close()
    return result


def main():
    all_specs = []
    for fname in RESULT_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"[缺失] {path}")
            continue
        print(f"[解析] {path}")
        spec = parse_file(path)
        all_specs.append(spec)

    out_path = REPORT_DIR / "template_spec.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_specs, f, ensure_ascii=False, indent=2)

    print(f"\n规格已保存到: {out_path}")

    # 同时在终端打印摘要
    for spec in all_specs:
        print(f"\n=== {spec['file']} ===")
        for s in spec["sheets"]:
            print(f"  sheet: {s['sheet_name']}")
            print(f"    第1行: {s['header_row']}")
            print(f"    时间列前10: {s['time_column']}")
            print(f"    时间范围: {s['time_first']} → {s['time_last']}")
            print(f"    数据区域: {s['data_shape']}")
            print(f"    预填值: {s['has_prefilled']} ({s['filled_cells']}/{s['total_cells']})")


if __name__ == "__main__":
    main()