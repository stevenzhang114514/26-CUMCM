"""
把四个 result 文件对齐到题目附件3 的模板格式        [交付前格式对齐]

模板事实（逐格实测 CUMCM2026Problems/A题/附件/附件3/）
------------------------------------------------------
    result1.xlsx   sheets = [温度, 水分浓度]
                   A1 = 时间\\到药材中心的距离
                   行1 = 0, 0.1, 0.2, …, 2   （21 个距离列，B1=0 是 int，其余 float）
                   A 列 = 1, 2, 3, …（整数秒，步长 1，从 1 起）
    result2.xlsx   同 result1
    result3.xlsx   sheets = [Sheet1]
                   行1 同 result1；A 列 = 60, 120, 180, …（步长 60）
    result4.xlsx   sheets = [Sheet1]
                   行1 = 0, 0.1, …, 1.9, 药材表面   （20 个距离列 + 末列字符串）

本次发现的偏差（对齐前）
------------------------
1. 🔴 **result2/3/4 的 A1 写成「时间\\到药材的**中心**距离」，模板是「时间\\到药材中心的距离」**
   —— 多了一个「的」字。result1 是对的。
2. ⚠️ result2/3/4 多出模板没有的「运行信息」sheet。
   处理：**从正式交付文件中移除**，另存为同目录的 `resultN_运行信息.xlsx`，
   既严格对齐模板、又不丢失配置溯源。

原则
----
**只搬移与改格式，不重算任何数值** —— 逐格复制后回读逐格比对，保证数值不变。
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

ROOT = Path("c:/Users/33154/Desktop/国赛/26-CUMCM")
TPL = Path("C:/Users/33154/Desktop/CUMCM2026Problems/A题/附件/附件3")

HEADER_FIX = "时间\\到药材中心的距离"    # 模板逐字符实测值

TARGETS = [
    # (标签, 源文件, 正式交付保留的 sheet)
    ("result1", ROOT / "甲day1/results/M0/result1.xlsx", ["温度", "水分浓度"]),
    ("result2", ROOT / "甲day2/results/M0/result2.xlsx", ["温度", "水分浓度"]),
    ("result3", ROOT / "甲day2/results/M0/result3.xlsx", ["Sheet1"]),
    ("result4", ROOT / "甲做第四问/results/result4.xlsx", ["Sheet1"]),
]
# 追加对齐（不拆 sheet，只修表头）
EXTRA = [
    ("result2_全程版", ROOT / "甲day2/results/M0/result2_全程版.xlsx"),
    ("result1_运行信息", ROOT / "甲day1/results/M0/result1_运行信息.xlsx"),
]


def P(*a):
    print(*a, flush=True)


def read_all(path):
    """读全部 sheet → {sheet: [[cell,...],...]}（保持 None，域外留空语义）。"""
    wb = openpyxl.load_workbook(path, data_only=True)
    out = {}
    for sn in wb.sheetnames:
        ws = wb[sn]
        out[sn] = [[ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
                   for r in range(1, ws.max_row + 1)]
    wb.close()
    return out


def write_sheets(path, sheets: dict):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for sn, rows in sheets.items():
        ws = wb.create_sheet(sn)
        for r, row in enumerate(rows, start=1):
            for c, v in enumerate(row, start=1):
                if v is not None:               # 域外留空：不写任何值
                    ws.cell(r, c, v)
    wb.save(path)
    wb.close()


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def norm_axis(vals):
    return [int(v) if is_num(v) and abs(v - round(v)) < 1e-12
            else (round(float(v), 10) if is_num(v) else v) for v in vals]


def main():
    P("=" * 80)
    P("  把四个 result 对齐到附件3 模板")
    P("=" * 80)
    changes = []

    for tag, path, keep in TARGETS:
        P(f"\n[{tag}]  {path.relative_to(ROOT)}")
        src = read_all(path)
        have = list(src)
        extra = [s for s in have if s not in keep]

        # ---------- 0. 先把额外 sheet 另存（在覆盖之前）----------
        if extra:
            meta_path = path.with_name(f"{path.stem}_运行信息.xlsx")
            write_sheets(meta_path, {s: src[s] for s in extra})
            P(f"   额外 sheet {extra} 已另存 → {meta_path.name}")

        # ---------- 1. 表头修正 ----------
        for sn in keep:
            old = src[sn][0][0]
            if old != HEADER_FIX:
                src[sn][0][0] = HEADER_FIX
                P(f"   表头修正 [{sn}]：{old!r}")
                P(f"                → {HEADER_FIX!r}")
                changes.append(f"{tag}/{sn} 表头 {old!r} → {HEADER_FIX!r}")
            else:
                P(f"   表头 [{sn}] 已正确 ✓")

        # ---------- 2. 与模板对拍 ----------
        tpl = read_all(TPL / f"{tag}.xlsx")
        tsn = list(tpl)[0]
        t_hdr, t_t0 = tpl[tsn][0], tpl[tsn][1][0]
        for sn in keep:
            hdr, t0 = src[sn][0], src[sn][1][0]
            ok_h = norm_axis(hdr[1:]) == norm_axis(t_hdr[1:])
            ok_c = len(hdr) == len(t_hdr)
            ok_t = t0 == t_t0
            P(f"   核对 [{sn}]  列数 {len(hdr)}/{len(t_hdr)} {'✓' if ok_c else '✗'}"
              f"   距离轴 {'✓' if ok_h else '✗'}"
              f"   时间起点 {t0}(模板 {t_t0}) {'✓' if ok_t else '✗'}")
            if not ok_h:
                P(f"      我方 {norm_axis(hdr[1:])[:5]} … {norm_axis(hdr[1:])[-2:]}")
                P(f"      模板 {norm_axis(t_hdr[1:])[:5]} … {norm_axis(t_hdr[1:])[-2:]}")

        # ---------- 3. 写出并回读逐格比对 ----------
        write_sheets(path, {s: src[s] for s in keep})
        back = read_all(path)
        bad = 0
        for sn in keep:
            a, b = src[sn], back[sn]
            if len(a) != len(b):
                P(f"   🔴 [{sn}] 行数变了 {len(a)}→{len(b)}"); bad += 1; continue
            for r in range(len(a)):
                if len(a[r]) != len(b[r]):
                    P(f"   🔴 [{sn}] 第{r+1}行列数变了"); bad += 1; break
                for c in range(len(a[r])):
                    va, vb = a[r][c], b[r][c]
                    if va is None and vb is None:
                        continue
                    if isinstance(va, str) or isinstance(vb, str):
                        if va != vb: bad += 1
                    elif va is None or vb is None or abs(float(va) - float(vb)) > 0:
                        bad += 1
        P(f"   已写出  sheets={keep}   回读逐格比对："
          f"{'全部一致 ✓' if bad == 0 else f'🔴 {bad} 处不一致'}")
        P(f"   规格  {len(back[keep[0]])} 行 × {len(back[keep[0]][0])} 列")

    # ---------- 4. 附带文件：只修表头，不拆 sheet ----------
    P("\n[附带文件] 只修表头")
    for tag, path in EXTRA:
        src = read_all(path)
        hit = [sn for sn in src if src[sn] and src[sn][0] and
               isinstance(src[sn][0][0], str) and "到药材" in str(src[sn][0][0])]
        if not hit:
            P(f"   [{tag}] 非数据表（无距离表头），跳过")
            continue
        for sn in hit:
            old = src[sn][0][0]
            if old != HEADER_FIX:
                src[sn][0][0] = HEADER_FIX
                changes.append(f"{tag}/{sn} 表头 {old!r} → {HEADER_FIX!r}")
                P(f"   [{tag}/{sn}] {old!r} → 已修正")
            else:
                P(f"   [{tag}/{sn}] 已正确 ✓")
        write_sheets(path, src)

    P("\n" + "=" * 80)
    P(f"  共修改 {len(changes)} 处")
    for c in changes:
        P(f"    · {c}")
    P("=" * 80)


if __name__ == "__main__":
    main()
