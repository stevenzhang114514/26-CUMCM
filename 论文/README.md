# 论文 · 药材的烘干问题

> **成品**：`main.pdf`（50 页）→ 已复制到 `C:\Users\33154\Desktop\药材的烘干问题.pdf`
> **编译**：`xelatex` 两遍（TeX Live 2026，`ctexart` + `fontset=windows`）

## 目录

```
论文/
├── main.tex            主文件（preamble + \input 各章）
├── sections/           15 个分章 .tex
├── tables/             tab1–tab6.tex（**由脚本生成，勿手改**）
├── figs/               23 张图（PNG 300 dpi + PDF 矢量）
├── scripts/            出图与出表代码
└── 修订记录.md          **先看这份**：相对 论文骨架1.md 改了哪些数
```

## 复现

```bash
cd 26-CUMCM/论文/scripts
export PYTHONIOENCODING=utf-8

py fig_schema.py     # G1  G1b  FIG-08
py fig_env.py        # G2  G2b  FIG-13  G3
py fig_p1p2.py       # G4  G5  FIG-16  FIG-18
py fig_p3p4.py       # G6  G12  G7  G11
py fig_checks.py     # G8  FIG-14  FIG-41  FIG-43
py fig_sens.py       # G9  G10  G13  G14
py make_tables.py    # 表 1–6

cd .. && xelatex main.tex && xelatex main.tex
```

## 两件必须先知道的事

1. **配色**：全图统一使用 6 色板 `#BD4DA3 / #8182BA / #B58581 / #88C6E2 / #FBF065 / #C5E4E7`，
   定义在 `scripts/palette.py`。**网格/坐标轴/参考线一律中性灰，不占色板**。
   该模块还带两个补丁：SimHei 缺 U+2212 字形（对数刻度要显式给 ASCII 标签）、
   PNG+PDF 双出。
2. **数据源**：`scripts/paperdata.py` 是唯一的数据入口，
   **只读甲的权威结果**（`甲day1`、`甲day2`、`甲做第四问`），
   根本不提供指向副本的路径 —— `丙day1/_superseded/` 下的 `result4.xlsx`
   是乙的旧版（73.03 h，网格欠分辨偏大 43%），**绝不可用**。

## 论文的核心结论（四个答案）

| | 答案 |
|---|---|
| 问题1 | 1800 s：中心 33.5763 ℃ / 表面 36.7861 ℃ |
| 问题2 | 3 h：中心 49.8494 ℃ / 表面 49.9666 ℃ |
| 问题3 | $t_*$ = **57.49 h**（2 位小数；未舍入 206959.9521 s） |
| 问题4 | $t_{dry}$ = **51.0906 h**（Δt=1 s 收敛口径），终态半径 1.2000 cm |
