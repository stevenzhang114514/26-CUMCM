# 乙 - 问题4 代码与结果

## 项目说明

本项目是 2026 国赛 A 题「药材的烘干问题」中乙负责的部分，
包括入口检查、半径拟合、问题4 求解、以及多项验证。

## 文件夹结构

- `code/`：所有 MATLAB 代码
- `data/`：结果数据和拟合函数
- `figs/`：所有图（可直接用于论文）
- `乙_任务汇总.pdf`：完整任务文档

## 核心结果

- 问题4 的 t_dry = 73.03 h
- 终态半径 = 1.198 cm
- 终态最大含水率 = 0.1500 kg/kg

## 代码清单

| 文件 | 作用 |
|---|---|
| `problem4_drying.m` | 问题4 主程序 |
| `problem4_drying_const.m` | 固定半径对比版 |
| `problem4_drying_linear.m` | 线性外推对比版 |
| `thomas.m` | 三对角求解器 |
| `geo_density_diag.m` | 几何—密度诊断 |
| `verify_bessel.m` | 解析基准验证 |

## 运行说明

1. 确保 `附件1.xlsx`、`附件2.xlsx`、`result4.xlsx`、`R_fit.mat` 在正确路径下
2. 在 MATLAB 中运行 `problem4_drying.m`
3. 结果输出到 `data/result4.xlsx`，图输出到 `figs/`

## 数据文件

- `data/result4.xlsx`：问题4 最终结果
- `data/R_fit.mat`：半径拟合函数

## 图片文件

- `figs/问题4_含水率曲线.png`：问题4 核心图
- `figs/半径拟合图.png`：半径拟合图
- `figs/几何密度诊断图.png`：几何—密度诊断图
- `figs/解析基准对比图.png`：解析基准验证图

## 联系

如有问题，请联系乙。