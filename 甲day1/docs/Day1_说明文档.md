# Day 1 说明文档 · 公共数值内核与问题1求解（甲）

> **模型层次**：M0（题设基线）
> **交付人**：甲（数值计算负责人）
> **对应清单**：`CODE-01 / 06 / 07 / 08 / 09 / 10 / 11 / 16 / 17 / 18(甲自用) / 20 / 21 / 29`
> **运行入口**：`scripts/run_day1.py`

---

## 一、30 秒上手

```bash
cd "26-CUMCM/甲day1"
PY="C:/Users/33154/AppData/Local/Programs/Python/Python314/python.exe"

# 完整流水线（约 31 s）
"$PY" scripts/run_day1.py

# 常用选项
"$PY" scripts/run_day1.py --N 200 --grading 1.5   # 指定网格（默认即此值）
"$PY" scripts/run_day1.py --theta 0.5             # Crank–Nicolson（仅用于时间阶验证）
"$PY" scripts/run_day1.py --no-fig                # 跳过出图

# 单元自检
"$PY" -m src.config                    # 单位与四档 D 关键值
"$PY" -m src.numerics.properties       # 三套物性对照
"$PY" -m src.data_prep.env_interp      # 附件1 数据特征
"$PY" -m src.validation.bessel_bench   # 常系数解析基准
```

**产出**：
- `results/M0/result1.xlsx` —— 题设要求的正式结果
- `figs/*.png` / `*.pdf` —— FIG-08、FIG-41
- `docs/run_log_<时间戳>.json` —— 本次运行的完整配置与验证记录

---

## 二、目录与模块职责

```
甲day1/
├── src/
│   ├── config.py                 CODE-01 全局配置、单位约定、四档 D 自检
│   ├── numerics/
│   │   ├── properties.py         CODE-07 三套物性（附录2/3/4）+ 干基密度
│   │   ├── fvm_cyl.py            CODE-06 圆柱FVM离散内核（半控制体+调和平均）
│   │   ├── nonlinear.py          CODE-09 Picard 迭代 + 停滞检测
│   │   └── time_stepper.py       CODE-08 θ-方法自适应推进 + 步长加倍误差控制
│   ├── data_prep/
│   │   └── env_interp.py         CODE-02 附件1 插值（忠实/受控平滑双版本）
│   ├── models/
│   │   └── problem1.py           CODE-10/11 问题1：先水后温，单向耦合
│   ├── io/
│   │   ├── resample.py           CODE-16 计算网格 → 0.0~2.0 cm 输出节点
│   │   └── excel_writer.py       CODE-17 写 result1.xlsx + 回读自检
│   ├── validation/
│   │   ├── bessel_bench.py       CODE-18 常系数解析基准（Bessel 级数）
│   │   ├── conservation.py       CODE-20 通量抵消 + 积分收支
│   │   └── sanity.py             CODE-21 物理合理性检查
│   └── figures/
│       ├── plot_utils.py         中文字体注册与 "xx图，（放置位置）" 命名导出
│       ├── fig08_fvm.py          FIG-08 离散示意图
│       └── fig41_history.py      FIG-41 步长/迭代历史
├── scripts/run_day1.py           总入口
├── results/M0/result1.xlsx       正式结果
├── figs/                         图（PNG + PDF）
└── docs/                         说明文档、结论、运行日志
```

---

## 三、接口约定（**Day1 上午冻结，乙/丙可直接调用**）

### 3.1 单位（全局唯一，禁止在别处二次转换）

| 量 | 单位 | 说明 |
|---|---|---|
| `t` | **s** | |
| `r`, `R0`, `L0` | **m** | `R0=0.02`, `L0=0.25`, `L_half=0.125` |
| `T` | **°C** | 场变量、初值、边界 |
| `T_K` | **K** | **仅**用于附录3/4 的 D 公式；由 `config.to_kelvin()` 完成 |
| `C` | **kg/kg** | 干基含水率 |
| `D`, `α` | **m²/s** | |
| `h` | **W/(m²·K)** | 传热用 `h/(ρcp)` 转成 m/s 后进入算子 |
| `h_m` | **m/s** | 直接进入算子 |

### 3.2 变量基准（CODE-34 入口检查第 1 项的甲侧确认）

- `C = m_w/m_d` 药材干基含水率；`Y = m_v/m_da` 空气含湿量 —— **二者不可相减**
- 本模块中 `C_inf` 一律指 **`C_inf_eq`（等效环境平衡含水率）**，
  按题设等效约定直接读自附件1，**不假装**是吸附平衡结果

### 3.3 核心可复用 API

```python
from src.config import R0, L0, H_COEF, HM_COEF, to_kelvin, check_units
from src.numerics.fvm_cyl import (Grid, assemble, implicit_step,
                                  surface_value, surface_flux, flux_loop)
from src.numerics.properties import (D_app2, D_app3, D_app4,
                                     rho_app3, cp_app3, k_app3,
                                     rho_app4, cp_app4, k_app4, rho_dry)
from src.numerics.time_stepper import AdaptiveStepper, StepRecord
from src.numerics.nonlinear import picard_solve, NonlinearFailure
from src.data_prep.env_interp import load_env, EnvInterpolator
```

**`Grid(N, R0, grading=1.0)`**
- `grading=1.0` 均匀；`grading>1` 向表面 `r=R0` 加密：`r(ζ)=R0·[1−(1−ζ)^γ]`
- 属性：`rf`（面，长度 N+1）、`rc`（中心，N）、`h`（中心间距）、`V`（体积）、`A_face`（面积）
- 全部按**单位轴向长度**

**`assemble(Gamma_cell, h_bc, phi_inf, grid) -> Operator`**
- `Gamma_cell`：传热传 `α=k/(ρcp)`，传质传 `D`
- `h_bc`：传热传 `h/(ρcp)`，传质传 `h_m`（两者量纲均为 m/s）
- 返回 `Operator`，含三对角 `diag/lower/upper` 与右端 `b`

**`implicit_step(op, phi_n, dt, theta=1.0, op_old=None)`** —— θ-方法单步，`theta=1` 为 L-稳定的 Backward Euler

**对外结果最小字段**（乙/丙对接用）：
`times, T[n_t,N], C[n_t,N], surface_T, surface_C, surface_flux_w, history, stop_reason`

---

## 四、关键设计决策与理由

### 4.1 为什么用有限体积法 + 中心半控制体

FVM 离散的是守恒形式 `d/dt∫φdV = ∮Γ∂φ/∂n dA`，**从不写出 `(1/r)∂_r(r∂_rφ)`**，
因此中心处不会出现除以趋零半径的运算。中心控制体是半径 Δr 的**实心圆柱**，
其内侧面 `r=0` 通量恒为 0，方程退化为 `dφ₀/dt = 2Γ₁(φ₁−φ₀)/Δr²`。

### 4.2 为什么界面扩散系数用调和平均

`Γ_j = 2Γ_{j-1}Γ_j/(Γ_{j-1}+Γ_j)` 是分段常数 Γ 的**精确串联电阻法则**。
本题 D 跨约 7 个数量级，算术平均会被湿侧主导、显著高估干燥侧受扩散限制的通量。

### 4.3 为什么必须隐式

半控制体 FVM 中心处显式稳定性上限约 `Δr²/(4α)`。均匀网格 Δr=0.5 mm 时为 **0.3701 s**，
无法满足 1 s 输出间隔。生产网格最小步长 0.00707 mm 时该上限更低至 **7.4×10⁻⁵ s**。
采用 L-稳定的 Backward Euler 后无此限制。
⚠️ 验收方式是**检查稳定性条件并拒绝不满足的设置**，而不是"等待观察到发散"。

### 4.4 为什么用 Picard 而不是 Newton

Picard 的迭代矩阵保持 **M-矩阵**性质 → 每个迭代满足极值原理 →
**C 的正性由构造保证**，无需裁剪，也不会像 Newton 那样过冲进入 `exp(−0.89/C)` 的悬崖区。
Newton（含解析 Jacobian）已实现为后备，供 Day2 耦合问题使用。

收敛采用**尺度归一化的双重判据**（更新量与残差**都要满足**），
并用"更新量不再显著下降"作**停滞检测**，失败即由外层步长折半重试。

### 4.5 ★ 为什么用渐变网格（本日最重要的数值决策）

**问题**：水分在表面形成极薄边界层。`t=1 s` 时厚度仅 `√(Dt) ≈ 0.070 mm`，
而均匀网格即便 `N=640`（Δr=0.031 mm）仍欠分辨，表面含水率误差达 4×10⁻⁴。

**实测收敛数据**（表面含水率）：

| N（均匀） | Δr / mm | C_surf(1 s) | C_surf(10 s) | C_surf(100 s) |
|---|---|---|---|---|
| 40 | 0.5000 | 2.4477013 | 2.4166996 | 2.2401394 |
| 80 | 0.2500 | 2.4922001 | 2.4419011 | 2.2453846 |
| 160 | 0.1250 | 2.5107838 | 2.4477087 | 2.2466572 |
| 320 | 0.0625 | 2.5161445 | 2.4490715 | 2.2469729 |
| 640 | 0.0312 | 2.5173491 | 2.4494081 | 2.2470520 |
| Richardson 外推 | — | **≈ 2.5178** | ≈ 2.4495 | ≈ 2.2471 |

收敛比 2.40 → 3.47 → 4.45，趋于 4，**确认为二阶**，
但需要极细网格才能压住早期边界层。

**解决方案**：向表面加密的渐变网格 `r(ζ)=R0·[1−(1−ζ)^γ]`。
中心区剖面平坦无需加密，表面区需要极细网格 —— 渐变网格正好匹配这一分布。

**生产网格 `N=200, γ=1.5`**（Δr: 0.00707 ~ 0.14981 mm，比 21:1）与参考解对比：

| 对比项 | 结果 |
|---|---|
| 相对均匀 N=640（3.2 倍单元数） | 内部各列 ΔT ≤ **2.5×10⁻⁵ °C**，ΔC ≤ **8.9×10⁻⁶ kg/kg** |
| 唯一超 4 位小数舍入量子(5×10⁻⁵)的列 | 含水率 r=2.0 cm，`t=1 s` 处 2.0×10⁻⁴（随后迅速衰减） |
| 计算耗时 | **23 s vs 51 s**（约一半），且精度更高 |

> **结论**：渐变网格以约一半的计算量取得了优于均匀 N=640 的精度。
> 表面列在最早期（`t ≲ 30 s`）的边界层欠分辨是**已知且已量化**的局限，见结论文档 §4。

### 4.6 输出采样：三段不同处理

| 输出位置 | 处理方式 | 理由 |
|---|---|---|
| `r = 0` | **偶函数二次外推** `φ(0)=(φ₀r₁²−φ₁r₀²)/(r₁²−r₀²)` | 单元平均值是体积平均，非轴心点值（偏差 O(Δr²)） |
| `0 < r < R₀` | **PCHIP 保形插值** | 保形 → **结构上不可能过冲**；三次样条会振铃到负值 |
| `r = R₀` | **Robin 边界重构值** `φ_s=(φ_N+Bi_dφ_∞)/(1+Bi_d)` | 与边界条件所依据的量完全一致，**不外插** |

⚠️ 过冲检验**只对内部插值列**进行：`r=0` 列是外推、`r=R₀` 列是边界重构，
二者合理地落在单元值域之外（当 `φ_∞ > φ_N` 时 `φ_s > φ_N`），这不是过冲。

---

## 五、输出规格（严格按附件3 模板实测）

`results/M0/result1.xlsx`：

| 项 | 规格 |
|---|---|
| sheet | 仅 `温度`、`水分浓度` 两个（运行信息单独存 `result1_运行信息.xlsx`） |
| `A1` | 字符串 `时间\到药材中心的距离`（含反斜杠，与模板逐字符一致） |
| `B1:V1` | `0.0, 0.1, …, 2.0`（21 个数值，单位 **cm**） |
| `A2:A1801` | 时间 **1, 2, …, 1800**（整数，**从 1 起，不含 0**） |
| 正文 | 4 位小数（存储舍入 + 单元格数字格式 `0.0000`） |

**回读自检**（`validate_output`）逐项核对 sheet 名（必须恰好两个）、表头、距离轴、
时间轴起止与行数、列数、无 NaN、数值范围、小数位数与数字格式 —— **本次全部通过**。

> 舍入**只在写盘时发生**；主数组保持未舍入，供后续阈值判定（CODE-15）使用。

---

## 六、图件

| 编号 | 文件 | 放置位置 |
|---|---|---|
| FIG-08 | `figs/有限体积离散示意图，（论文第4章 问题1·数值方法）.png/.pdf` | 论文第4章 问题1·数值方法小节 |
| FIG-41 | `figs/自适应时间步长与非线性迭代历史图，（论文第8章 模型检验·数值验证）.png/.pdf` | 论文第8章 模型检验·数值验证小节 |

命名遵循分工要求 `"xx图，（放置位置）"`，同时导出 PNG(300 dpi) 与 PDF 矢量版。
字体：SimHei / Microsoft YaHei，已设 `axes.unicode_minus=False` 与
`mathtext.fontset='dejavusans'`。

---

## 七、给乙 / 丙的交接说明

**已冻结、可直接复用**：
- `Grid` / `assemble` / `implicit_step` / `flux_loop` / `surface_value` / `surface_flux`
- `D_app3` / `rho_app3` / `cp_app3` / `k_app3`（问题2/3 直接调用）
- `AdaptiveStepper`（问题2/3/4 复用）
- `EnvInterpolator`（含 `plateau_stats()`，供恒温段边界设定）
- `resample` / `excel_writer`（问题2 结构相同）

**乙（D1 下午起）需要动的地方**：
1. `CODE-14` 移动边界内核 —— 建议新建 `src/models/p4_moving_boundary.py`，
   通过 `assemble` 传入 `grid` 与按 ξ 变化的 `Gamma_cell` 即可复用现有离散
2. `CODE-04` 半径拟合 —— `src/data_prep/fit_radius.py`（甲未创建，归乙）
3. 若需扩展 `Operator` 接口，**先与甲确认**

**丙（D1 下午起）需要动的地方**：
1. `CODE-31` 模板解析 —— 已实测确认规格见 §5，可直接落 `output_spec.json`
2. `CODE-32` 论文表格 —— 从 `results/M0/result1.xlsx` 抽取表1/表2
3. `CODE-29` 绘图 —— 复用 `plot_utils.setup_style()` 与 `save_fig()`

**已冻结的接口不变量**：
- `times` 从 0 起（含 0），写 result 时用 `res.times >= 1` 选择
- `T` / `C` 形状 `(n_t, N)`，按 `grid.rc` 排列
- 所有物理量单位见 §3.1

---

## 八、本次运行记录

见 `docs/run_log_<时间戳>.json`，含：四档 D 关键值、派生无量纲数、
环境数据统计、网格信息、接受/拒绝步数、守恒检验值、合理性检查结论、
输出自检结果、图件路径、总耗时。

**最近一次运行**：`20260911-155033`，状态 `OK`，总耗时 31.0 s。
