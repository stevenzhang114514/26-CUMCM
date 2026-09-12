"""
构建「支撑材料」文件夹                    [交付打包脚本]

产出：C:/Users/33154/Desktop/支撑材料/
    01_result/              四问结果文件 + 题设表 1–6
    02_代码/                按 问题1→问题4 顺序 + 00_公共内核
    03_参考文献/
    04_提示词与分工清单/
    05_图片/                23 张论文插图 + 出图脚本

为什么这样分
------------
《全国大学生数学建模竞赛论文格式规范》第十一条：
支撑材料「至少应包含建模所用到的所有可运行源程序、自主查阅使用的数据资料、
较大篇幅中间结果的图表等」，其文件列表放入论文附录。
《人工智能工具使用规定》第 4 条：支撑材料需含「AI 工具使用详情.pdf」。

🔴 代码必须**能运行**（规范第五条：程序不能运行可能被取消评奖资格），
   故 00_公共内核 里放的是**完整的 src/ 包**（保持 import 结构）；
   01–04 是按题归档的**该题专属模块与脚本**，便于按题查阅。

⚠️ 本脚本**只用 Write/复制**，不用 Python 字符串拼 LaTeX ——
   先前用脚本生成 .tex 时 \n \t \r 被当转义吃掉，损坏过文件。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Windows GBK 控制台会把中文输出打成乱码（实测）—— 显式改 UTF-8。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent          # .../26-CUMCM
DESK = Path.home() / "Desktop"
OUT = DESK / "支撑材料"

# 统一内核：甲做第四问/src 是 甲day3/src 的**严格超集**（仅多 T_TARGET 与 write_result4）
KERNEL = ROOT / "甲做第四问" / "src"


def P(*a):
    print(*a, flush=True)


def mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def cp(src: Path, dst: Path):
    if not src.exists():
        P(f"    ⚠ 缺失，跳过：{src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# ==========================================================================
def stage_result():
    P("\n[01] result")
    d = mk(OUT / "01_result")
    # 四个正式结果 + 全程版；运行信息为**伴随文件**（模板里没有这个 sheet，
    # 依《格式规范》与附件3 对齐后另存，见 build_support 上游的 align_results.py）
    SRC = {"result1.xlsx": ROOT / "甲day1/results/M0/result1.xlsx",
           "result2.xlsx": ROOT / "甲day2/results/M0/result2.xlsx",
           "result2_全程版.xlsx": ROOT / "甲day2/results/M0/result2_全程版.xlsx",
           "result3.xlsx": ROOT / "甲day2/results/M0/result3.xlsx",
           "result4.xlsx": ROOT / "甲做第四问/results/result4.xlsx",
           "result1_运行信息.xlsx": ROOT / "甲day1/results/M0/result1_运行信息.xlsx",
           "result2_运行信息.xlsx": ROOT / "甲day2/results/M0/result2_运行信息.xlsx",
           "result3_运行信息.xlsx": ROOT / "甲day2/results/M0/result3_运行信息.xlsx",
           "result4_运行信息.xlsx": ROOT / "甲做第四问/results/result4_运行信息.xlsx"}
    n = 0
    for name, p in SRC.items():
        if cp(p, d / name):
            n += 1
    P(f"    结果文件 {n} 个")

    # ---- 题设表 1–6：**从权威结果重新生成**，不用丙的旧 CSV ----
    # 丙的 表6_水分浓度.csv 基于旧的 73.03 h 解（末行 54 h），已过时。
    import sys
    sys.path.insert(0, str(ROOT / "论文" / "scripts"))
    import numpy as np
    import paperdata as PD

    tdir = mk(d / "题设表")
    COL = [0.0, 0.5, 1.0, 1.5, 2.0]

    def dump_csv(fn, times, rows, colhead, rowlabel):
        lines = ["," + ",".join(colhead)]
        for lab, vals in zip(times, rows):
            cells = ["" if (v is None or not np.isfinite(v)) else f"{v:.4f}"
                     for v in vals]
            lines.append(f"{lab}," + ",".join(cells))
        (tdir / fn).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def sample(dd, ts, cols):
        out = []
        for t_ in ts:
            k = int(np.argmin(np.abs(dd["t"] - t_)))
            out.append([float(np.interp(c, dd["r_cm"], dd["F"][k])) for c in cols])
        return out

    r1, r2, r3 = PD.result1(), PD.result2(), PD.result3()
    r1T = {"t": r1["t"], "r_cm": r1["r_cm"], "F": r1["T"]}
    r1C = {"t": r1["t"], "r_cm": r1["r_cm"], "F": r1["C"]}
    r2T = {"t": r2["t"], "r_cm": r2["r_cm"], "F": r2["T"]}
    r2C = {"t": r2["t"], "r_cm": r2["r_cm"], "F": r2["C"]}

    lab1 = [100, 300, 600, 900, 1200, 1500, 1800]
    head5 = [f"{c}cm" for c in COL]
    dump_csv("表1_温度.csv", lab1, sample(r1T, lab1, COL), head5, "时间/s")
    dump_csv("表2_水分浓度.csv", lab1, sample(r1C, lab1, COL), head5, "时间/s")

    lab2h = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    lab2 = [int(h * 3600) for h in lab2h]
    dump_csv("表3_温度.csv", lab2, sample(r2T, lab2, COL), head5, "时间/s")
    dump_csv("表4_水分浓度.csv", lab2, sample(r2C, lab2, COL), head5, "时间/s")
    # 把时间列改成小时标注
    for fn, lab in (("表3_温度.csv", lab2h), ("表4_水分浓度.csv", lab2h)):
        p = tdir / fn
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, h in enumerate(lab, 1):
            lines[i] = f"{h}," + lines[i].split(",", 1)[1]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 表5：问题3，末行为达标时刻
    h5 = [h for h in range(6, 55, 6) if h <= PD.TSTAR_H]
    rows5, labs5 = [], []
    for h in h5:
        k = int(np.argmin(np.abs(r3["t"] - h * 3600)))
        rows5.append([float(np.interp(c, r3["r_cm"], r3["C"][k])) for c in COL])
        labs5.append(h)
    rows5.append([float(np.interp(c, r3["r_cm"], r3["C"][-1])) for c in COL])
    labs5.append(f"{PD.TSTAR_H:.4f}(达标)")
    dump_csv("表5_水分浓度.csv", labs5, rows5, head5, "时间/h")

    # 表6：问题4，Δt=1 s 口径，列到 1.5 cm + 药材表面
    d6 = PD.p4_paper_table6()
    c6 = [0.0, 0.5, 1.0, 1.5]
    lines = [",0.0cm,0.5cm,1.0cm,1.5cm,药材表面,半径/cm"]
    for rec in d6["表6行"]:
        vals = rec[1:1 + 4] + [rec[1 + 4]]
        cells = ["" if v is None else f"{v:.4f}" for v in vals]
        lines.append(f"{rec[0]:g}," + ",".join(cells) + f",{rec[6]:.4f}")
    (tdir / "表6_水分浓度.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    P(f"    题设表 1–6 已按权威结果重新生成 → 题设表/")

    (d / "说明.md").write_text(
        "# 01_result　结果文件\n\n"
        "## 格式对齐声明\n\n"
        "四个 `result*.xlsx` 的格式**已逐格对齐题目附件3 的模板**，实测确认：\n\n"
        "| 检查项 | result1 | result2 | result3 | result4 |\n|---|---|---|---|---|\n"
        "| sheet 名与顺序 | 温度, 水分浓度 | 温度, 水分浓度 | Sheet1 | Sheet1 |\n"
        "| A1（逐字符） | `时间\\到药材中心的距离` | 同左 | 同左 | 同左 |\n"
        "| 列数 | 22 | 22 | 22 | 22 |\n"
        "| 距离轴 | 0, 0.1, …, 2.0 | 同左 | 同左 | 0, 0.1, …, **1.9**, 药材表面 |\n"
        "| A 列起点 / 步长 | 1 / 1 s | 1 / 1 s | 60 / 60 s | 60 / 60 s |\n\n"
        "> ⚠️ 对齐前的偏差（已修正）：`result2/3/4` 的表头曾写成\n"
        "> `时间\\到药材的**中心**距离`（多一个「的」字），与模板不符；\n"
        "> 且这三份曾多带一个模板没有的「运行信息」sheet。\n"
        "> 现表头已逐字符改正，多余 sheet 已移出为下面的伴随文件。\n\n"
        "## 正式交付（M0 题设基线）\n\n"
        "| 文件 | 内容 | 规格 |\n|---|---|---|\n"
        "| `result1.xlsx` | 问题1 温度、水分浓度 | 双 sheet，1 s × 0.1 cm，1–1800 s |\n"
        "| `result2.xlsx` | 问题2 温度、水分浓度 | 双 sheet，1 s × 0.1 cm，1–10800 s |\n"
        "| `result2_全程版.xlsx` | 问题2 全程 | 双 sheet，60 s × 0.1 cm，60–206960 s |\n"
        "| `result3.xlsx` | 问题3 水分浓度 | 单 sheet，60 s × 0.1 cm，60–206960 s，**末行为达标时刻** |\n"
        "| `result4.xlsx` | 问题4 水分浓度（收缩） | 单 sheet，60 s，距离轴只到 1.9 cm，"
        "末列为字符串「药材表面」，**域外留空** |\n\n"
        "## 伴随文件（运行信息）\n\n"
        "`result1_运行信息.xlsx` … `result4_运行信息.xlsx`：各题的运行配置、"
        "网格与容差、版本标识、判定值等溯源信息。\n\n"
        "> 这些内容**原先放在结果文件的第二个 sheet 里**，与模板结构不符，\n"
        "> 故在对齐时移出为独立文件——既保证交付文件与模板一致，又不丢失配置溯源。\n\n"
        "## 题设表 1–6（`题设表/`）\n\n"
        "由脚本从上述结果文件**直接取样生成**（不手工抄录），保留四位小数，域外留空。\n\n"
        "> ⚠️ 表 6 采用 **Δt = 1 s 收敛口径**（t_dry = 51.0906 h，终态半径 1.2000 cm）。\n"
        "> 早期版本曾用 60 s 输出网格上的 t_dry = 51.1667 h，两者不一致，以本表为准。\n\n"
        "## 未列入的中间结果\n\n"
        "M1（含蒸发潜热的条件性扩展）的结果**不列入正式交付**——\n"
        "论文明确声明 M1 只作对照、不修订四问答案。其原始产物保留在开发工作区。\n",
        encoding="utf-8")


# ==========================================================================
def stage_code():
    P("\n[02] 代码")
    d = mk(OUT / "02_代码")

    # ---- 00_公共内核：完整可运行的 src/ 包 ----
    kern = mk(d / "00_公共内核" / "src")
    n = 0
    for f in sorted(KERNEL.rglob("*.py")):
        if "__pycache__" in str(f):
            continue
        rel = f.relative_to(KERNEL)
        cp(f, kern / rel)
        n += 1
    P(f"    00_公共内核/src：{n} 个模块（四问共用，保持 import 结构，可运行）")

    # ---- 按题归档 ----
    J = {
        "01_问题1": {
            "模块": ["src/models/problem1.py"],
            "脚本": {"scripts/run_day1.py": ROOT / "甲day1/scripts/run_day1.py",
                     "scripts/check_q1_vs_public.py": ROOT / "甲day1/scripts/check_q1_vs_public.py",
                     "scripts/check_q1_bc_level.py": ROOT / "甲day1/scripts/check_q1_bc_level.py",
                     "scripts/check_q1_dt_refine.py": ROOT / "甲day1/scripts/check_q1_dt_refine.py",
                     "scripts/check_q1_tolerance.py": ROOT / "甲day1/scripts/check_q1_tolerance.py"},
        },
        "02_问题2": {
            "模块": ["src/models/problem2.py"],
            "脚本": {"scripts/run_day2.py": ROOT / "甲day2/scripts/run_day2.py",
                     "scripts/make_figures_day2.py": ROOT / "甲day2/scripts/make_figures_day2.py",
                     "scripts/make_tables_day2.py": ROOT / "甲day2/scripts/make_tables_day2.py",
                     "scripts/check_regression_day1.py": ROOT / "甲day2/scripts/check_regression_day1.py"},
        },
        "03_问题3": {
            "模块": ["src/models/problem3.py", "src/models/problem3_2d.py"],
            "脚本": {"scripts/run_day3.py": ROOT / "甲day3/scripts/run_day3.py",
                     "scripts/run_day3_endo.py": ROOT / "甲day3/scripts/run_day3_endo.py",
                     "scripts/run_day3_crust.py": ROOT / "甲day3/scripts/run_day3_crust.py",
                     "scripts/run_sens_param.py": ROOT / "甲day3/scripts/run_sens_param.py",
                     "scripts/run_robust_data.py": ROOT / "甲day3/scripts/run_robust_data.py",
                     "scripts/fig_robust.py": ROOT / "甲day3/scripts/fig_robust.py",
                     "scripts/audit_data_provenance.py": ROOT / "甲day3/scripts/audit_data_provenance.py",
                     "scripts/explain_reference_gap.py": ROOT / "甲做第四问/scripts/explain_reference_gap.py"},
        },
        "04_问题4": {
            "模块": ["src/models/problem4.py"],
            "脚本": {"scripts/run_p4.py": ROOT / "甲做第四问/scripts/run_p4.py",
                     "scripts/p4_paper.py": ROOT / "甲做第四问/scripts/p4_paper.py",
                     "scripts/p4_time_conv.py": ROOT / "甲做第四问/scripts/p4_time_conv.py",
                     "scripts/check_p4_vs_yi.py": ROOT / "甲做第四问/scripts/check_p4_vs_yi.py",
                     "scripts/review_yi_p4.py": ROOT / "甲做第四问/scripts/review_yi_p4.py",
                     "scripts/review_yi_p4_profiles.py": ROOT / "甲做第四问/scripts/review_yi_p4_profiles.py",
                     "scripts/compare_reference.py": ROOT / "甲做第四问/scripts/compare_reference.py"},
        },
    }
    tot = 0
    for k, spec in J.items():
        dd = mk(d / k)
        for rel in spec["模块"]:
            cp(KERNEL / rel.replace("src/", ""), dd / rel)
            tot += 1
        for rel, src in spec["脚本"].items():
            if cp(src, dd / rel):
                tot += 1
    P(f"    01–04 按题归档：{tot} 个文件")

    (d / "README_代码说明.md").write_text(
        "# 02_代码　代码说明\n\n"
        "## 目录\n\n"
        "```\n"
        "02_代码/\n"
        "├── 00_公共内核/        完整的 src/ 包（四问共用，**可运行**）\n"
        "├── 01_问题1/           该题专属模块 + 运行/校验脚本\n"
        "├── 02_问题2/           （同上）\n"
        "├── 03_问题3/           （同上，含灵敏度与稳健性实验）\n"
        "└── 04_问题4/           （同上，含与另一实现的对照）\n"
        "```\n\n"
        "## 运行方法\n\n"
        "**代码以 `00_公共内核/` 为可运行根**（`src/` 包保持完整 import 结构）。\n"
        "把 `00_公共内核/` 作为工作目录，并把同题的脚本放到其 `scripts/` 下即可运行：\n\n"
        "```bash\n"
        "cd 00_公共内核\n"
        "pip install -r requirements.txt\n"
        "python scripts/run_day3.py --stage all    # 问题1-3\n"
        "python scripts/run_p4.py   --stage all    # 问题4\n"
        "```\n\n"
        "> 01–04 各题文件夹里的 `src/` 模块是 `00_公共内核/` 中同名文件的副本，\n"
        "> **仅为按题查阅方便**；运行请以 `00_公共内核/` 为准。\n\n"
        "## 各题脚本清单\n\n"
        "| 题 | 脚本 | 作用 |\n|---|---|---|\n"
        "| 1 | `run_day1.py` | 问题1 主运行，写 result1.xlsx |\n"
        "| 1 | `check_q1_vs_public.py` | 与公开解答的差异对照 |\n"
        "| 1 | `check_q1_bc_level.py` | 边界取值时刻（t^n vs t^{n+1}）检验 |\n"
        "| 1 | `check_q1_dt_refine.py` / `check_q1_tolerance.py` | 时间步长与容差细化 |\n"
        "| 2 | `run_day2.py` | 问题2 主运行，写 result2/result3.xlsx |\n"
        "| 2 | `check_regression_day1.py` | 三级回归检验 |\n"
        "| 2 | `make_tables_day2.py` / `make_figures_day2.py` | 题设表与出图 |\n"
        "| 3 | `run_day3.py` | 问题3 冻结、收敛阶、复现（`--stage sens/conv/freeze/repro`）|\n"
        "| 3 | `run_sens_param.py` | 参数灵敏度 ±5/10/20%（28 次运行）|\n"
        "| 3 | `run_robust_data.py` | 数据扰动稳健性（63 次运行）|\n"
        "| 3 | `run_day3_endo.py` / `run_day3_crust.py` | 端面效应 / 结壳对照 |\n"
        "| 3 | `explain_reference_gap.py` | 与参考解的差异归因 |\n"
        "| 3 | `audit_data_provenance.py` | 数据溯源审计 |\n"
        "| 4 | `run_p4.py` | 问题4 主运行，写 result4.xlsx |\n"
        "| 4 | `p4_paper.py` | 论文口径的表6（Δt=1 s）与 B 方案 |\n"
        "| 4 | `p4_time_conv.py` | 时间收敛 Δt=30/10/3/1 s |\n"
        "| 4 | `check_p4_vs_yi.py` | 准入闸门：均匀 N=20 复现另一实现的 73.033 h |\n"
        "| 4 | `review_yi_p4*.py` | 另一实现的逐行移植与单因素扰动归因 |\n"
        "| 4 | `compare_reference.py` | 与参考解的精确求根对照 |\n\n"
        "## 依赖\n\n"
        "Python 3.14.2、NumPy 2.4.4、SciPy 1.18.0、Matplotlib 3.11.0、"
        "pandas、openpyxl。纯 CPU、单线程即可复现；\n"
        "数据扰动 Monte Carlo 部分固定随机种子。\n",
        encoding="utf-8")

    (d / "00_公共内核" / "requirements.txt").write_text(
        "numpy>=2.0\nscipy>=1.11\npandas>=2.0\nopenpyxl>=3.1\nmatplotlib>=3.8\n",
        encoding="utf-8")


# ==========================================================================
def stage_refs():
    P("\n[03] 参考文献")
    d = mk(OUT / "03_参考文献")
    R = ROOT / "参考文献原文"
    n = 0
    for f in sorted(R.glob("*.md")) + sorted(R.glob("*.json")):
        n += cp(f, d / f.name)
    # 下载工具（可复现检索过程）
    tool = mk(d / "检索工具")
    for f in sorted(R.glob("*.py")):
        n += cp(f, tool / f.name)
    # 已下载的原文 PDF
    pdf = mk(d / "原文PDF")
    for f in sorted((R / "PDF").glob("*")):
        n += cp(f, pdf / f.name)
    # 四份文献研报
    yb = mk(d / "文献研报")
    for f in sorted((ROOT / "相关文献").glob("*")):
        n += cp(f, yb / f.name)
    P(f"    {n} 个文件（含原文 PDF 与检索工具）")


# ==========================================================================
def stage_prompt():
    P("\n[04] 提示词与分工清单")
    d = mk(OUT / "04_提示词与分工清单")
    n = 0
    for f in sorted(ROOT.glob("A题_*.md")):
        n += cp(f, d / f.name)
    n += cp(ROOT / "论文池_药材烘干21因素.md", d / "论文池_药材烘干21因素.md")
    n += cp(ROOT / "论文骨架1.md", d / "论文骨架1.md")
    n += cp(ROOT / "论文" / "修订记录.md", d / "论文修订记录.md")
    tp = mk(d / "文献检索提示词")
    for f in sorted((ROOT / "文献检索提示词").glob("*")):
        n += cp(f, tp / f.name)
    P(f"    {n} 个文件")


# ==========================================================================
def stage_figs():
    P("\n[05] 图片")
    d = mk(OUT / "05_图片")
    n = 0
    for f in sorted((ROOT / "论文" / "figs").glob("*.png")):
        n += cp(f, d / f.name)
    for f in sorted((ROOT / "论文" / "figs").glob("*.pdf")):
        n += cp(f, d / f.name)
    n += cp(ROOT / "论文图片" / "图目录.md", d / "图目录.md")
    # 出图脚本（可复现）
    sc = mk(d / "出图脚本")
    for f in sorted((ROOT / "论文" / "scripts").glob("*.py")):
        n += cp(f, sc / f.name)
    P(f"    {n} 个文件（23 张图 × PNG+PDF + 出图脚本）")


# ==========================================================================
def stage_readme():
    """
    生成根目录 README.md。

    🔴 修正：README.md 一直列在 MANAGED 里（每次重建先删），
       但**没有任何 stage 重新生成它** —— 它自某次重建后就不存在了，
       而论文附录 A 的文件列表却写着「README.md 支撑材料总说明与复现步骤」。
       实测确认缺失后补上本函数，使列表与实物一致。
    """
    P("\n[根] README")
    n_py = sum(1 for _ in (OUT / "02_代码").rglob("*.py"))
    n_fig = len(list((OUT / "05_图片").glob("*.png")))
    (OUT / "README.md").write_text(
        "# 支撑材料　2026 高教社杯全国大学生数学建模竞赛　A 题：药材的烘干问题\n\n"
        "> 本包为参赛作品的**支撑材料**。按《全国大学生数学建模竞赛论文格式规范》\n"
        "> 第十条，**论文正本（含摘要、正文、附录）作为单独文件提交，不在本包内**；\n"
        "> 本包的文件列表见论文附录 A。\n\n"
        "## 目录结构\n\n"
        "| 目录 / 文件 | 内容 |\n"
        "|---|---|\n"
        "| `01_result/` | 四个正式结果 `result1`–`result4.xlsx`；各结果对应的 "
        "`resultN_运行信息.xlsx`（网格、步长、判据口径与终止条件）；题设表 1–6；`说明.md` |\n"
        "| `02_代码/00_公共内核/` | 完整的 `src/` 包（保持 import 结构，**可直接运行**） |\n"
        "| `02_代码/01_问题1/` … `04_问题4/` | 按题归档的专属模块与运行、校验脚本 |\n"
        "| `02_代码/README_代码说明.md` | 运行方法与全部脚本清单 |\n"
        "| `03_参考文献/` | 引用核实清单（含 DOI 反查结果）、原文 PDF、下载记录、文献研报 |\n"
        "| `04_提示词与分工清单/` | 分工方案、论文骨架、求解框架、论文修订记录 |\n"
        "| `05_图片/` | 论文插图 PNG（300 dpi）与 PDF（矢量）、出图脚本、`图目录.md` |\n"
        "| `AI 工具使用详情.pdf` | 依《人工智能工具使用规定》第 4 条提供 |\n\n"
        f"本包共 {n_py} 个 Python 源文件（含运行与校验脚本）、"
        f"{n_fig} 张插图（每张 PNG + PDF）。\n\n"
        "## 复现步骤\n\n"
        "代码以 `02_代码/00_公共内核/` 为可运行根（`src/` 包保持完整 import 结构）。\n"
        "把同题的脚本放到其 `scripts/` 下即可运行：\n\n"
        "```bash\n"
        "cd 02_代码/00_公共内核\n"
        "pip install -r requirements.txt\n"
        "python scripts/run_day3.py --stage all         # 问题1-3：冻结、收敛、复现\n"
        "python scripts/run_p4.py   --stage all         # 问题4：主结果、网格、时间收敛\n"
        "python scripts/p4_paper.py --stage all         # 论文口径的表6 与 B 方案\n"
        "python scripts/run_sens_param.py --stage all   # 参数灵敏度 ±5/10/20%\n"
        "python scripts/run_robust_data.py --stage all  # 数据扰动稳健性\n"
        "```\n\n"
        "## 结果文件与口径\n\n"
        "四个 `result*.xlsx` 的 sheet 名、表头、距离轴与时间轴**均按题目附件3 的模板\n"
        "逐字符对齐**，数值未经任何手工修改。每个正式结果另附 `resultN_运行信息.xlsx`，\n"
        "记录该次运行的网格、步长、判据口径与终止条件 —— **仅作溯源，不属于题目要求交付的表**。\n\n"
        "论文正文中出现的全部四位小数均取自这些结果文件，由脚本生成、不手工抄录。\n\n"
        "## 数据来源\n\n"
        "- 环境温湿度与风速：题目附件1（插值方案与平滑对照见 `05_图片/FIG-13_smoothing`）；\n"
        "- 半径随时间变化：题目附件2（PCHIP 拟合，见 `05_图片/G3_radius`）；\n"
        "- 物性公式：题目附录 2/3/4，实现见 "
        "`02_代码/00_公共内核/src/numerics/properties.py`。\n\n"
        "## 声明\n\n"
        "- 除题目给定的附件与附录外，未使用任何外部实测数据；\n"
        "- 除数据扰动 Monte Carlo 部分（固定随机种子）外，全部计算不含随机数；\n"
        "- 参考文献的 DOI 已逐条反查，发现并更正 4 条、弃用 1 条，详见 "
        "`03_参考文献/引用核实与影响分析.md` 与论文附录 E。\n",
        encoding="utf-8")
    P("    README.md 已生成")


# ==========================================================================
def stage_root():
    P("\n[根] AI 工具使用详情")
    cp(ROOT / "论文" / "AI 工具使用详情.pdf", OUT / "AI 工具使用详情.pdf")
    # 🔴 论文正本**不放入支撑材料**：
    #    《格式规范》第十条要求参赛论文是**单独一个文件**，
    #    第十一条列举的支撑材料内容也不含论文本身；且它占 8.7 MB，
    #    会把压缩包推到 20 MB 上限之上（实测 32.3 MB）。


# 本脚本管理的子目录/文件 —— 重新构建时只清理这些，不动别人放的东西
MANAGED = ["01_result", "02_代码", "03_参考文献", "04_提示词与分工清单",
           "05_图片", "AI 工具使用详情.pdf", "README.md"]


def main():
    mk(OUT)
    # ⚠️ 首版用 shutil.rmtree(OUT) 清空整个目录 —— 那是**不可逆删除**，
    #    且当时并未先查看该目录里原本有什么。现改为**只删本脚本管理的项**。
    for name in MANAGED:
        p = OUT / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    stage_result()
    stage_code()
    stage_refs()
    stage_prompt()
    stage_figs()
    stage_readme()          # ⚠️ 必须在 MANAGED 清理之后、其余 stage 之后（要统计文件数）
    stage_root()
    P(f"\n完成：{OUT}")
    for p in sorted(OUT.iterdir()):
        kind = "目录" if p.is_dir() else "文件"
        cnt = f"{len(list(p.rglob('*')))} 项" if p.is_dir() else f"{p.stat().st_size/1024:.0f} KB"
        P(f"  {p.name:26s} {kind}  {cnt}")


if __name__ == "__main__":
    main()
