"""
CODE-30 最小复现材料与运行记录
整理运行入口、配置标识、计算网格、时间精度、
覆盖时间、最大残差、验证状态。
生成：
    report/reproduce_record.md
    report/reproduce_record.json
"""

from pathlib import Path
import json
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import matplotlib

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_env_info():
    """记录环境信息。"""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_config_info(config_id):
    """记录配置信息。
    甲、乙给结果后，把这里替换成他们的 config.json 内容。
    """
    return {
        "config_id": config_id,
        "problem": None,
        "time_unit": "s",
        "space_unit": "cm",
        "temperature_unit": "C",
        "moisture_unit": "kg/kg",
        "time_range": None,
        "space_range": None,
        "grid": {
            "time_step": None,
            "space_step": None,
        },
        "solver": {
            "method": None,
            "time_scheme": None,
            "max_residual": None,
        },
        "validation": {
            "status": None,
            "notes": None,
        },
    }


def get_result_files():
    """列出结果文件。"""
    frozen = ROOT / "results" / "frozen"
    data = ROOT / "data"
    files = []
    if frozen.exists():
        for f in sorted(frozen.glob("*.csv")):
            files.append(str(f.relative_to(ROOT)))
    if data.exists():
        for f in sorted(data.glob("result*.xlsx")):
            files.append(str(f.relative_to(ROOT)))
    return files


def write_markdown(env, configs, files):
    lines = []
    lines.append("# 最小复现材料与运行记录\n")
    lines.append(f"生成时间：{env['generated_at']}\n")

    lines.append("## 1. 环境信息\n")
    lines.append(f"- Python：{env['python_version'].splitlines()[0]}")
    lines.append(f"- 平台：{env['platform']}")
    lines.append(f"- numpy：{env['numpy']}")
    lines.append(f"- pandas：{env['pandas']}")
    lines.append(f"- scipy：{env['scipy']}")
    lines.append(f"- matplotlib：{env['matplotlib']}")
    lines.append("")

    lines.append("## 2. 配置信息\n")
    for cfg in configs:
        lines.append(f"### {cfg['config_id']}\n")
        lines.append(f"- 问题：{cfg['problem']}")
        lines.append(f"- 时间范围：{cfg['time_range']}")
        lines.append(f"- 空间范围：{cfg['space_range']}")
        lines.append(f"- 时间步长：{cfg['grid']['time_step']}")
        lines.append(f"- 空间步长：{cfg['grid']['space_step']}")
        lines.append(f"- 求解方法：{cfg['solver']['method']}")
        lines.append(f"- 时间格式：{cfg['solver']['time_scheme']}")
        lines.append(f"- 最大残差：{cfg['solver']['max_residual']}")
        lines.append(f"- 验证状态：{cfg['validation']['status']}")
        lines.append(f"- 备注：{cfg['validation']['notes']}")
        lines.append("")

    lines.append("## 3. 结果文件\n")
    for f in files:
        lines.append(f"- {f}")
    lines.append("")

    lines.append("## 4. 运行入口\n")
    lines.append("```powershell")
    lines.append("cd D:\\math_modeling\\project")
    lines.append("python io\\code31_template_parser.py")
    lines.append("python io\\code31b_build_templates.py")
    lines.append("python data_prep\\code02_env_interp.py")
    lines.append("python data_prep\\code03_stage_identify.py")
    lines.append("python io\\code16_resample.py")
    lines.append("python io\\code17_excel_export.py")
    lines.append("python io\\code30_reproduce.py")
    lines.append("```")
    lines.append("")

    lines.append("## 5. 待确认项\n")
    lines.append("- [ ] 阶段边界时间是否与甲、乙一致")
    lines.append("- [ ] 14400 s 后环境边界外推规则")
    lines.append("- [ ] 问题2 完整时间范围")
    lines.append("- [ ] 问题3 烘干结束时间")
    lines.append("- [ ] 问题4 烘干结束时间")
    lines.append("- [ ] result4 中间列是物理坐标还是变换坐标")
    lines.append("- [ ] result3 是否包含 0 s")
    lines.append("- [ ] result1–4 A 列是否从 0 开始")
    lines.append("- [ ] 论文表3/4 是 3 h 还是完整过程")

    out = REPORT_DIR / "reproduce_record.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Markdown 已保存:", out)


def write_json(env, configs, files):
    data = {
        "env": env,
        "configs": configs,
        "files": files,
    }
    out = REPORT_DIR / "reproduce_record.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("JSON 已保存:", out)


def main():
    env = get_env_info()

    # 甲、乙给结果后，把 config_id 换成他们的
    configs = [
        get_config_info("p1_v1"),
        get_config_info("p2_v1"),
        get_config_info("p3_v1"),
        get_config_info("p4_v1"),
    ]

    files = get_result_files()

    write_markdown(env, configs, files)
    write_json(env, configs, files)


if __name__ == "__main__":
    main()