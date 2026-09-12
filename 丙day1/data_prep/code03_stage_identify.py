"""
CODE-03 阶段识别
根据附件1，识别预热平衡阶段和恒温干燥阶段
输出：
    report/stage_identification.md
    figs/G2b_stage.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"
FIGS_DIR = ROOT / "figs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_env():
    df = pd.read_csv(DATA_DIR / "env_interp.csv")
    return df["time_s"].to_numpy(), df["T_inf_C"].to_numpy(), df["C_inf_kg_per_kg"].to_numpy()


def identify_stages(t, T, C):
    """
    识别阶段：
    - 预热平衡阶段：温度明显上升，未趋于稳定
    - 恒温干燥阶段：温度趋于稳定，接近最终温度
    """
    T_final = T[-1]
    T_start = T[0]
    T_range = T_final - T_start

    # 阈值：温度达到最终升温的 95%
    T_95 = T_start + 0.95 * T_range
    idx_95 = np.argmax(T >= T_95)
    t_95 = t[idx_95]

    # 温度变化率
    dTdt = np.gradient(T, t)

    # 温度变化率小于 0.001 ℃/s 认为趋于稳定
    stable_idx = np.where(np.abs(dTdt) < 0.001)[0]
    if len(stable_idx) > 0:
        t_stable = t[stable_idx[0]]
    else:
        t_stable = t[-1]

    return {
        "T_start": T_start,
        "T_final": T_final,
        "T_95": T_95,
        "t_95": t_95,
        "t_stable": t_stable,
        "dTdt_max": dTdt.max(),
        "dTdt_min": dTdt.min(),
    }


def plot_stage(t, T, C, info):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, T, "-", lw=1.2, label="T_inf(t)")
    axes[0].axvline(info["t_95"], color="orange", ls="--",
                    label=f"95% 升温点 t={info['t_95']:.0f}s")
    axes[0].axvline(info["t_stable"], color="red", ls="--",
                    label=f"趋于稳定 t={info['t_stable']:.0f}s")
    axes[0].set_ylabel("温度 / ℃")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, C, "-", lw=1.2, label="C_inf(t)")
    axes[1].axvline(info["t_95"], color="orange", ls="--")
    axes[1].axvline(info["t_stable"], color="red", ls="--")
    axes[1].set_xlabel("时间 / s")
    axes[1].set_ylabel("水分浓度 / (kg/kg)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("G2b 阶段识别")
    fig.tight_layout()

    out_png = FIGS_DIR / "G2b_stage.png"
    out_pdf = FIGS_DIR / "G2b_stage.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("图已保存:", out_png)
    print("图已保存:", out_pdf)


def write_report(info):
    lines = []
    lines.append("# 阶段识别报告\n")
    lines.append("## 1. 附件1 环境数据\n")
    lines.append(f"- 温度起始值: {info['T_start']:.4f} ℃")
    lines.append(f"- 温度最终值: {info['T_final']:.4f} ℃")
    lines.append(f"- 95% 升温点: t = {info['t_95']:.0f} s")
    lines.append(f"- 趋于稳定点: t = {info['t_stable']:.0f} s")
    lines.append(f"- 最大升温速率: {info['dTdt_max']:.6f} ℃/s")
    lines.append(f"- 最小升温速率: {info['dTdt_min']:.6f} ℃/s")
    lines.append("")
    lines.append("## 2. 阶段划分\n")
    lines.append("### 预热平衡阶段")
    lines.append(f"- 时间范围: 0 ~ {info['t_95']:.0f} s")
    lines.append("- 特征: 温度快速上升，水分浓度缓慢上升")
    lines.append("- 对应: 问题1")
    lines.append("")
    lines.append("### 恒温干燥阶段")
    lines.append(f"- 时间范围: {info['t_95']:.0f} s ~ 烘干结束")
    lines.append("- 特征: 温度趋于稳定，水分浓度趋于稳定")
    lines.append("- 对应: 问题2、问题3、问题4")
    lines.append("")
    lines.append("## 3. 待确认项\n")
    lines.append("- [ ] 阶段边界时间是否与甲、乙一致")
    lines.append("- [ ] 14400 s 后环境边界外推规则")
    lines.append("- [ ] 问题2 完整时间范围")

    out = REPORT_DIR / "stage_identification.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("报告已保存:", out)


def main():
    t, T, C = load_env()
    info = identify_stages(t, T, C)

    print("阶段识别结果:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    plot_stage(t, T, C, info)
    write_report(info)


if __name__ == "__main__":
    main()