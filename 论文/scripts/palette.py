"""
统一配色与绘图样式                     [论文出图 · 公共模块]

配色
----
论文全图统一使用以下 **6 色分类色板**，按数据序列顺序取用：

    #BD4DA3  品红    第 1 条序列
    #8182BA  蓝紫    第 2 条序列
    #B58581  灰褐    第 3 条序列
    #88C6E2  天蓝    第 4 条序列
    #FBF065  亮黄    第 5 条序列
    #C5E4E7  浅青    第 6 条序列

**网格线、坐标轴、参考线、标注框一律用中性灰**，不占色板 ——
否则"某条灰线是数据还是辅助"会被误读。

两个必须记住的坑（都在本项目里真实踩过）
----------------------------------------
1. 🔴 **SimHei 缺 U+2212（减号）字形**。matplotlib 在对数轴刻度、
   负号、mathtext 里默认用 U+2212，会画成方框并刷屏
   `Font 'default' does not have a glyph for '−'`。
   对策：`axes.unicode_minus=False` **并且**对对数轴/自定义刻度显式给
   ASCII 标签（见 `ascii_log_ticks`）。
2. 🔴 **PDF 与 PNG 都要出**。PNG 300 dpi 供预览，PDF 矢量供 LaTeX 排版；
   只出 PNG 时论文里的线条会被栅格化。

用法
----
    from palette import PALETTE, C, setup, save

    setup()                       # 注册字体 + 全局样式（每个脚本开头调一次）
    ax.plot(x, y, color=C(0))     # 第 1 条序列
    save(fig, "G1_geometry")      # 同时出 figs/G1_geometry.png 与 .pdf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

# ==========================================================================
# 配色
# ==========================================================================
PALETTE = ["#BD4DA3", "#8182BA", "#B58581", "#88C6E2", "#FBF065", "#C5E4E7"]

# 语义化别名 —— 跨图引用同一含义时用，避免"第几个序列"随图而变
C_TEMP = PALETTE[0]      # 温度 `#BD4DA3`
C_MOIST = PALETTE[1]     # 含水率 `#8182BA`
C_CENTER = PALETTE[2]    # 中心 `#B58581`
C_SURF = PALETTE[3]      # 表面 `#88C6E2`
C_MEAN = PALETTE[4]      # 平均 `#FBF065`
C_REF = PALETTE[5]       # 参照/对照 `#C5E4E7`

# 辅助元素（不占色板）
GRAY = "#7F7F7F"         # 网格、坐标轴
GRAY_L = "#BFBFBF"       # 次级网格
GRAY_D = "#4D4D4D"       # 文字标注
INK = "#1A1A1A"          # 主要文字


def C(i: int) -> str:
    """按序取色（自动循环）。"""
    return PALETTE[i % len(PALETTE)]


def ramp(n: int, i0: int = 3, i1: int = 1):
    """
    从色板第 i0 色平滑过渡到第 i1 色，返回 n 个颜色。

    用于"同一含义、强弱不同"的场合（如有限体积示意图里的环形控制体
    由内向外的深浅），**不引入色板之外的颜色**。
    """
    import matplotlib.colors as mc
    c0 = np.array(mc.to_rgb(PALETTE[i0 % len(PALETTE)]))
    c1 = np.array(mc.to_rgb(PALETTE[i1 % len(PALETTE)]))
    t = np.linspace(0.0, 1.0, n)[:, None]
    return [mc.to_hex((1 - t[k]) * c0 + t[k] * c1) for k in range(n)]


# ==========================================================================
# 字体与全局样式
# ==========================================================================
FIGS = Path(__file__).resolve().parents[1] / "figs"


def setup() -> list[str]:
    """注册中文字体并设置全局 rcParams。每个出图脚本开头调用一次。"""
    cand = [
        ("C:/Windows/Fonts/simhei.ttf", "SimHei"),
        ("C:/Windows/Fonts/msyh.ttc", "Microsoft YaHei"),
        ("C:/Windows/Fonts/simsun.ttc", "SimSun"),
    ]
    names = []
    for path, name in cand:
        try:
            if Path(path).exists():
                font_manager.fontManager.addfont(path)
                names.append(name)
        except Exception:
            pass
    if not names:
        names = ["DejaVu Sans"]

    plt.rcParams.update({
        "font.sans-serif": names + ["DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,          # 坑 1：不要用 U+2212
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        # ---- 色板 ----
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        # ---- 尺寸与样式 ----
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.edgecolor": GRAY_D,
        "axes.linewidth": 0.9,
        "axes.labelcolor": INK,
        "xtick.color": GRAY_D,
        "ytick.color": GRAY_D,
        "grid.color": GRAY_L,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "axes.grid": True,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": GRAY_L,
    })
    return names


# ==========================================================================
# 对数轴 / 自定义刻度的 ASCII 标签（坑 1 的补丁）
# ==========================================================================
def ascii_log_ticks(ax, axis="x", ticks=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1000),
                    labels=None):
    """
    给对数轴换**纯 ASCII** 的刻度标签。

    默认的 `LogFormatterSciNotation` 会生成 mathtext `$10^{-4}$`，
    其中的减号是 U+2212，SimHei 无该字形 → 方框 + 警告刷屏。
    """
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(list(ticks)))
    a.set_minor_formatter(NullFormatter())
    if labels is None:
        labels = [f"{t:g}" for t in ticks]
    if axis == "x":
        ax.set_xticklabels(labels)
    else:
        ax.set_yticklabels(labels)
    return ax


def ascii_num_ticks(ax, axis="y"):
    """把某个轴的刻度改成 f"{v:g}" 纯 ASCII（避免 mathtext 的 U+2212）。"""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())
    return ax


# ==========================================================================
# 保存
# ==========================================================================
def save(fig, name: str, outdir: Path | None = None, also_pdf: bool = True):
    """同时保存 PNG(300 dpi) 与 PDF(矢量)。返回路径列表。"""
    outdir = Path(outdir) if outdir else FIGS
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    png = outdir / f"{name}.png"
    fig.savefig(png)
    paths.append(png)
    if also_pdf:
        pdf = outdir / f"{name}.pdf"
        fig.savefig(pdf)
        paths.append(pdf)
    plt.close(fig)
    print(f"    → {png.name}" + (f" + {pdf.name}" if also_pdf else ""), flush=True)
    return paths


def finish(ax, title=None, xlabel=None, ylabel=None, legend=False, loc="best"):
    """常用收尾：标题、轴标签、图例、网格。"""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.75)
    if legend:
        ax.legend(loc=loc, fontsize=9.5)
    return ax
