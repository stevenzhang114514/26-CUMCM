"""
绘图公共设施                          [丙 · Day1下午起]

统一字体、统一导出。
文件命名约定（按分工要求）："xx图，（放置位置）"
    save_fig(fig, "有限体积离散示意", "论文第4章 问题1·数值方法")
    → figs/有限体积离散示意图，（论文第4章 问题1·数值方法）.png / .pdf

统一原则（v3 清单 §4）：
  * 温度与含水率**分子图**展示，**禁用双纵轴**（避免视觉误读）
  * 保存 PDF 矢量版 + PNG(300 dpi)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from ..config import FIG_DIR

_PDF_OK = True


def setup_style():
    """注册中文字体并设置全局样式。"""
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
    plt.rcParams["font.sans-serif"] = names + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False   # 负号显示
    # 数学文本字体：mathtext 默认字体缺 U+2212（减号），需显式指定
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    # 对数轴刻度由 LogFormatterSciNotation 生成 mathtext（如 $10^{-11}$），
    # 其**斜体**字形缺 U+2212（减号），会刷屏 "Font 'default' does not have a glyph"。
    # 改用 regular 体即可（DejaVu Sans 正体有该字形）。
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = ":"
    return names


def save_fig(fig, name: str, location: str, outdir: Path | None = None):
    """
    按 "xx图，（放置位置）" 命名保存 PNG + PDF。
        name     : 图名（不含"图"字），如 "有限体积离散示意"
        location : 放置位置，如 "论文第4章 问题1·数值方法"
    """
    outdir = Path(outdir) if outdir else FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}图，（{location}）"
    paths = []
    png = outdir / f"{stem}.png"
    fig.savefig(png)
    paths.append(png)
    if _PDF_OK:
        try:
            pdf = outdir / f"{stem}.pdf"
            fig.savefig(pdf)
            paths.append(pdf)
        except Exception:
            pass
    plt.close(fig)
    return paths


def plain_log_ticks(ax, axis="y"):
    """
    把对数轴的刻度标签换成**纯文本**（如 "1e-10"）。

    为什么需要：对数轴默认由 LogFormatterSciNotation 生成 mathtext（$10^{-11}$），
    其中的减号是 U+2212，而 matplotlib 的 mathtext 默认字形缺这个码位，
    会刷屏 "Font 'default' does not have a glyph for '−'"。
    图仍能出，但输出里全是警告，且部分环境会渲染成方框。
    """
    import matplotlib.ticker as mticker
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(fmt)
    a.set_minor_formatter(mticker.NullFormatter())


def figsize_single(width=6.4, height=4.2):
    return (width, height)


def figsize_double(width=6.4, height=7.0):
    return (width, height)
