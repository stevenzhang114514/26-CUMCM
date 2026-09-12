"""
Day2 出图脚本                              [甲 · Day2]

用法
----
    python scripts/make_figures_day2.py            # 全部
    python scripts/make_figures_day2.py fig16 g12  # 指定

产出（命名遵循分工约定「xx图，（放置位置）」）
---------------------------------------------
    figs/无量纲数分析图，（论文第5章 问题2·模型建立与量级分析）.png/.pdf
    figs/扩散系数等值线图，（论文第5章 问题2·模型建立与量级分析）.png/.pdf
    figs/含水率三曲线对比图，（论文第6章 问题3·达标判据）.png/.pdf
    figs/潜热开关对照图，（论文第9章 模型扩展·蒸发潜热）.png/.pdf
    figs/收敛性检验图，（论文第8章 模型检验·数值验证）.png/.pdf   （需先跑 --stage conv）

🔴 出图只读**已冻结的结果文件**（npz / json），不重算模型。
   若结果更新，必须重新出图，不得手改图。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIG_DIR  # noqa: E402

PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"


def _npz(layer, kind):
    p = PROC / f"core_{layer}_{kind}.npz"
    return p if p.exists() else None


def do_fig16():
    from src.figures import fig16_dimensionless as m
    return m.make(_npz("M0", "3h"))


def do_fig18():
    from src.figures import fig18_D_contour as m
    return m.make(_npz("M0", "full") or _npz("M0", "3h"))


def do_g12():
    from src.figures import g12_curves as m
    p = _npz("M0", "full")
    if p is None:
        print("  [跳过] 缺 core_M0_full.npz（先跑 --stage core）")
        return []
    paths, st = m.make(p, layer="M0")
    (DOCS / "g12_stats.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    print("    G12 统计:", json.dumps(st, ensure_ascii=False))
    return paths


def do_fig43():
    from src.figures import fig43_latent as m
    if _npz("M0", "3h") is None or _npz("M1", "3h") is None:
        print("  [跳过] 缺 core_M0_3h.npz / core_M1_3h.npz")
        return []
    return m.make()


def do_conv():
    from src.figures import fig15_convergence as m
    p = DOCS / "day2_conv.json"
    if not p.exists():
        print("  [跳过] 缺 day2_conv.json（先跑 --stage conv）")
        return []
    return m.make(p)


TASKS = {"fig16": do_fig16, "fig18": do_fig18, "g12": do_g12,
         "fig43": do_fig43, "conv": do_conv}


def main():
    keys = sys.argv[1:] or list(TASKS)
    print(f"Day2 出图: {keys}")
    n = 0
    for k in keys:
        if k not in TASKS:
            print(f"  未知图组 {k!r}，可选 {list(TASKS)}")
            continue
        print(f"\n[{k}]")
        try:
            fs = TASKS[k]()
            for f in fs or []:
                print("   ", f)
                n += 1
        except Exception as e:
            print(f"   **失败**: {type(e).__name__}: {e}")
    print(f"\n共写出 {n} 个文件到 {FIG_DIR}")


if __name__ == "__main__":
    main()
