#!/bin/sh
# CODE-30 最小复现（甲负责部分：问题1—3 的数值内核与结果）
# 干净环境下依次执行即可复现 results/ 与 figs/
set -e
cd "$(dirname "$0")"
python scripts/check_regression_day1.py   # ① 内核未被改坏（三级回归）
python scripts/run_day3.py --stage sens   # ② 离散敏感度（决定冻结配置）
python scripts/run_day3.py --stage conv   # ③ CODE-19 误差表 / TAB-04
python scripts/run_day3.py --stage freeze # ④ 冻结结果 M0 三件
python scripts/run_day3.py --stage repro  # ⑤ 本记录
