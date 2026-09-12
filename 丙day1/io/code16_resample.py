"""
CODE-16 采样/重映射
把甲、乙的计算结果，采样到题目要求的时间点和距离点上。
现在还没有甲、乙的结果，所以先写框架，用模拟数据测试。

输入（将来由甲、乙提供）：
    results/raw/p1_temperature_*.csv
    results/raw/p1_moisture_*.csv
    ...
输出：
    results/frozen/p1_temperature_sampled.csv
    results/frozen/p1_moisture_sampled.csv
    ...
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
FROZEN_DIR = ROOT / "results" / "frozen"
FROZEN_DIR.mkdir(parents=True, exist_ok=True)


def load_raw(path):
    """读甲、乙给的 CSV。
    约定：
    - 第 1 列是时间，单位 s
    - 其余列是距离，单位 cm
    - 第 1 行是列名
    """
    df = pd.read_csv(path)
    t = df.iloc[:, 0].to_numpy(dtype=float)
    r = df.columns[1:].astype(float).to_numpy()
    values = df.iloc[:, 1:].to_numpy(dtype=float)
    return t, r, values


def build_interpolator(t, r, values):
    """构建二维插值器：时间 × 空间。"""
    interp = RegularGridInterpolator(
        (t, r), values,
        method="linear",
        bounds_error=False,
        fill_value=None,  # 线性外推
    )
    return interp


def resample(t, r, values, t_target, r_target):
    """把结果采样到目标时间、目标距离。"""
    interp = build_interpolator(t, r, values)

    # 构造网格
    TT, RR = np.meshgrid(t_target, r_target, indexing="ij")
    pts = np.stack([TT.ravel(), RR.ravel()], axis=-1)
    sampled = interp(pts).reshape(len(t_target), len(r_target))
    return sampled


def make_target_axes(problem):
    """按题目要求生成目标时间轴和距离轴。"""
    r_target = np.round(np.arange(0, 2.0 + 1e-9, 0.1), 1)

    if problem == 1:
        t_target = np.arange(0, 1800 + 1, 1)
    elif problem == 2:
        t_target = np.arange(0, 10800 + 1, 1)   # 先按 3h
    elif problem == 3:
        t_target = np.arange(60, 259200 + 1, 60)
    elif problem == 4:
        t_target = np.arange(60, 259200 + 1, 60)
    else:
        raise ValueError(problem)

    return t_target, r_target


def check_no_negative(sampled, name):
    """检查采样结果是否出现负值。"""
    if np.any(sampled < 0):
        print(f"[警告] {name} 出现负值: min = {sampled.min()}")
    else:
        print(f"[OK] {name} 无负值")


def check_no_nan(sampled, name):
    """检查采样结果是否出现 NaN。"""
    if np.any(np.isnan(sampled)):
        print(f"[警告] {name} 出现 NaN")
    else:
        print(f"[OK] {name} 无 NaN")


def resample_one(problem, var, raw_path, out_name):
    """处理一个变量（温度或水分浓度）。"""
    print(f"\n=== 问题{problem} {var} ===")
    t, r, values = load_raw(raw_path)
    print("原始网格：", t.shape, r.shape, values.shape)

    t_target, r_target = make_target_axes(problem)
    print("目标网格：", t_target.shape, r_target.shape)

    sampled = resample(t, r, values, t_target, r_target)
    check_no_nan(sampled, var)
    check_no_negative(sampled, var)

    # 保存
    df = pd.DataFrame(sampled, columns=[f"{x:.1f}" for x in r_target])
    df.insert(0, "time_s", t_target)
    out_path = FROZEN_DIR / out_name
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("已保存:", out_path)


def test_with_synthetic():
    """用模拟数据测试整条采样管线。"""
    print("=== 模拟数据测试 ===")
    t = np.linspace(0, 1800, 100)
    r = np.linspace(0, 2.0, 50)
    TT, RR = np.meshgrid(t, r, indexing="ij")
    values = 28 + 10 * (1 - np.exp(-TT / 600)) * (1 - 0.1 * RR)

    t_target, r_target = make_target_axes(1)
    sampled = resample(t, r, values, t_target, r_target)
    check_no_nan(sampled, "模拟温度")
    check_no_negative(sampled, "模拟温度")
    print("模拟采样形状：", sampled.shape)


def main():
    # 先跑模拟测试
    test_with_synthetic()

    # 等甲、乙给结果后，取消下面注释
    # resample_one(1, "temperature",
    #              RAW_DIR / "p1_temperature_p1_v1.csv",
    #              "p1_temperature_sampled.csv")
    # resample_one(1, "moisture",
    #              RAW_DIR / "p1_moisture_p1_v1.csv",
    #              "p1_moisture_sampled.csv")


if __name__ == "__main__":
    main()