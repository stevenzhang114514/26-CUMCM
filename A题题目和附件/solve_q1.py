# -*- coding: utf-8 -*-
"""
2026 CUMCM A 题 问题1 求解脚本
预热平衡阶段 (0-1800 s) 药材温度与水分浓度变化规律
模型: 无限长圆柱一维轴对称 传热-传质耦合 (附录2 常物性)
  热:  rho*cp dT/dt = (1/r) d/dr (k r dT/dr)
  质:  dC/dt        = (1/r) d/dr (D(C) r dC/dr),  D = 7e-9 exp(-0.89/C)
  边界 r=R:  -k dT/dr = h(T_s - T_inf) + lam*j
             -D dC/dr = h_m (C_s - C_inf),  j = rho_d * h_m (C_s - C_inf)
  初值 T=28 C, C=2.55
求解: 有限体积(守恒形式) + MOL + BDF 隐式积分
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix, csc_matrix

# ---------------- 参数 (附录2) ----------------
R      = 0.02            # m
RHO    = 820.0           # kg/m3
CP     = 2600.0          # J/(kg K)
KK     = 0.36            # W/(m K)
H      = 25.0            # W/(m2 K)
HM     = 8e-7            # m/s
LAM    = 2.26e6          # J/kg  汽化潜热
T0     = 28.0            # C
C0     = 2.55            # kg/kg
RHO_D  = RHO / (1 + C0)  # 干固体密度 (恒定)

def DofC(C):
    return 7e-9 * np.exp(-0.89 / C)

# ---------------- 附件1 边界条件 ----------------
f1 = r'C:\Users\33154\Desktop\国赛\26-CUMCM\A题题目和附件\附件\附件1.xlsx'
raw = pd.read_excel(f1, header=0).values
tb = raw[:, 0].astype(float)          # s
Tinf = PchipInterpolator(tb, raw[:, 1].astype(float))
Cinf = PchipInterpolator(tb, raw[:, 2].astype(float))

# ---------------- 有限体积离散 ----------------
def solve(N, rtol=1e-7, atol_T=1e-8, atol_C=1e-10):
    dr = R / N
    rf = np.arange(N + 1) * dr              # 界面 0..R
    rc = (np.arange(N) + 0.5) * dr          # 中心
    A  = 2 * np.pi * rf                     # 单位长度界面面积
    V  = np.pi * (rf[1:]**2 - rf[:-1]**2)   # 单位长度控制体体积

    def rhs(t, y):
        T = y[:N]
        C = y[N:2 * N]
        C = np.maximum(C, 1e-12)
        D = DofC(C)
        Df = 0.5 * (D[:-1] + D[1:])         # 界面扩散系数(算术平均)

        # ---- 质量方程 ----
        flux_in = np.zeros(N + 1)           # 界面通量 f = +D dC/dr
        flux_in[1:N] = Df * (C[1:] - C[:-1]) / dr
        # 表面: 半网格阻力 + 对流传质阻力
        g_out = (C[-1] - Cinf(t)) / (dr / (2 * D[-1]) + 1 / HM)   # 向外为正
        flux_in[N] = -g_out
        j = RHO_D * g_out                   # 水质量通量 kg/(m2 s), 向外为正
        dC = (A[1:] * flux_in[1:] - A[:-1] * flux_in[:-1]) / V

        # ---- 能量方程 ----
        q = np.zeros(N + 1)                 # 界面通量 q = +k dT/dr
        q[1:N] = KK * (T[1:] - T[:-1]) / dr
        # 表面: q_in(向内) = (Tinf - T[-1] - lam*j/H) / (dr/(2k) + 1/H)
        q_in = (Tinf(t) - T[-1] - LAM * j / H) / (dr / (2 * KK) + 1 / H)
        q[N] = q_in
        dT = (A[1:] * q[1:] - A[:-1] * q[:-1]) / (RHO * CP * V)
        # 累积量: 累计蒸发水量 / 累计净流入热量 (用于守恒检验)
        dM = A[-1] * j
        dE = A[-1] * q_in
        return np.concatenate([dT, dC, [dM, dE]])

    y0 = np.concatenate([np.full(N, T0), np.full(N, C0), [0.0, 0.0]])
    # 雅可比稀疏结构 (三对角 + 边界耦合)
    sp = lil_matrix((2 * N + 2, 2 * N + 2))
    for i in range(N):
        sp[i, max(0, i - 1):i + 2] = 1
    sp[N - 1, 2 * N - 1] = 1          # 能量边界依赖 C_{N-1}
    for i in range(N, 2 * N):
        sp[i, max(N, i - 1):i + 2] = 1
    sp[2 * N, N - 1] = 1; sp[2 * N, 2 * N - 1] = 1
    sp[2 * N + 1, N - 1] = 1; sp[2 * N + 1, 2 * N - 1] = 1
    sol = solve_ivp(rhs, (0, 1800), y0, method='BDF',
                    t_eval=np.arange(1, 1801),
                    rtol=rtol, atol=np.concatenate([np.full(N, atol_T),
                                                    np.full(N, atol_C),
                                                    [1e-10, 1e-6]]),
                    jac_sparsity=csc_matrix(sp))
    assert sol.success, sol.message
    return sol, dr, rc, V, A

sol, dr, rc, V, A = solve(320)
sol2, dr2, rc2, V2, A2 = solve(640)

# ---------------- 输出节点取值 ----------------
rout = np.arange(0, 21) * 0.1 / 100.0       # 0.0..2.0 cm -> m

def pick(T, C, dr_, t):
    """由控制体中心值求输出节点值 (含 r=0 与表面 r=R)"""
    N = len(T)
    D = DofC(C)
    # r=0: 二次外推 (对称)
    Tc0 = (9 * T[0] - T[1]) / 8
    Cc0 = (9 * C[0] - C[1]) / 8
    # 表面: 由通量平衡求 T_s, C_s
    g_out = (C[-1] - Cinf(t)) / (dr_ / (2 * D[-1]) + 1 / HM)
    j = RHO_D * g_out
    q_in = (Tinf(t) - T[-1] - LAM * j / H) / (dr_ / (2 * KK) + 1 / H)
    Ts = T[-1] + q_in * dr_ / (2 * KK)
    Cs = C[-1] - g_out * dr_ / (2 * D[-1])
    return Tc0, Ts, Cc0, Cs

def sample(sol_, dr_, rc_):
    N = len(rc_)
    nt = sol_.y.shape[1]
    Tt = np.zeros((nt, 21))
    Ct = np.zeros((nt, 21))
    for k in range(nt):
        t = sol_.t[k]
        T = sol_.y[:N, k]
        C = sol_.y[N:2 * N, k]
        Tc0, Ts, Cc0, Cs = pick(T, C, dr_, t)
        Tt[k, 0], Ct[k, 0] = Tc0, Cc0
        Tt[k, -1], Ct[k, -1] = Ts, Cs
        Tt[k, 1:-1] = np.interp(rout[1:-1], rc_, T)
        Ct[k, 1:-1] = np.interp(rout[1:-1], rc_, C)
    return Tt, Ct

T1, C1 = sample(sol, dr, rc)
T2, C2 = sample(sol2, dr2, rc2)
dT = np.abs(T1 - T2)
dC = np.abs(C1 - C2)
kT = np.unravel_index(dT.argmax(), dT.shape)
kC = np.unravel_index(dC.argmax(), dC.shape)
print('网格无关性 (N=320 vs 640) 最大偏差: T %.2e (t=%ds, r=%.1fcm), C %.2e (t=%ds, r=%.1fcm)'
      % (dT.max(), kT[0] + 1, kT[1] * 0.1, dC.max(), kC[0] + 1, kC[1] * 0.1))
# Richardson 外推 (二阶收敛): 细化误差约 (细-粗)/3
TR = T2 + (T2 - T1) / 3
CR = C2 + (C2 - C1) / 3
T1, C1, sol, dr, rc, V, A = TR, CR, sol2, dr2, rc2, V2, A2

# ---------------- 守恒检验 (用 ODE 内部累积量, N=640) ----------------
N = len(rc)
C_end = sol.y[N:2 * N, -1]
T_end = sol.y[:N, -1]
mass0 = RHO_D * C0 * V.sum()
mass_end = RHO_D * (C_end * V).sum()
evap = sol.y[2 * N, -1]
print('质量守恒: 失水 %.6e, 累计蒸发 %.6e, 相对偏差 %.2e'
      % (mass0 - mass_end, evap, abs((mass0 - mass_end) - evap) / (mass0 - mass_end)))
E0 = RHO * CP * T0 * V.sum()
E_end = RHO * CP * (T_end * V).sum()
heat = sol.y[2 * N + 1, -1]
print('能量守恒: 内能增量 %.6e, 累计净流入热 %.6e, 相对偏差 %.2e'
      % (E_end - E0, heat, abs((E_end - E0) - heat) / (E_end - E0)))

# ---------------- 写 result1.xlsx ----------------
times = np.arange(1, 1801)
cols = ['时间\\到药材中心的距离'] + [round(0.1 * i, 1) for i in range(21)]
dfT = pd.DataFrame(np.round(T1, 4), columns=cols[1:])
dfC = pd.DataFrame(np.round(C1, 4), columns=cols[1:])
dfT.insert(0, cols[0], times)
dfC.insert(0, cols[0], times)

out = r'C:\Users\33154\Desktop\result1.xlsx'
with pd.ExcelWriter(out, engine='openpyxl') as w:
    dfT.to_excel(w, sheet_name='温度', index=False)
    dfC.to_excel(w, sheet_name='水分浓度', index=False)
print('已写出:', out)

# ---------------- 论文用 表1/表2 (7x5 子块) ----------------
ti = [100, 300, 600, 900, 1200, 1500, 1800]
ri = [0, 5, 10, 15, 20]
sub_T = pd.DataFrame(np.round(T1[[t - 1 for t in ti]][:, ri], 4),
                     index=ti, columns=[0, 0.5, 1, 1.5, 2])
sub_C = pd.DataFrame(np.round(C1[[t - 1 for t in ti]][:, ri], 4),
                     index=ti, columns=[0, 0.5, 1, 1.5, 2])
sub_T.to_excel(r'C:\Users\33154\Desktop\国赛\26-CUMCM\A题题目和附件\表1_温度.xlsx')
sub_C.to_excel(r'C:\Users\33154\Desktop\国赛\26-CUMCM\A题题目和附件\表2_水分浓度.xlsx')
print('\n表1 温度 (C):'); print(sub_T.to_string())
print('\n表2 水分浓度 (kg/kg):'); print(sub_C.to_string())
