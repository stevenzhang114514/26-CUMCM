%% 几何—密度一致性诊断（修正版）
clear; clc; close all;

%% 1. 读取问题4 结果
base_dir = '/Users/solacelumi/Downloads/CUMCM2026Problems/A题/附件/';
result_file = [base_dir, 'result4.xlsx'];
radius_file = [base_dir, 'R_fit.mat'];

C_data = readmatrix(result_file, 'Sheet', 'Sheet1');
t_out = C_data(2:end, 1);
C_mat = C_data(2:end, 2:21);
C_surf = C_data(2:end, 22);

S = load(radius_file);
R_fit = S.R_fit;
Rof = @(t) R_fit(t) / 100;

%% 2. 物性
rho_fun = @(C) 760 + 90 * C;
rho_d_fun = @(C) rho_fun(C) ./ (1 + C);

%% 3. 数值积分（修正版）
L = 0.25;
dr = 0.0005;   % 更细的积分步长

n_time = length(t_out);
M_d = zeros(n_time, 1);

r_fixed_cm = 0:0.1:1.9;

for i = 1:n_time
    tk = t_out(i);
    Rk = Rof(tk);           % 当前半径 m
    Rk_cm = Rk * 100;       % 当前半径 cm
    
    % 只取当前半径以内的点
    mask = r_fixed_cm <= Rk_cm + 1e-9;
    r_valid_cm = r_fixed_cm(mask);
    C_valid = C_mat(i, mask);
    
    % 加上表面点
    r_all_cm = [r_valid_cm, Rk_cm];
    C_all = [C_valid, C_surf(i)];
    
    % 转成 m
    r_all = r_all_cm / 100;
    
    % 插值到精细网格
    r_fine = 0:dr:Rk;
    C_fine = interp1(r_all, C_all, r_fine, 'pchip', 'extrap');
    C_fine = max(C_fine, 0);
    
    % 积分
    rho_d_fine = rho_d_fun(C_fine);
    integrand = rho_d_fine .* r_fine;
    M_d(i) = 2 * pi * L * trapz(r_fine, integrand);
end

%% 4. e_d
M_d_0 = M_d(1);
e_d = (M_d - M_d_0) / M_d_0;

%% 5. 输出
fprintf('=== 几何—密度一致性诊断（修正版）===\n');
fprintf('时间(h)\tM_d(kg)\t\te_d\n');
idx_print = round(linspace(1, n_time, 10));
for i = idx_print
    fprintf('%.2f\t\t%.6f\t%.4e\n', t_out(i)/3600, M_d(i), e_d(i));
end

%% 6. 画图
figure('Color', 'w');
plot(t_out/3600, e_d, 'b-', 'LineWidth', 1.5);
xlabel('时间 t / h');
ylabel('干物质质量相对变化 e_d');
title('几何—密度一致性诊断（修正版）');
grid on;
ylim([-0.1, 0.1]);

%% 7. 上界检查
rho_d_0 = rho_d_fun(2.55);
R_f_min = 0.02 * sqrt(rho_d_0 / 760);

fprintf('\n=== 条件性上界检查 ===\n');
fprintf('初始干固体密度 rho_d(2.55) = %.4f kg/m^3\n', rho_d_0);
fprintf('理论半径下界 R_f_min = %.4f cm\n', R_f_min * 100);
fprintf('实测终态半径 R_f = %.4f cm\n', Rof(t_out(end)) * 100);
fprintf('差额 = %.4f cm\n', (R_f_min - Rof(t_out(end))) * 100);