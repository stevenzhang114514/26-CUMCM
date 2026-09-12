%% problem4_drying_B.m  （固定半径一致性检验版）
clear; clc; close all;

%% 0. 路径
att1    = '/Users/solacelumi/Downloads/CUMCM2026Problems/A题/附件/附件1.xlsx';
rmat    = '/Users/solacelumi/Downloads/CUMCM2026Problems/A题/附件/R_fit.mat';
outfile = '/Users/solacelumi/Downloads/CUMCM2026Problems/A题/附件/result4_B.xlsx';

T_stage = 50;  C_stage = 0.05;

%% 1. 环境
useAtt1 = exist(att1,'file')==2;
if useAtt1
    raw = readmatrix(att1);  raw = raw(~any(isnan(raw),2),:);
    t_env = raw(:,1);  Tinf_v = raw(:,2);  Cinf_v = raw(:,3);
    ppT = pchip(t_env,Tinf_v);  ppC = pchip(t_env,Cinf_v);
    Tinf = @(t) ppval(ppT, min(t,t_env(end)));
    Cinf = @(t) ppval(ppC, min(t,t_env(end)));
else
    Tinf = @(t) T_stage;  Cinf = @(t) C_stage;
end
Tenv = @(t) (t<=1800 && useAtt1) .* Tinf(t) + (t>1800 || ~useAtt1) .* T_stage;
Cenv = @(t) (t<=1800 && useAtt1) .* Cinf(t) + (t>1800 || ~useAtt1) .* C_stage;

%% 2. 半径——★固定为2cm★
S = load(rmat);  R_fit = S.R_fit;
% Rof = @(t) R_fit(t)/100;              % 原版：半径随时间变
Rof = @(t) 0.02 * ones(size(t));        % ★固定版：半径恒为 2cm★

%% 3. 网格
N = 20;  dxi = 1/(N-1);
xi = (0:N-1).'*dxi;
w = xi*dxi;  w(1) = (dxi/2)^2/2;  w(end) = (1-(1-dxi/2)^2)/2;
xf = ((1:N-1).'-0.5)*dxi;
h  = 25;   hm = 8e-7;

T = 28*ones(N,1);  C = 2.55*ones(N,1);
dt = 10;  tmax = 432000;  Ctarget = 0.15;
tsample = (60:60:tmax).';  isamp = 1;
Crec = zeros(numel(tsample),N);  Trec = zeros(numel(tsample),N);
trec = zeros(numel(tsample),1);

%% 4. 主循环
t = 0;  tdry = NaN;
while t < tmax
    Co = C;  To = T;  R = Rof(t);
    Te = Tenv(t);  Ce = Cenv(t);

    for it = 1:10
        TK = T + 273.15;
        rho = 760 + 90*C;
        cp  = 1850 + 2150*C./(C+1);
        k   = 0.12 + 0.20*C./(C+1);
        D   = 4.2e-4 .* exp(-0.30./C) .* exp(-3850./TK);

        Df = 2*D(1:end-1).*D(2:end)./(D(1:end-1)+D(2:end));
        kf = 2*k(1:end-1).*k(2:end)./(k(1:end-1)+k(2:end));

        G  = xf.*Df/(R^2*dxi);
        GT = xf.*kf/(R^2*dxi);

        dl = [0; -G];  du = [-G; 0];
        dd = w/dt + [G;0] + [0;G];  b = w/dt.*Co;
        dd(end) = dd(end) + hm/R;
        b(end)  = b(end)  + hm/R*Ce;
        Cn = thomas(dl,dd,du,b);

        cap = rho.*cp.*w/dt;
        dl = [0; -GT];  du = [-GT; 0];
        dd = cap + [GT;0] + [0;GT];  b = cap.*To;
        dd(end) = dd(end) + h/R;
        b(end)  = b(end)  + h/R*Te;
        Tn = thomas(dl,dd,du,b);

        if max(abs(Cn-C))<1e-8 && max(abs(Tn-T))<1e-8
            C=Cn; T=Tn; break;
        end
        C = Cn;  T = Tn;
    end

    t = t + dt;

    if isamp<=numel(tsample) && t>=tsample(isamp)
        Crec(isamp,:)=C.'; Trec(isamp,:)=T.'; trec(isamp)=t; isamp=isamp+1;
    end
    if isnan(tdry) && max(C)<Ctarget
        tdry = ceil(t/tsample(1))*tsample(1);
    end
    if ~isnan(tdry) && t>=tdry, break; end
end

if isnan(tdry)
    warning('72h 内未达标');  tdry = t;
end
nsamp = find(trec>0,1,'last');
trec=trec(1:nsamp); Crec=Crec(1:nsamp,:); Trec=Trec(1:nsamp,:);

%% 5. 自检
fprintf('========== 版本B（固定半径）结果 ==========\n');
fprintf('t_dry = %.2f h | 终态Cmax = %.4f | 终态T(1) = %.2f\n',...
        tdry/3600, max(Crec(end,:)), Trec(end,1));

%% 6. 画图
figure('Color','w');
plot(trec/3600, Crec(:,1),'b-','LineWidth',1.8); hold on;
plot(trec/3600, Crec(:,end),'r-','LineWidth',1.8);
yline(0.15,'--k'); xline(tdry/3600,':k');
xlabel('t / h'); ylabel('C / (kg·kg^{-1})');
legend('中心 \xi=0','表面 \xi=1'); grid on;
title('版本B：固定半径，附录4 物性');

%% ================= 局部函数 =================
function x = thomas(dl,dd,du,b)
    n=numel(b); cp=zeros(n,1); dp=zeros(n,1);
    cp(1)=du(1)/dd(1); dp(1)=b(1)/dd(1);
    for i=2:n
        m=dd(i)-dl(i)*cp(i-1);
        if i<n, cp(i)=du(i)/m; end
        dp(i)=(b(i)-dl(i)*dp(i-1))/m;
    end
    x=zeros(n,1); x(n)=dp(n);
    for i=n-1:-1:1, x(i)=dp(i)-cp(i)*x(i+1); end
end