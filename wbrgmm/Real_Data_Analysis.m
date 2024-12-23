%% Description
% Real data analysis of Old faithful geyser erruption data
Y_faithful = csvread('faithful_data.csv', 1, 1);
%% Initialization
% Running RGM on faithful data
[n, p] = size(Y_faithful);
a0 = 1; b0 = 1; lsig2 = 0.01; 
usig2 = 10000; B = 1000; nmc = 1000; 
tau = 10; g0 = 1; m = 2;
t = 5;
K_max = 20;
log_V = log_V_nt(n, K_max);
Z = numerical_ZK(K_max, tau, p, g0);
tic;
[gamma_mc, Gamma_mc, K_mc] = blocked_collapsed_Gibbs(Y_faithful', B, nmc, log_V, a0, b0, tau, g0, lsig2, usig2, Z);
toc;
plot(K_mc)
% g0_seq = 1:10;
% reject_number_seq = zeros(1, length(g0_seq));
% for g0 = 1:10
%     tic;
%     [~, ~, ~, number_reject] = blocked_collapsed_Gibbs(Y_faithful', B, nmc, log_V, a0, b0, tau, g0, lsig2, usig2, Z);
%     toc;
%     reject_number_seq(g0) = number_reject;
% end
% reject_number_seq
save('faithful_RGM_result_2Revision.mat')
