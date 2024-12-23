% Numerical Computation of ZK as in algorithm 1
function Z = numerical_ZK(K_max, tau, p, g_0)
% K_max = 50;
% tau = 1;
% p = 2;
% g_0 = 1;
%% Initialize
M = 2000;
mu_mc = tau * randn(p, K_max, M);
g = zeros(K_max, M);
Z = zeros(1, K_max);
for m = 1:M
    for K = 2:K_max
        g_tmp = ones(K, K);
        for i = 1:(K - 1)
            for j = (i + 1):K
                d = sqrt(sum((mu_mc(:, i, m) - mu_mc(:, j, m)).^2));
                g_tmp(i, j) = d/(g_0 + d);
            end
        end
        g(K, m) = min(min(g_tmp));
    end
end
Z(1) = 0;
for K = 2:K_max
    Z(K) = log(mean(g(K,:)));
end
end