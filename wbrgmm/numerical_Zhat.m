% Numerical Computation of ZK as in algorithm 1
function Zhat = numerical_Zhat(gamma_pos_mc, Gamma_pos_mc, g_0, t, m)
    [p, ~, M] = size(gamma_pos_mc);
    g = zeros(m, M);
    Zhat = zeros(1, m);
    for n = 1:M
        for K = t:(t + m - 1)
            g_tmp = ones(K, K);
            for i = 1:(K - 1)
                for j = (i + 1):K
                    d = wasserstein(gamma_pos_mc(:, i, n), Gamma_pos_mc(:, :, i, n), gamma_pos_mc(:, j, n), Gamma_pos_mc(:, :, j, n));
                    g_tmp(i, j) = d/(g_0 + d); 
                end
            end
            g(K - t + 1, n) = min(min(g_tmp));
        end
    end
    for K = t:(t + m - 1)
        Zhat(K - t + 1) = log(mean(g(K - t + 1, :)));
    end 
end