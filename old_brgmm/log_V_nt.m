function log_V = log_V_nt(n, t_max)
%% Description:
%  This function computes V_n(t) using the finite approximation to the
%  infinite sum for V_n(t).
%  Also see the Julia script written by Jeffrey Miller at
%  https://github.com/jwmi/BayesianMixtures.jl/blob/master/src/MFM.jl
%% Input:
%  n     - number of observations
%  t_max - estimated upper bound on number of partitions
%% Output:
%  V - matrix of V_n(t)'s with n = 1,...,n_max and t = 1,...,K_max
%% Initialize
log_V = zeros(1, t_max);
tol = 1e-12;
for t = 1:t_max
    if t > n
        log_V(t) = -Inf; 
    else
        a = 0;
        c = -Inf;
        k = 1;
        p = 0;
        while (abs(a - c) > tol || (p < 1.0 - tol))
            %   Note: The first condition is false when a = c = -Inf
            if k >= t
                a = c;
                b = gammaln(k + 1) - gammaln(k - t + 1) - gammaln(k + n) ...
                    +gammaln(k) + log(1/(exp(1) - 1)/factorial(k));
                m = max(a, b);
                if m == -Inf
                    c = -Inf;
                else
                    c = m + log(exp(a - m) + exp(b - m));
                end
            end
            p = p + exp(log(1/(exp(1) - 1)/factorial(k)));
            k = k + 1;
        end
        log_V(t) = c;
    end
end
end