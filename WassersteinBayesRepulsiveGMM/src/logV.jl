function logV_nt(n, t_max)::Vector{Float64} 
    log_V = zeros(t_max)
    tol = 1e-12 

    log_exp_m_1 = log(exp(1) - 1)
    @inbounds for t = 1:t_max
        log_V[t] = -Inf
        if t <= n 
            a, c, k, p = 0, -Inf, 1, 0
            while abs(a - c) > tol || p < 1.0 - tol
                # Note: The first condition is false when a = c = -Inf
                if k >= t
                    a = c 
                    # b = loggamma(k + 1) - loggamma(k - t + 1) - loggamma(k + n)  
                    # b += loggamma(k) - log_exp_m_1 - logfactorial(k) 
                    b =  - loggamma(k - t + 1) - loggamma(k + n)  
                    b += loggamma(k) - log_exp_m_1  
                    m = max(a, b);
                    c = m == -Inf ? -Inf : m + log(exp(a - m) + exp(b - m))
                end
                p += exp(-log_exp_m_1 - logfactorial(k))
                k += 1 
            end
            log_V[t] = c
        end 
    end
    return log_V
end