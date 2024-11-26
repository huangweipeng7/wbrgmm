function [gamma_mc, Gamma_mc, K_mc] = blocked_collapsed_Gibbs(...
    Y, B, nmc, log_V, a0, b0, tau, g0, lsig2, usig2, Z)
%% Description: Generalized-Urn-Model-based sampler for Repulsive MFM
%% Input:
%  Observations (yi)ni=1;
%  Hyperparameters(a0,b0),tau, g0, lsig2,usig2;
%  Burn-in time B;
%  Number of posterior samples nmc;
%  Guess upper bound K_max on K;
%  Perturbation range m for approximate sampling p(K|C).
%  Precomputed coefficients V_n(t) for t <= Kmax
%  Precomputed normalization constant Z's
%% Output:
%  Posterior samples (theta_1^(t),...,theta_n^(t)) for t = 1,...,B + T
%% Initialize:
[p, n] = size(Y);
K = size(Z, 2) / 2;
mu_init = tau * randn(p, K);
Sigma_init = zeros(p, p, K);
gamma_mc = zeros(p, n, B + nmc);
Gamma_mc = zeros(p, p, n, B + nmc);
K_mc = zeros(1, B + nmc);
reject_time = zeros(1, B + nmc);
for k = 1:K
    Sigma_init(:, :, k) = eye(p, p);
end
gam = zeros(p, n);
Gam = zeros(p, p, n);
for i = 1:n
    D = sum((Y(:, i) * ones(1, K) - mu_init).^2, 1);
    [~, k_i] = min(D);
    gam(:, i) = mu_init(:, k_i);
    Gam(:, :, i) = Sigma_init(:, :, k_i);
end
gamma_mc(:, :, 1) = gam;
Gamma_mc(:, :, :, 1) = Gam;
K_mc(1) = K;
m = 2;
% tic;
%% Gibbs sampling
for iter = 2:(B + nmc) 
    gam = gamma_mc(:, :, iter - 1); % gam is gamma
    Gam = Gamma_mc(:, :, :, iter - 1); % Gam is Gamma
    ind_n = 1:n; 
    % Step 1: Given (theta_1,...,theta_n), sample C
    for i = 1:n
        mu1_mc = tau * randn(p, 30);
        mu2_mc = tau * randn(p, 30);
        minus_i = ind_n(ind_n ~= i); % minus_i is the indices 1:n excluding i
        gam_minus_i = gam(:, minus_i); % gamma_{-i}
        Gam_minus_i = Gam(:, :, minus_i); % Gamma_{-i}
        gam_star_1 = unique(gam_minus_i(1, :)); % First coordinate of gamma_star
        t = length(gam_star_1); % Number of partition t = |C_{-i}|
        % Initialize gamma_star and Gamma_star
        gam_star = zeros(p, K);
        Gam_star = zeros(p, p, K);
        % Label the unique values of (theta_1,...,theta_n) as theta_star's
        N = zeros(1, t);
        for k = 1:t
            j = 1;
            while (gam_minus_i(1, j) ~= gam_star_1(k))
                j = j + 1;
            end
            N(k) = sum(gam_minus_i(1, :) == gam_star_1(k));
            gam_star(:, k) = gam_minus_i(:, j);
            Gam_star(:, :, k) = Gam_minus_i(:, :, j);
        end
        % Now check if the ith observation forms a single cluster
        i_singleton = true;
        for k = 1:t
            if (abs(gam(1, i) - gam_star_1(k)) < 10 ^ (-8))
                i_singleton = false;
            end
        end
        % Initialize the auxiliary variables (gam_star_c, Gam_star_c)
        gam_star_c = zeros(p, 1);
        Gam_star_c = zeros(p, p, 1);
        if (i_singleton == true) 
            % If ith observation is a singleton, 
            % then assign (gam_star_c, Gam_star_c) = theta_i
            gam_star_c = gam(:, i);
            Gam_star_c = Gam(:, :, i);
        else
            % Otherwise sample (gam_star_c, Gam_star_c) from G_tilde as follows
            % Compute I0, I1, and I2
            [logI0, logI1, logI2] = Numerical_I(gam_star, Z, g0, mu1_mc, mu2_mc);
            % Sampling K ~ p(K | t + 1, theta_{-i}) approximately
            log_prob = zeros(1, 2);
            log_prob(1) = log(t + n + 2) + logI1;
            log_prob(2) = log(t + 2) + logI2;
%             for K = (t + 1):(t + m)
%                 log_prob(K - t) = gammaln(K + 1) - gammaln(K - t + 1) - gammaln(K + n + 1);
%             end
            log_prob = log_prob - max(log_prob);
            pr = exp(log_prob);
            pr = pr/sum(pr);
            tmp = sum((rand(1, 1) >= cumsum(pr))) + 1;
            K = t + tmp;
            % Sample extra components that are not associated with observations
            for k = (t + 1): K
                lambda = gamrnd(a0, 1/b0, p, 1);
                while ((min(lambda) < usig2^(-1)) || (max(lambda) > lsig2^(-1)))
                    lambda = gamrnd(a0, 1/b0, p, 1);
                end
                Gam_star(:, :, k) = eye(p, p)/diag(lambda);
            end
            % Accept-Reject sampling from gamma_star's associated with
            % no observations
            g = zeros(K, K);
            while (rand(1) >= min(min(g)))
                gam_star(:, (t + 1): K) = tau/2 * randn(p, K - t);
                g = ones(K, K);
                for k1 = 1:(K - 1)
                    for k2 = (k1 + 1):K
                        d = sum((gam_star(:, k1) - gam_star(:, k2)).^2);
                        g(k1, k2) = sqrt(d)/(g0 + sqrt(d));
                        g(k2, k1) = g(k1, k2);
                    end
                end
            end
            gam_star_c = gam_star(:, K);
            Gam_star_c = Gam_star(:, :, K);
        end
        % Sample from the urn-model:
        log_pr = zeros(1, t + 1);
        logI = [logI0, logI1, logI2];
        logI = logI - max(logI);
        for k = 1:t
            log_pr(k) = -0.5 * log(det(2 * pi *Gam_star(:, :, k))) - ...
                0.5 * (Y(:, i)' - gam_star(:, k)')/Gam_star(:, :, k) * ...
                (Y(:, i) - gam_star(:, k));
            log_pr(k) = log_pr(k) + log(N(k) + 1) + ...
                log((t + n + 1) * exp(logI(1)) + (t + 1) * exp(logI(2))) - log(2 * t + n + 2);
        end
        log_pr(t + 1) = -0.5 * log(det(2 * pi *Gam_star_c)) - ...
                0.5 * (Y(:, i)' - gam_star_c')/Gam_star_c * ...
                (Y(:, i) - gam_star_c);
        log_pr(t + 1) = log_pr(t + 1) + log_V(t + 1) - log_V(t) + ...
            log((t + n + 2) * exp(logI(2)) + (t + 2) * exp(logI(3))) - log(2 * t + n + 4);
        log_pr = log_pr - max(log_pr);
        pr = exp(log_pr);
        pr = pr/sum(pr);
        tmp = sum((rand(1, 1) >= cumsum(pr))) + 1;
        if tmp == t + 1
            gam(:, i) = gam_star_c;
            Gam(:, :, i) = Gam_star_c;
        else
            gam(:, i) = gam_star(:, tmp);
            Gam(:, :, i) = Gam_star(:, :, tmp);
        end
    end
    % Step 2: Given C, re-sample (theta_1,...,theta_n)
    % Sampling K ~ p(K | t, y_1:n, Gamma_star) approximately from p(K | t)
    gam_star_1 = unique(gam(1, :));
    t = length(gam_star_1);
    n_Zhat = 500;
    gamma_pos_mc = zeros(p, t + m - 1, n_Zhat);
    for k = 1:(t + m - 1)
        if k <= t
            ind_k = ind_n(gam(1, :) == gam_star_1(k));
            Vk = eye(p, p)/(eye(p, p)/Gam_star(:, :, k) * N(k) + 1/tau^2 * eye(p, p));
            mk = Vk * (eye(p, p)/Gam_star(:, :, k) * sum(Y(:, ind_k), 2));
        else
            Vk = tau^2 * eye(p, p);
            mk = zeros(p, 1);
        end
        for j = 1:p
            gamma_pos_mc(j, k, :) = mk(j) + sqrt(Vk(j, j)) * randn(1, n_Zhat);
        end
    end
    Zhat = numerical_Zhat(gamma_pos_mc, g0, t, m);
    log_prob = zeros(1, m);
    for K = t:(t + m - 1)
        log_prob(K - t + 1) = Zhat(K - t + 1) - Z(K) + gammaln(K + 1) - gammaln(K - t + 1) - gammaln(K + n + 1);
    end
    log_prob = log_prob - max(log_prob);
    pr = exp(log_prob);
    pr = pr/sum(pr);
    tmp = sum((rand(1, 1) >= cumsum(pr))) + 1;
    K = t + tmp - 1;
    % Initialize gamma_star and Gamma_star
    gam_star = zeros(p, K);
    Gam_star = zeros(p, p, K);
    % Relabel (c: c in C_script) to (c_1,...,c_t) and gather unique values
    % gamma_k^star, Gamma_k^star, k = 1,...,t
    N = zeros(1, t);
    for k = 1:t
        j = 1;
        while (gam(1, j) ~= gam_star_1(k))
            j = j + 1;
        end
        N(k) = sum(gam(1, :) == gam_star_1(k));
        gam_star(:, k) = gam(:, j);
        Gam_star(:, :, k) = Gam(:, :, j);
    end
    % Given gamma_k_star's, sample Gamma_k_star's
    for k = 1:K
        if k <= t
            a_k = a0 + N(k)/2;
            ind_k = ind_n(gam(1, :) == gam_star_1(k));
            tmp = Y(:, ind_k) - gam_star(:, k) * ones(1, length(ind_k));
            SSE = tmp * tmp';
        else
            a_k = a0;
            SSE = zeros(p, p);
        end
        lambda = zeros(1, p);
        for j = 1:p
            b_k = b0 + 0.5 * SSE(j, j);
            lambda(j) = gamrnd(a_k, 1/b_k);
            while (lambda(j) < usig2^(-1)) || (lambda(j) > lsig2^(-1))
                lambda(j) = gamrnd(a_k, 1/b_k);
            end
        end
        Gam_star(:, :, k) = eye(p, p)/diag(lambda);
    end
    % Given Gamma_k_star's, sample gamma_k_star's using accept-reject
    % sampler
    g = zeros(K, K);
    count = 0;
    while (rand(1) >= min(min(g)))
        count = count + 1;
        for k = 1:K
            if k <= t
                ind_k = ind_n(gam(1, :) == gam_star_1(k));
                Vk = eye(p, p)/(eye(p, p)/Gam_star(:, :, k) * N(k) + 1/tau^2 * eye(p, p));
                mk = Vk * (eye(p, p)/Gam_star(:, :, k) * sum(Y(:, ind_k), 2));
            else
                Vk = tau^2 * eye(p, p);
                mk = zeros(p, 1);
            end
            for j = 1:p
                gam_star(j, k) = mk(j) + sqrt(Vk(j, j)) * randn(1);
            end
        end
        g = ones(K, K);
        for k1 = 1:(K - 1)
            for k2 = (k1 + 1):K
                d = sum((gam_star(:, k1) - gam_star(:, k2)).^2);
                g(k1, k2) = sqrt(d)/(g0 + sqrt(d));
                g(k2, k1) = g(k1, k2);
            end
        end
    end
    reject_time(iter) = count - 1;
    % Assign newly sample theta_k^star's to each observation using previous
    % sampled partition
    for i = 1:n
        k = 1;
        while (gam(1, i) ~= gam_star_1(k))
            k = k + 1;
        end
        gam(:, i) = gam_star(:, k);
        Gam(:, :, i) = Gam_star(:, :, k);
    end
    gamma_mc(:, :, iter) = gam;
    Gamma_mc(:, :, :, iter) = Gam;
    K_mc(iter) = K;
    if (mod(iter, 50) == 0)
        fprintf('Iteration #%d; Time Consumed: %s\n', iter, sprintf('%.0fs;', toc))
    end
end
fprintf('rejection rate in step 4: %s\n', sprintf('%.4f;', sum(reject_time((B + 1):(B + nmc)))/nmc));
% toc
end
