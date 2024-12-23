function [logI0, logI1, logI2] = Numerical_I(gam_star, Z, g0, mu1_mc, mu2_mc)
%% Numerically compute the quantities I0, I1, and I2 (in logarithm scale)
%  I0 = h(gam_star)/Z_t
%  I1 = int h(mu, gam_star)/Z_{t+1}p(mu)dmu
%  I2 = int h(mu1, mu2, gam_star)/Z_{t+1}p(mu1)p(mu2)dmu1dmu2
[~, t] = size(gam_star);
[~, nmc] = size(mu1_mc);
h_mat = ones(t, t);
for k1 = 1:(t - 1)
    for k2 = (k1 + 1):t
        d = sum((gam_star(:, k1) - gam_star(:, k2)).^2);
        h_mat(k1, k2) = sqrt(d)/(g0 + sqrt(d));
        h_mat(k2, k1) = h_mat(k1, k2);
    end
end
logI0 = log(min(min(h_mat))) - Z(t);
h_mat1 = ones(t + 1, t + 1, nmc);
h_mat2 = ones(t + 2, t + 2, nmc);
h_1 = ones(1, nmc);
h_2 = ones(1, nmc);
for iter = 1:nmc
    h_mat1(1:t, 1:t, iter) = h_mat;
    h_mat2(1:t, 1:t, iter) = h_mat;
    for k = 1:t
        d = sum((gam_star(:, k) - mu1_mc(:, iter)).^2);
        h_mat1(k, t + 1, iter) = sqrt(d)/(g0 + sqrt(d));
        h_mat1(t + 1, k, iter) = h_mat1(k, t + 1, iter);
        h_mat2(k, t + 1, iter) = sqrt(d)/(g0 + sqrt(d));
        h_mat2(t + 1, k, iter) = h_mat2(k, t + 1, iter);

        d = sum((gam_star(:, k) - mu2_mc(:, iter)).^2);
        h_mat2(k, t + 2, iter) = sqrt(d)/(g0 + sqrt(d));
        h_mat2(t + 2, k, iter) = h_mat2(k, t + 2, iter);
    end
    d = sum((mu2_mc(:, iter) - mu1_mc(:, iter)).^2);
    h_mat2(t + 1, t + 2, iter) = sqrt(d)/(g0 + sqrt(d));
    h_mat2(t + 2, t + 1, iter) = h_mat2(t + 1, t + 2, iter);
    h_1(iter) = min(min(h_mat1(:, :, iter)));
    h_2(iter) = min(min(h_mat2(:, :, iter)));
end
logI1 = log(mean(h_1)) - Z(t + 1);
logI2 = log(mean(h_2)) - Z(t + 2);
end