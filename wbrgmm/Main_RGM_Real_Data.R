##########################################################################################################
# Loading required packages
##########################################################################################################
library(mvtnorm)
library(gplots)
library(Rcpp)
library(RColorBrewer)
library(coda)
library(gtools)
library(plotly)
library(DPpackage)
library(R.matlab)
library(BNPdensity)
# Loading Old faithful geyser eruption data
data("faithful")
# Transformed the univariate data to the 2-Dimensional space
faithful_2D = matrix(NA, nrow = 271, ncol = 2)
faithful_2D[, 1] = faithful$eruptions[1:271]
faithful_2D[, 2] = faithful$eruptions[2:272]
# Save the data
write.csv(faithful_2D, "faithful_data.csv")
# my_palette = colorRampPalette(c("white", "blue", "purple"))(n = 299)
##########################################################################################################
# Load the MCMC results produced by the blocked-collapsed Gibbs sampler.
# The MCMC results are stored in the "faithful_RGM_results.mat" file.
# The blocked-collapsed Gibbs sampler is implemented in Matlab.
# The m-function for the algorithm is named "blocked_collapsed_Gibbs.m"
# The "faithful_RGM_results.mat" data file is reproducable by running the m-file "Real_Data_Analysis.m".
#########################################################################################################
old_faithful = readMat("faithful_RGM_result_2Revision.mat")
attach(old_faithful)
n = dim(Y.faithful)[1]
p = dim(Y.faithful)[2]
B = c(B)
nmc = c(nmc)
##########################################################################################################
# For comparison consider the DPP mixtures with RJ-MCMC
# The following RJ-MCMC code is available at http://www.ams.jhu.edu/~yxu70/DPP.zip
##########################################################################################################
library(MASS)
library(gtools)
library(plyr)
library(MCMCpack) #for riwish
library(mvtnorm)
library(mc2d)
source("multi_update_pos.R")
D = dim(Y.faithful)[2] #dimension
Num = dim(Y.faithful)[1]
Eigenvector = diag(1, D)
##hyperparameters###
hyper = NULL
hyper$r = 4
hyper$l = 4
hyper$delta = 1
hyper$sigmapro = 0.2
hyper$theta = 4
#start_time = proc.time()
Niter = 2000
burn.in = 1000
lag = 1
mcmc = NULL
mcmc$z = matrix(NA, Niter, Num)
mcmc$w = matrix(NA, Niter, Num)
mcmc$mu = array(NA, c(Niter, Num, D))
mcmc$K = rep(NA, Niter)
mcmc$lambda = array(0, c(Niter, Num, D))
##Initial values using K-means
set.seed(2)
mcmc$K[1] = 10
cl = kmeans(Y.faithful, mcmc$K[1])
mcmc$z[1,] = cl$cluster
mcmc$w[1,1:mcmc$K[1]] = cl$size/sum(cl$size)
mcmc$mu[1,1:mcmc$K[1],] = cl$centers
mcmc$lambda[1,1:mcmc$K[1],] = matrix(2, mcmc$K[1], D)
y = Y.faithful
start_time = proc.time()
for (iter in 2:Niter){
  tmp = update_zK(mcmc$K[iter-1], mcmc$mu[iter-1,1:mcmc$K[iter-1],], mcmc$lambda[iter-1,1:mcmc$K[iter-1],], mcmc$w[iter-1,1:mcmc$K[iter-1]])
  mcmc$K[iter] = tmp$K
  mcmc$z[iter,] = tmp$z
  mcmc$mu[iter,1:mcmc$K[iter],] = tmp$mu
  mcmc$w[iter,1:mcmc$K[iter]] = tmp$w
  mcmc$lambda[iter,1:mcmc$K[iter],] = tmp$lambda
  mcmc$w[iter,1:mcmc$K[iter]] = update_w(mcmc$K[iter], mcmc$z[iter,])
  mcmc$lambda[iter,1:mcmc$K[iter],] = update_lambda(mcmc$mu[iter,1:mcmc$K[iter],],mcmc$K[iter],  mcmc$z[iter,])
  mcmc$mu[iter,1:mcmc$K[iter],] = update_mu(mcmc$K[iter], mcmc$mu[iter,1:mcmc$K[iter],], mcmc$z[iter,],mcmc$lambda[iter,1:mcmc$K[iter],])
  tmp2 = update_rj(mcmc$K[iter],mcmc$lambda[iter,1:mcmc$K[iter],], mcmc$mu[iter,1:mcmc$K[iter],], mcmc$w[iter,1:mcmc$K[iter]],mcmc$z[iter,])
  mcmc$K[iter] = tmp2$K
  mcmc$w[iter,1:mcmc$K[iter]] = tmp2$w
  mcmc$mu[iter,1:mcmc$K[iter],] = tmp2$mu
  mcmc$lambda[iter,1:mcmc$K[iter],] = tmp2$lambda
  mcmc$z[iter,] = tmp2$z
  if (floor(iter/100) == iter/100){
    print(paste("Iteration # ", iter, sep = ""))
    print(paste("Number of Components K = ", mcmc$K[iter], sep = ""))
  }
}
proc.time()-start_time
##########################################################################################################
# The RJ-MCMC sampler is time consuming and the computation time on the following machine is 274.208s:
# MacBook Pro (Retina, 15-inch, Mid 2015)
# Operation System: MacOS Sierra 10.12.1
# Processor: 2.8 GHz Intel Core i7
# Memory: 16 GB 1600 MHz DDR3
# Graphics Card: Intel Iris Pro 1536 MB
##########################################################################################################
# Finding a cluster configuration with minimum MSE
cluster_eval <- function(mcmc){
  Hmatrix = matrix(0, Num, Num)
  tmp = array(0, c(Num, Num, 500))
  for (i in 1:500)
  {
    iter = burn.in + i*lag
    tmp1 = mcmc$z[iter,]%*%t(mcmc$z[iter,])
    id = which(tmp1==1 | tmp1==4 | tmp1==9)
    tmp1[id] = 1
    tmp1[-id] = 0
    tmp[,,i] = tmp1
    Hmatrix = Hmatrix + tmp[,,i]
  }
  Hmatrix_ave = Hmatrix/500
  diff = rep(0, 500)
  for (i in 1:500)
  {
    diff[i] = sum((Hmatrix_ave-tmp[,,i])^2)
  }
  idmin = which(diff==min(diff))[1]
  cluster = mcmc$z[burn.in + idmin*lag,]
  # print(burn.in + idmin*lag)
  return(cluster)
}
cluster_DPP = cluster_eval(mcmc)
##########################################################################################################
# Finding a cluster configuration with minimum mean-squared error
##########################################################################################################
Rcpp::sourceCpp('RGM_10D.cpp')
Rcpp::sourceCpp('gmm_density.cpp')
p_pos = matrix(0, nrow = n, ncol = n)
for (iter in (B + 1): (B + nmc)){
  p_pos = p_pos + coclustering(gamma.mc[1, , iter], tol = 1e-12)
}
p_pos = p_pos/nmc
p_error_pos = rep(0, nmc)
for (iter in (B + 1): (B + nmc)){
  i = iter - B
  p_error_pos[i] = mean((coclustering(gamma.mc[1, , iter], tol = 1e-12) - p_pos)^2)
}
iter_hat_RGM = which.min(p_error_pos) + B
# Misclassification error
K = K.mc[iter_hat_RGM]
cluster_RGM = rep(NA, nrow = n)
gamma_star_1 = sort(unique(gamma.mc[1, , iter_hat_RGM]))
for (k in 1:K){
  cluster_k_ind = which(gamma.mc[1, , iter_hat_RGM] == gamma_star_1[k])
  cluster_RGM[cluster_k_ind] = k
}
##########################################################################################################
# Convergence diagnostics
##########################################################################################################
ind_i = c(9, 11)
pdf("trace_plot_acf_mu_oldfaithful.pdf", width = 12, height = 8)
par(mfrow = c(2, 2))
for (i in ind_i){
  plot(gamma.mc[1, i, ], type = "l", lwd = 3, main = "Trace Plot", xlab = "Iteration", ylab = "gamma")
  grid(nx = 6, ny = 6, col = "grey80", lty = "dashed", lwd = 3)
  plot(acf(gamma.mc[2, i, ], plot = FALSE), main = "Autocorrelation", ylab = "Acf")
}
dev.off()
##########################################################################################################
# Compute posterior predictive density as an estimate
##########################################################################################################
x = matrix(NA, nrow = p, ncol = 100)
for (j in 1:p){
  x[j, ] = seq(0, 6, length = ncol(x))
}
p_pred = p_pred_DPP = matrix(0, nrow = ncol(x), ncol = ncol(x))
##########################################################################################################
# Computing bivariate posterior predictive density is very time consuming !!!
# The computational time for this is roughly 350s on the following machine:
# MacBook Pro (Retina, 15-inch, Mid 2015)
# Operation System: MacOS Sierra 10.12.1
# Processor: 2.8 GHz Intel Core i7
# Memory: 16 GB 1600 MHz DDR3
# Graphics Card: Intel Iris Pro 1536 MB
##########################################################################################################
ptm = proc.time()
for (iter in (B + 1): (B + nmc)){
  gam = gamma.mc[, , iter]
  Gam = apply(Gamma.mc[, , , iter], 3, diag)
  # gam_DPP = t(mcmc$mu[iter, mcmc$z[iter, ], ])
  # Gam_DPP = t(mcmc$lambda[iter, mcmc$z[iter, ], ])
  p_pred = p_pred + p_pred_density(x[1, ], x[2, ], gam, Gam)
  # p_pred_DPP = p_pred_DPP + p_pred_density(x[1, ], x[2, ], gam_DPP, Gam_DPP)
  if (floor(iter/100) == (iter/100)){
    print(paste("Iteration # ", iter, sep = ""))
    ptm = proc.time() - ptm
    print(paste("Time consumed: ", as.integer(ptm[3]), "s", sep = ""))
    ptm = proc.time()
  }
}
##########################################################################################################
# Implementing DP mixtures using the DPpackage for Comparison
##########################################################################################################
# For comparison we use DP mixture for bivariate density estimation as an alternative. 
prior = list(alpha = 1, nu1 = 4, s1 = diag(rep(1, p)), m1 = rep(0, p), 
             psiinv1 = diag(rep(1, p)), tau1 = 1, tau2 = 1)
mcmc_DP = list(nburn = 1000, nsave = 1000, nskip = 1, ndisplay = 100)
DPM_fit = DPdensity(faithful_2D, prior = prior, mcmc = mcmc_DP, state = NULL, status = TRUE, na.action = na.omit)
theta_save = DPM_fit$save.state$thetasave
DPM_cluster = DPM_fit$state$ss
##########################################################################################################
# Density estimate comparison using contour plots and histogram
##########################################################################################################
sample_ind = (B + 1):(B + nmc)
pdf("density_estimate_compare_faithful.pdf", width = 10, height = 12)
Y = t(Y.faithful)
par(mfrow = c(3, 2))
# Plot results for RGM
contour(x[1, ], x[2, ], p_pred/nmc, main = "(a) Predictive Density under RGM", col = "grey", 
        levels = seq(min(p_pred/nmc), max(p_pred/nmc), length = 200), lwd = 1, cex.main = 2, cex.lab = 1.5,
        drawlabels = FALSE, xlim = c(1, 6), ylim = c(0, 7), xlab = "y1", ylab = "y2")
grid(nx = 6, ny = 6, col = "grey80", lty = "dashed", lwd = 3)
points(Y[1, which(cluster_RGM == 2)], Y[2, which(cluster_RGM == 2)], col = 2, type = "p", lwd = 3)
points(Y[1, which(cluster_RGM == 3)], Y[2, which(cluster_RGM == 3)], col = 3, type = "p", lwd = 3)
points(Y[1, which(cluster_RGM == 4)], Y[2, which(cluster_RGM == 4)], col = 4, type = "p", lwd = 3)
points(Y[1, which(cluster_RGM == 1)], Y[2, which(cluster_RGM == 1)], col = 5, type = "p", lwd = 3)
legend("topright", legend = c("Component 1", "Component 2", "Component 3", "Component 4"), 
       col = c(2, 3, 4, 5), lwd = c(3, 3, 3, 3), cex = 1.5, bty = "n")
hist(c(K.mc)[sample_ind], breaks = min(c(K.mc)[sample_ind] - 1.5): max(c(K.mc)[sample_ind] + 1.5),
     xlab = "# Components", main = "(b) Histogram for K using RGM", 
     col = "violetred", border = TRUE, cex.main = 2, cex.axis = 1.5, cex.lab = 1.5, probability = TRUE)
# Plot Results for DPP
contour(x[1, ], x[2, ], p_pred_DPP/nmc, col = "grey", main = "(c) Predictive Density under DPP Mixtures", cex.main = 2,
        levels = seq(min(p_pred/nmc), max(p_pred/nmc), length = 200), lty = "solid", lwd = 1, drawlabels = FALSE,
        xlim = c(1, 6), ylim = c(0, 7), xlab = "y1", ylab = "y2")
grid(nx = 6, ny = 6, col = "grey80", lwd = 3)
points(Y[1, which(cluster_DPP == 1)], Y[2, which(cluster_DPP == 1)], col = 2, type = "p", lwd = 3)
points(Y[1, which(cluster_DPP == 2)], Y[2, which(cluster_DPP == 2)], col = 3, type = "p", lwd = 3)
points(Y[1, which(cluster_DPP == 3)], Y[2, which(cluster_DPP == 3)], col = 4, type = "p", lwd = 3)
legend("topright", legend = c("Component 1", "Component 2", "Component 3"), 
       col = c(2, 3, 4), lwd = c(3, 3, 3), cex = 1.5, bty = "n")
hist(mcmc$K[sample_ind], breaks = min(mcmc$K[sample_ind] - 1.5): max(mcmc$K[sample_ind] + 1.5),
     xlab = "# Components", main = "(d) Histogram for K under DPP Mixtures", 
     col = "wheat2", border = TRUE, cex.main = 2, cex.axis = 1.5, cex.lab = 1.5, probability = TRUE)
# Plot Results for DPM
contour(DPM_fit$x1, DPM_fit$x2, DPM_fit$dens, main = "(e) Predictive Density under DP Mixtures", col = "grey", 
        levels = seq(min(p_pred/nmc), max(p_pred/nmc), length = 200), lwd = 1, cex.main = 2, cex.lab = 1.5,
        drawlabels = FALSE, xlim = c(1, 6), ylim = c(0, 7), xlab = "y1", ylab = "y2")
grid(nx = 6, ny = 6, col = "grey80", lty = "dashed", lwd = 3)
points(Y[1, which(DPM_cluster == 1)], Y[2, which(DPM_cluster == 1)], col = 4, type = "p", lwd = 3)
points(Y[1, which(DPM_cluster == 2)], Y[2, which(DPM_cluster == 2)], col = 2, type = "p", lwd = 3)
points(Y[1, which(DPM_cluster == 3)], Y[2, which(DPM_cluster == 3)], col = 5, type = "p", lwd = 3)
points(Y[1, which(DPM_cluster == 4)], Y[2, which(DPM_cluster == 4)], col = 3, type = "p", lwd = 3)
legend("topright", legend = c("Component 1", "Component 2", "Component 3", "Component 4"), 
       col = c(2, 3, 4, 5), lwd = c(3, 3, 3, 3), cex = 1.5, bty = "n")
hist(DPM_fit$save.state$thetasave[, "ncluster"], 
     breaks = (min(DPM_fit$save.state$thetasave[, "ncluster"]) - 1.5): 
       max(DPM_fit$save.state$thetasave[, "ncluster"] + 1.5),
     xlab = "# Non-empty Components", main = "(f) Histogram for K under DP Mixtures", 
     col = "coral", border = TRUE, cex.main = 2, cex.axis = 1.5, cex.lab = 1.5, probability = TRUE)
dev.off()
##########################################################################################################
# Compute Conditional Predictive Ordinate(CPO)
##########################################################################################################
LPML_RGM = LPML_DPP = matrix(NA, nrow = n, ncol = nmc)
ptm = proc.time()
for (i in 1:n){
  if (floor(i/100) == i/100){
    print(paste("Observation i = ", i, sep = ""))
    print(proc.time() - ptm)
    ptm = proc.time()
  }
  for (iter in 1:nmc){
    LPML_DPP[i, iter] = dmvnorm(Y[, i], mean = mcmc$mu[B + iter, mcmc$z[B + iter, i], ], 
                                  sigma = diag(mcmc$lambda[B + iter, mcmc$z[B + iter, i], ]))
    LPML_RGM[i, iter] = dmvnorm(Y[, i], mean = gamma.mc[, i, B + iter], sigma = Gamma.mc[, , i, B + iter])
  }
}
LPML_RGM = 1/apply(LPML_RGM, 1, mean)
LPML_DPP = 1/apply(LPML_DPP, 1, mean)
print(-sum(log(LPML_RGM))) 
# -242.7644
print(-sum(log(LPML_DPP)))
# -315.1032
print(sum(log(DPM_fit$cpo)))
# -512.6564
save.image("Old_Faithful_Data_Results.RData")
