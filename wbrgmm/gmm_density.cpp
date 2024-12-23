#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::export]]
NumericMatrix gmm_density(NumericVector y1, NumericVector y2){
  int n = y1.size();
  NumericMatrix p_density(n, n);
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      p_density(i, j) = 0.4 * R::dnorm4(y1(i), 0, sqrt(2), 0) * R::dnorm4(y2(j), 0, 1, 0) + 
        0.3 * R::dnorm4(y1(i), 6, sqrt(3), 0) * R::dnorm4(y2(j), 6, sqrt(3), 0) + 
        0.3 * R::dnorm4(y1(i), -6, sqrt(2), 0) * R::dnorm4(y2(j), -6, sqrt(2), 0);
    }
  }
  return(p_density);
}

// [[Rcpp::export]]
NumericMatrix p_pred_density(NumericVector y1, NumericVector y2, NumericMatrix gam, NumericMatrix Gam){
  int m = y1.size();
  int n = gam(0, _).size();
  NumericMatrix pred_density(m, m);
  for (int i = 0; i < m; i ++){
    for (int j = 0; j < m; j ++){
      pred_density(i, j) = 0;
      for (int k = 0; k < n; k ++){
        double tmp1 = R::dnorm4(y1(i), gam(0, k), sqrt(Gam(0, k)), 0);
        double tmp2 = R::dnorm4(y2(j), gam(1, k), sqrt(Gam(1, k)), 0);
        pred_density(i, j) = pred_density(i, j) + tmp1 * tmp2;
      }
      pred_density(i, j) = pred_density(i, j)/n;
    }
  }
  return(pred_density);
}