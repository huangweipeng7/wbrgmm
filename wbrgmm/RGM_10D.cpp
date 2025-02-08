#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix association(IntegerVector Z){
  int n = Z.size();
  NumericMatrix S(n, n);
  for (int i = 0; i < n; i ++){
    for (int j = 0; j < n; j ++){
      if (Z(i) == Z(j)){
        S(i, j) = 1;
      } else {
        S(i, j) = 0;
      }
    }
  }
  return(S);
}

// [[Rcpp::export]]
NumericMatrix coclustering(NumericVector gamma, double tol){
  int n = gamma.size();
  NumericMatrix S(n, n);
  for (int i = 0; i < n; i ++){
    for (int j = 0; j < n; j ++){
      if ((gamma(i) - gamma(j) < tol) && (gamma(j) - gamma(i) < tol)){
        S(i, j) = 1;
      } else {
        S(i, j) = 0;
      }
    }
  }
  return(S);
}
