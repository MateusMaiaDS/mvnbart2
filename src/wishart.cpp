#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::mat cppWishart(double df, arma::mat Sigma){
     return arma::wishrnd(Sigma,df);
}
