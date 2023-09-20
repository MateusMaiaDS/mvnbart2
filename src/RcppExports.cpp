// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// gamma_pdf
double gamma_pdf(double x, double a, double b);
RcppExport SEXP _mvnbart_gamma_pdf(SEXP xSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(gamma_pdf(x, a, b));
    return rcpp_result_gen;
END_RCPP
}
// r_gamma_pdf
double r_gamma_pdf(double x, double a, double b);
RcppExport SEXP _mvnbart_r_gamma_pdf(SEXP xSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(r_gamma_pdf(x, a, b));
    return rcpp_result_gen;
END_RCPP
}
// print_mat_subset
void print_mat_subset(arma::mat X);
RcppExport SEXP _mvnbart_print_mat_subset(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    print_mat_subset(X);
    return R_NilValue;
END_RCPP
}
// log_dmvn
double log_dmvn(arma::vec& x, arma::mat& Sigma);
RcppExport SEXP _mvnbart_log_dmvn(SEXP xSEXP, SEXP SigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type Sigma(SigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_dmvn(x, Sigma));
    return rcpp_result_gen;
END_RCPP
}
// cppbart
Rcpp::List cppbart(arma::mat x_train, arma::vec c_train, arma::vec q_train, arma::mat x_test, int n_tree, int node_min_size, double alpha, double beta, int n_mcmc, int n_burn, arma::mat P, double mu_c, double mu_q, double tau_mu, double tau_lambda, double df_wish, arma::mat s_0_wish, double A_c, double A_q);
RcppExport SEXP _mvnbart_cppbart(SEXP x_trainSEXP, SEXP c_trainSEXP, SEXP q_trainSEXP, SEXP x_testSEXP, SEXP n_treeSEXP, SEXP node_min_sizeSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP n_mcmcSEXP, SEXP n_burnSEXP, SEXP PSEXP, SEXP mu_cSEXP, SEXP mu_qSEXP, SEXP tau_muSEXP, SEXP tau_lambdaSEXP, SEXP df_wishSEXP, SEXP s_0_wishSEXP, SEXP A_cSEXP, SEXP A_qSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x_train(x_trainSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c_train(c_trainSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type q_train(q_trainSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x_test(x_testSEXP);
    Rcpp::traits::input_parameter< int >::type n_tree(n_treeSEXP);
    Rcpp::traits::input_parameter< int >::type node_min_size(node_min_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< int >::type n_mcmc(n_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type n_burn(n_burnSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type P(PSEXP);
    Rcpp::traits::input_parameter< double >::type mu_c(mu_cSEXP);
    Rcpp::traits::input_parameter< double >::type mu_q(mu_qSEXP);
    Rcpp::traits::input_parameter< double >::type tau_mu(tau_muSEXP);
    Rcpp::traits::input_parameter< double >::type tau_lambda(tau_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type df_wish(df_wishSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type s_0_wish(s_0_wishSEXP);
    Rcpp::traits::input_parameter< double >::type A_c(A_cSEXP);
    Rcpp::traits::input_parameter< double >::type A_q(A_qSEXP);
    rcpp_result_gen = Rcpp::wrap(cppbart(x_train, c_train, q_train, x_test, n_tree, node_min_size, alpha, beta, n_mcmc, n_burn, P, mu_c, mu_q, tau_mu, tau_lambda, df_wish, s_0_wish, A_c, A_q));
    return rcpp_result_gen;
END_RCPP
}
// mat_init
arma::mat mat_init(int n);
RcppExport SEXP _mvnbart_mat_init(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(mat_init(n));
    return rcpp_result_gen;
END_RCPP
}
// vec_init
arma::vec vec_init(int n);
RcppExport SEXP _mvnbart_vec_init(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(vec_init(n));
    return rcpp_result_gen;
END_RCPP
}
// std_inv
arma::mat std_inv(arma::mat A, arma::vec diag);
RcppExport SEXP _mvnbart_std_inv(SEXP ASEXP, SEXP diagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type diag(diagSEXP);
    rcpp_result_gen = Rcpp::wrap(std_inv(A, diag));
    return rcpp_result_gen;
END_RCPP
}
// std_pinv
arma::mat std_pinv(arma::mat A, arma::vec diag);
RcppExport SEXP _mvnbart_std_pinv(SEXP ASEXP, SEXP diagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type diag(diagSEXP);
    rcpp_result_gen = Rcpp::wrap(std_pinv(A, diag));
    return rcpp_result_gen;
END_RCPP
}
// faster_simple_std_inv
arma::mat faster_simple_std_inv(arma::mat A, arma::vec diag);
RcppExport SEXP _mvnbart_faster_simple_std_inv(SEXP ASEXP, SEXP diagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type diag(diagSEXP);
    rcpp_result_gen = Rcpp::wrap(faster_simple_std_inv(A, diag));
    return rcpp_result_gen;
END_RCPP
}
// log_test
double log_test(double a);
RcppExport SEXP _mvnbart_log_test(SEXP aSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    rcpp_result_gen = Rcpp::wrap(log_test(a));
    return rcpp_result_gen;
END_RCPP
}
// faster_std_inv
arma::mat faster_std_inv(arma::mat A, arma::vec diag);
RcppExport SEXP _mvnbart_faster_std_inv(SEXP ASEXP, SEXP diagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type diag(diagSEXP);
    rcpp_result_gen = Rcpp::wrap(faster_std_inv(A, diag));
    return rcpp_result_gen;
END_RCPP
}
// rMVN2
arma::vec rMVN2(const arma::vec& b, const arma::mat& Q);
RcppExport SEXP _mvnbart_rMVN2(SEXP bSEXP, SEXP QSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type b(bSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q(QSEXP);
    rcpp_result_gen = Rcpp::wrap(rMVN2(b, Q));
    return rcpp_result_gen;
END_RCPP
}
// rMVNslow
arma::vec rMVNslow(const arma::vec& b, const arma::mat& Q);
RcppExport SEXP _mvnbart_rMVNslow(SEXP bSEXP, SEXP QSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type b(bSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Q(QSEXP);
    rcpp_result_gen = Rcpp::wrap(rMVNslow(b, Q));
    return rcpp_result_gen;
END_RCPP
}
// matrix_mat
arma::mat matrix_mat(arma::cube array);
RcppExport SEXP _mvnbart_matrix_mat(SEXP arraySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type array(arraySEXP);
    rcpp_result_gen = Rcpp::wrap(matrix_mat(array));
    return rcpp_result_gen;
END_RCPP
}
// cppWishart
arma::mat cppWishart(double df, arma::mat Sigma);
RcppExport SEXP _mvnbart_cppWishart(SEXP dfSEXP, SEXP SigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type df(dfSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Sigma(SigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(cppWishart(df, Sigma));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mvnbart_gamma_pdf", (DL_FUNC) &_mvnbart_gamma_pdf, 3},
    {"_mvnbart_r_gamma_pdf", (DL_FUNC) &_mvnbart_r_gamma_pdf, 3},
    {"_mvnbart_print_mat_subset", (DL_FUNC) &_mvnbart_print_mat_subset, 1},
    {"_mvnbart_log_dmvn", (DL_FUNC) &_mvnbart_log_dmvn, 2},
    {"_mvnbart_cppbart", (DL_FUNC) &_mvnbart_cppbart, 19},
    {"_mvnbart_mat_init", (DL_FUNC) &_mvnbart_mat_init, 1},
    {"_mvnbart_vec_init", (DL_FUNC) &_mvnbart_vec_init, 1},
    {"_mvnbart_std_inv", (DL_FUNC) &_mvnbart_std_inv, 2},
    {"_mvnbart_std_pinv", (DL_FUNC) &_mvnbart_std_pinv, 2},
    {"_mvnbart_faster_simple_std_inv", (DL_FUNC) &_mvnbart_faster_simple_std_inv, 2},
    {"_mvnbart_log_test", (DL_FUNC) &_mvnbart_log_test, 1},
    {"_mvnbart_faster_std_inv", (DL_FUNC) &_mvnbart_faster_std_inv, 2},
    {"_mvnbart_rMVN2", (DL_FUNC) &_mvnbart_rMVN2, 2},
    {"_mvnbart_rMVNslow", (DL_FUNC) &_mvnbart_rMVNslow, 2},
    {"_mvnbart_matrix_mat", (DL_FUNC) &_mvnbart_matrix_mat, 1},
    {"_mvnbart_cppWishart", (DL_FUNC) &_mvnbart_cppWishart, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_mvnbart(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
