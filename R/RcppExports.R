# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

gamma_pdf <- function(x, a, b) {
    .Call('_mvnbart_gamma_pdf', PACKAGE = 'mvnbart', x, a, b)
}

r_gamma_pdf <- function(x, a, b) {
    .Call('_mvnbart_r_gamma_pdf', PACKAGE = 'mvnbart', x, a, b)
}

print_mat_subset <- function(X) {
    invisible(.Call('_mvnbart_print_mat_subset', PACKAGE = 'mvnbart', X))
}

log_dmvn <- function(x, Sigma) {
    .Call('_mvnbart_log_dmvn', PACKAGE = 'mvnbart', x, Sigma)
}

cppbart <- function(x_train, c_train, q_train, x_test, n_tree, node_min_size, alpha, beta, n_mcmc, n_burn, P, mu_c, mu_q, tau_mu, tau_lambda, df_wish, s_0_wish, A_c, A_q) {
    .Call('_mvnbart_cppbart', PACKAGE = 'mvnbart', x_train, c_train, q_train, x_test, n_tree, node_min_size, alpha, beta, n_mcmc, n_burn, P, mu_c, mu_q, tau_mu, tau_lambda, df_wish, s_0_wish, A_c, A_q)
}

mat_init <- function(n) {
    .Call('_mvnbart_mat_init', PACKAGE = 'mvnbart', n)
}

vec_init <- function(n) {
    .Call('_mvnbart_vec_init', PACKAGE = 'mvnbart', n)
}

std_inv <- function(A, diag) {
    .Call('_mvnbart_std_inv', PACKAGE = 'mvnbart', A, diag)
}

std_pinv <- function(A, diag) {
    .Call('_mvnbart_std_pinv', PACKAGE = 'mvnbart', A, diag)
}

faster_simple_std_inv <- function(A, diag) {
    .Call('_mvnbart_faster_simple_std_inv', PACKAGE = 'mvnbart', A, diag)
}

log_test <- function(a) {
    .Call('_mvnbart_log_test', PACKAGE = 'mvnbart', a)
}

faster_std_inv <- function(A, diag) {
    .Call('_mvnbart_faster_std_inv', PACKAGE = 'mvnbart', A, diag)
}

rMVN2 <- function(b, Q) {
    .Call('_mvnbart_rMVN2', PACKAGE = 'mvnbart', b, Q)
}

rMVNslow <- function(b, Q) {
    .Call('_mvnbart_rMVNslow', PACKAGE = 'mvnbart', b, Q)
}

matrix_mat <- function(array) {
    .Call('_mvnbart_matrix_mat', PACKAGE = 'mvnbart', array)
}

cppWishart <- function(df, Sigma) {
    .Call('_mvnbart_cppWishart', PACKAGE = 'mvnbart', df, Sigma)
}

