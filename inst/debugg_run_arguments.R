rm(list=ls())
source("inst/BART_test_simulation.R")
source("R/other_functions.R")
Rcpp::sourceCpp("src/mvnbart.cpp")
x_train <- data_train |> dplyr::select(dplyr::starts_with("X")) #|> as.matrix()
x_test <- data_test |> dplyr::select(dplyr::starts_with("X")) #|> as.matrix()
c_train <- c <- data_train |> dplyr::pull("C") |> as.matrix()
q_train <- q <- data_train |> dplyr::pull("Q") |> as.matrix()
n_tree = 100
n_mcmc = 2000
n_burn = 500
alpha = 0.95
beta = 2
dif_order = 0
nIknots = 20
df = 3
sigquant = 0.9
kappa = 2
scale_bool = TRUE
# Hyperparam for tau_b and tau_b_0
nu = 2
delta = 1
a_delta = 0.0001
d_delta = 0.0001
df_tau_b = 3
prob_tau_b = 0.9
stump <- FALSE
node_min_size <- 5
fixed_P <- FALSE

