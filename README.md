
<!-- README.md is generated from README.Rmd. Please edit that file -->

# mvnbart

## Installation

You can install the development version of mvnbart from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("MateusMaiaDS/mvnbart")
```


## First experiments

``` r
# simulate data ####

# true regression functions
f_true_C <- function(X){
  as.numeric(
    2 * sin(X[1]*X[2]) + X[4]^2
    )
}

f_true_Q <- function(X){
  as.numeric(
    (X[1] + X[3])^2 + cos(X[2]) * X[3]
    )
}

# true covariance matrix for residuals
Sigma <- matrix(c(1,1*1*0.8,1*1*0.8,1), nrow = 2)
# Sigma <- diag(nrow = 2)
Sigma_chol <- t(chol(Sigma))

# sample size
N <- 300

data <- data.frame(X1 = rep(NA, N))
data$X1 <- runif(N, -1, 1)
data$X2 <- runif(N, -1, 1)
data$X3 <- runif(N, -1, 1)
data$X4 <- runif(N, -1, 1)

data$C <- NA
data$EC <- NA
data$Q <- NA
data$EQ <- NA

for (i in 1:N){
  resid <- Sigma_chol %*% rnorm(2)
  data$EC[i] <- f_true_C(data[i,1:4])
  data$C[i] <- f_true_C(data[i,1:4]) + resid[1]
  data$EQ[i] <- f_true_Q(data[i,1:4])
  data$Q[i] <- f_true_Q(data[i,1:4]) + resid[2]
}


# fit bivariate model ####
# ...
x_train <- data |> dplyr::select(dplyr::starts_with("X")) #|> as.matrix()
x_test <- x_train
c <- data |> dplyr::pull("C")
q <- data |> dplyr::pull("Q")
n_mcmc = 2000
n_burn = 500

mvnBart_mod <- mvnbart::mvnbart(x_train = x_train,
                                c_train = c,q_train = q,
                                x_test = x_test,scale_bool = FALSE,
                                n_tree = 200)
#

# Estimating the correlation
mvnBart_mod$rho_post %>% plot(type = "l")
```
