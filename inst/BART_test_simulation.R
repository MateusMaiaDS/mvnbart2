# library(BART)
# library(mvnbart)
devtools::load_all()

# simulate data ####

# true regression functions
f_true_C <- function(X){
  as.numeric(
    (cos(2*X[1]) + 2 * X[2]^2 * X[3])
  )
}

f_true_Q <- function(X){
  as.numeric(
    3 * X[1] * X[4]^3 + X[2]
  )
}

# true covariance matrix for residuals
sigma_c <- 10
sigma_q <- 0.1
rho <- 0.8
Sigma <- matrix(c(sigma_c^2,sigma_c*sigma_q*rho,sigma_c*sigma_q*rho,sigma_q^2), nrow = 2)
Sigma_chol <- t(chol(Sigma))

# sample size
N <- 400

data_train <- data.frame(X1 = rep(NA, N))
data_train$X1 <- runif(N, -1, 1)
data_train$X2 <- runif(N, -1, 1)
data_train$X3 <- runif(N, -1, 1)
data_train$X4 <- runif(N, -1, 1)

data_train$C <- NA
data_train$EC <- NA
data_train$Q <- NA
data_train$EQ <- NA

for (i in 1:N){
  resid <- Sigma_chol %*% rnorm(2)
  data_train$EC[i] <- f_true_C(data_train[i,1:4])
  data_train$C[i] <- (f_true_C(data_train[i,1:4]) + resid[1]) * 1
  data_train$EQ[i] <- f_true_Q(data_train[i,1:4])
  data_train$Q[i] <- (f_true_Q(data_train[i,1:4]) + resid[2]) * 1
}

data_test <- data.frame(X1 = rep(NA, N))
data_test$X1 <- runif(N, -1, 1)
data_test$X2 <- runif(N, -1, 1)
data_test$X3 <- runif(N, -1, 1)
data_test$X4 <- runif(N, -1, 1)

data_test$C <- NA
data_test$EC <- NA
data_test$Q <- NA
data_test$EQ <- NA

for (i in 1:N){
  resid <- Sigma_chol %*% rnorm(2)
  data_test$EC[i] <- f_true_C(data_test[i,1:4])
  data_test$C[i] <- (f_true_C(data_test[i,1:4]) + resid[1]) * 1
  data_test$EQ[i] <- f_true_Q(data_test[i,1:4])
  data_test$Q[i] <- (f_true_Q(data_test[i,1:4]) + resid[2]) * 1
}


mvnBart_wrapper <- function(x_train, x_test, c_train, q_train, rescale){

  if (rescale == TRUE){

    c_rescaled <- (c_train - min(c_train))/(max(c_train) - min(c_train)) - 0.5
    q_rescaled <- (q_train - min(q_train))/(max(q_train) - min(q_train)) - 0.5

    mvnBart_fit <- mvnbart::mvnbart(x_train = x_train,
                                    c_train = c_rescaled, q_train = q_rescaled,
                                    x_test = x_test, scale_bool = FALSE,
                                    n_tree = 200)

    c_hat_test <- (mvnBart_fit$c_hat_test + 0.5) * (max(c_train) - min(c_train)) + min(c_train)
    q_hat_test <- (mvnBart_fit$q_hat_test + 0.5) * (max(q_train) - min(q_train)) + min(q_train)

    # something is wrong with the following backtransformations
    sigma_c <- (max(c_train) - min(c_train)) * mvnBart_fit$tau_c_post^(-1/2)
    sigma_q <- (max(q_train) - min(q_train)) * mvnBart_fit$tau_c_post^(-1/2)
    rho = mvnBart_fit$rho_post

  }

  else{
    mvnBart_fit <- mvnbart::mvnbart(x_train = x_train,
                                    c_train = c_train, q_train = q_train,
                                    x_test = x_test, scale_bool = FALSE,
                                    n_tree = 200)

    # something is wrong with the following backtransformations
    sigma_c <-  mvnBart_fit$tau_c_post^(-1/2)
    sigma_q <-  mvnBart_fit$tau_c_post^(-1/2)
    rho = mvnBart_fit$rho_post
  }

  return(
    list(
      c_hat_test = mvnBart_fit$c_hat_test,
      q_hat_test = mvnBart_fit$q_hat_test,
      sigma_c = mvnBart_fit$tau_c_post^(-1/2),
      sigma_q = mvnBart_fit$tau_q_post^(-1/2),
      rho = mvnBart_fit$rho_post
    )
  )
}


# Running the model
#
fit <- mvnBart_wrapper(data_train[,1:4], data_test[,1:4], data_train$C, data_train$Q, rescale = F)

1/N * sum((apply(fit$c_hat_test, 1, mean) - data_test$C)) # out of sample bias - c
1/N * sum((apply(fit$q_hat_test, 1, mean) - data_test$Q)) # out of sample bias - q
sqrt(1/N * sum((apply(fit$c_hat_test, 1, mean) - data_test$C)^2)) # out of sample RMSE - c
sqrt(1/N * sum((apply(fit$q_hat_test, 1, mean) - data_test$Q)^2)) # out of sample RMSE - q

mean(fit$sigma_c)
mean(fit$sigma_q)
mean(fit$rho)

plot(fit$sigma_c, type = "l")
plot(fit$sigma_q, type = "l")
plot(fit$rho, type = "l")

# fit seperate models ####
univariate.fit.C <- BART::wbart(x.train = as.matrix(data_train[,1:4]), y.train = as.matrix(data_train$C),
                          x.test = as.matrix(data_test[,1:4]), ntree = 200,
                          ndpost=1500, nskip=500)

univariate.fit.Q <- BART::wbart(x.train = as.matrix(data_train[,1:4]), y.train = as.matrix(data_train$Q),
                          x.test = as.matrix(data_test[,1:4]), ntree = 200,
                          ndpost=1500, nskip=500)


1/N * sum(univariate.fit.C$yhat.test.mean - data_test$C) # out of sample bias - c
1/N * sum(univariate.fit.Q$yhat.test.mean - data_test$Q) # out of sample bias - q
sqrt(1/N * sum((univariate.fit.C$yhat.test.mean - data_test$C)^2)) # out of sample RMSE - c
sqrt(1/N * sum((univariate.fit.Q$yhat.test.mean - data_test$Q)^2)) # out of sample RMSE - q

