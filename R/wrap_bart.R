## Bart
#' @useDynLib mvnbart
#' @importFrom Rcpp sourceCpp
#'


# Getting the BART wrapped function
#' @export
mvnbart <- function(x_train,
                  c_train,
                  q_train,
                  x_test,
                  n_tree = 2,
                  node_min_size = 5,
                  n_mcmc = 2000,
                  n_burn = 500,
                  alpha = 0.95,
                  beta = 2,
                  df = 3,
                  sigquant = 0.9,
                  kappa = 2,
                  scale_bool = TRUE,
                  stump = FALSE,
                  fixed_P = NULL
                  ) {


     # Verifying if x_train and x_test are matrices
     if(!is.data.frame(x_train) || !is.data.frame(x_test)){
          stop("Insert valid data.frame for both data and xnew.")
     }


     # Getting the valid
     dummy_x <- base_dummyVars(x_train)

     # Create a data.frame aux

     # Create a list
     if(length(dummy_x$facVars)!=0){
             for(i in 1:length(dummy_x$facVars)){
                     # See if the levels of the test and train matches
                     if(!all(levels(x_train[[dummy_x$facVars[i]]])==levels(x_test[[dummy_x$facVars[i]]]))){
                        levels(x_test[[dummy_x$facVars[[i]]]]) <- levels(x_train[[dummy_x$facVars[[i]]]])
                     }
                     df_aux <- data.frame( x = x_train[,dummy_x$facVars[i]],y = c_train)
                     formula_aux <- stats::aggregate(y~x,df_aux,mean)
                     formula_aux$y <- rank(formula_aux$y)
                     x_train[[dummy_x$facVars[i]]] <- as.numeric(factor(x_train[[dummy_x$facVars[[i]]]], labels = c(formula_aux$y)))-1

                     # Doing the same for the test set
                     x_test[[dummy_x$facVars[i]]] <- as.numeric(factor(x_test[[dummy_x$facVars[[i]]]], labels = c(formula_aux$y)))-1

             }
     }
     x_train_scale <- as.matrix(x_train)
     x_test_scale <- as.matrix(x_test)

     # Scaling x
     x_min <- apply(as.matrix(x_train_scale),2,min)
     x_max <- apply(as.matrix(x_train_scale),2,max)

     # Storing the original
     x_train_original <- x_train
     x_test_original <- x_test


     # Normalising all the columns
     for(i in 1:ncol(x_train)){
             x_train_scale[,i] <- normalize_covariates_bart(y = x_train_scale[,i],a = x_min[i], b = x_max[i])
             x_test_scale[,i] <- normalize_covariates_bart(y = x_test_scale[,i],a = x_min[i], b = x_max[i])
     }


     # Scaling the 'c' and 'q'
     min_c <- min(c_train)
     max_c <- max(c_train)
     min_q <- min(q_train)
     max_q <- max(q_train)

     # Getting the min and max for each column
     min_x <- apply(x_train_scale,2,min)
     max_x <- apply(x_train_scale, 2, max)

     # Scaling "y"
     if(scale_bool){
        c_scale <- normalize_bart(y = c_train,a = min_c,b = max_c)
        q_scale <- normalize_bart(y = q_train, a = min_q, b = max_q)

        tau_mu <- tau_lambda <- (4*n_tree*(kappa^2))

     } else {
        c_scale <- c_train
        q_scale <- q_train

        tau_mu <- (4*n_tree*(kappa^2))/((max_c-min_c)^2)
        tau_lambda <- (4*n_tree*(kappa^2))/((max_q-min_q)^2)
     }

     # Getting the naive sigma value
     nsigma_c <- naive_sigma(x = x_train_scale,y = c_scale)
     nsigma_q <- naive_sigma(x = x_train_scale, y = q_scale)

     # Define the ensity function
     phalft <- function(x, A, nu){
       f <- function(x){
         2 * gamma((nu + 1)/2)/(gamma(nu/2)*sqrt(nu * pi * A^2)) * (1 + (x^2)/(nu * A^2))^(- (nu + 1)/2)
       }
       integrate(f, lower = 0, upper = x)$value
     }

     # define parameters
     nu <- df
     # solve for A

     A_c <- uniroot(f = function(A){sigquant - phalft(nsigma_c, A, nu)}, interval = c(0.001,1000))$root
     A_q <- uniroot(f = function(A){sigquant - phalft(nsigma_q, A, nu)}, interval = c(0.001,1000))$root

     # Calculating tau hyperparam
     a_tau <- df/2

     # Calculating lambda
     qchi <- stats::qchisq(p = 1-sigquant,df = df,lower.tail = 1,ncp = 0)
     lambda_c <- (nsigma_c*nsigma_c*qchi)/df
     rate_tau_c <- (lambda_c*df)/2

     lambda_q <- (nsigma_q*nsigma_q*qchi)/df
     rate_tau_q <- (lambda_q*df)/2

     # Transforming back the parameters that are going to be used as the Wishart prior
     df_wish <- 2*a_tau
     # s_0_wish <- 0.5*diag(c(rate_tau_c,rate_tau_q))
     s_0_wish <- 0.5*diag(c(1/rate_tau_c,1/rate_tau_q))


     # Remin that this is the precision matrix
     init_P <- solve(stats::rWishart(n = 1,df = df_wish,Sigma = s_0_wish)[,,1])

     mu_init_c <- mean(c_scale)
     mu_init_q <- mean(q_scale)

     if(!is.null(fixed_P)){
       init_P <- fixed_P
     }

     # Generating the BART obj
     bart_obj <- cppbart(x_train_scale,
          c_scale,
          q_scale,
          x_test_scale,
          n_tree,
          node_min_size,
          alpha,
          beta,
          n_mcmc,
          n_burn,
          init_P,
          mu_init_c,
          mu_init_q,
          tau_mu,
          tau_lambda,
          df_wish,
          s_0_wish)


     if(scale_bool){
             # Tidying up the posterior elements
             c_train_post <- unnormalize_bart(z = bart_obj[[1]],a = min_c,b = max_c)
             c_test_post <- unnormalize_bart(z = bart_obj[[3]],a = min_c,b = max_c)
             q_train_post <- unnormalize_bart(z = bart_obj[[2]],a = min_q,b = max_q)
             q_test_post <- unnormalize_bart(z = bart_obj[[4]],a = min_q,b = max_q)

             tau_c_post <- bart_obj[[5]]/((max_c-min_c)^2)
             tau_q_post <- bart_obj[[6]]/((max_q-min_q)^2)
             rho_post <- bart_obj[[7]]

             all_tau_c_post <- bart_obj[[10]]/((max_c-min_c)^2)
             all_tau_q_post <- bart_obj[[11]]/((max_q-min_q)^2)
             all_rho <- bart_obj[[12]]

     } else {
             c_train_post <- bart_obj[[1]]
             c_test_post <- bart_obj[[3]]
             q_train_post <- bart_obj[[2]]
             q_test_post <- bart_obj[[4]]

             tau_c_post <- bart_obj[[5]]
             tau_q_post <- bart_obj[[6]]
             rho_post <- bart_obj[[7]]

             all_tau_c_post <- bart_obj[[10]]
             all_tau_q_post <- bart_obj[[11]]
             all_rho <- bart_obj[[12]]


     }


     # # Ploting some minor tests
     # plot(rowMeans(q_train_post[,1001:1500]),q_train)
     # plot(rowMeans(c_train_post[,1001:1500]),c_train)
     #
     # plot(all_rho, type = "l")
     # plot(all_tau_c_post, type = "l")
     # plot(all_tau_q_post, type = "l")
     #
     # mean(all_rho)

      # Return the list with all objects and parameters
     return(list(c_hat = c_train_post,
                 q_hat = q_train_post,
                 c_hat_test = c_test_post,
                 q_hat_test = q_test_post,
                 tau_c_post = tau_c_post,
                 tau_q_post = tau_q_post,
                 rho_post = rho_post,
                 all_tau_c_post = all_tau_c_post,
                 all_tau_q_post = all_tau_q_post,
                 all_rho_post = all_rho,
                 prior = list(n_tree = n_tree,
                              alpha = alpha,
                              beta = beta,
                              tau_mu = tau_mu,
                              tau_lambda = tau_lambda,
                              a_tau = a_tau,
                              d_tau_c = rate_tau_c,
                              d_tau_q = rate_tau_q),
                 mcmc = list(n_mcmc = n_mcmc,
                             n_burn = n_burn),
                 data = list(x_train = x_train,
                             c_train = c_train,
                             q_train = q_train,
                             x_test = x_test,
                             move_proposal = bart_obj[[8]],
                             move_acceptance = bart_obj[[9]])))
}

