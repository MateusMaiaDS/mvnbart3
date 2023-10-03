## Bart
#' @useDynLib mvnbart3
#' @importFrom Rcpp sourceCpp
#'
# Getting the BART wrapped function
#' @export
mvnbart3 <- function(x_train,
                  y_mat,
                  x_test,
                  n_tree = 100,
                  node_min_size = 5,
                  n_mcmc = 2000,
                  n_burn = 500,
                  alpha = 0.95,
                  beta = 2,
                  df = 3,
                  sigquant = 0.9,
                  kappa = 2,
                  numcut = 100L, # Defining the grid of split rules
                  usequants = FALSE
                  ) {

     # Verifying if it's been using a y_mat matrix
     if(NCOL(y_mat)<2){
         stop("Insert a valid multivariate response. ")
     }

     # Verifying if x_train and x_test are matrices
     if(!is.data.frame(x_train) || !is.data.frame(x_test)){
          stop("Insert valid data.frame for both data and xnew.")
     }


     # Getting the valid
     dummy_x <- base_dummyVars(x_train)

     # Create a data.frame aux

     # Create a list
     if(length(dummy_x$facVars)!=0){

             # Selected rank_var categorical
             rank_var <- 1

             for(i in 1:length(dummy_x$facVars)){
                     # See if the levels of the test and train matches
                     if(!all(levels(x_train[[dummy_x$facVars[i]]])==levels(x_test[[dummy_x$facVars[i]]]))){
                        levels(x_test[[dummy_x$facVars[[i]]]]) <- levels(x_train[[dummy_x$facVars[[i]]]])
                     }
                     df_aux <- data.frame( x = x_train[,dummy_x$facVars[i]], y = y_mat[,rank_var])
                     formula_aux <- stats::aggregate(y~x,df_aux,mean)
                     formula_aux$y <- rank(formula_aux$y)
                     x_train[[dummy_x$facVars[i]]] <- as.numeric(factor(x_train[[dummy_x$facVars[[i]]]], labels = c(formula_aux$y)))-1

                     # Doing the same for the test set
                     x_test[[dummy_x$facVars[i]]] <- as.numeric(factor(x_test[[dummy_x$facVars[[i]]]], labels = c(formula_aux$y)))-1

             }
     }

     # Getting the train and test set
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

     # Creating the numcuts matrix of splitting rules
     xcut_m <- matrix(NA,nrow = numcut,ncol = ncol(x_train_scale))
     for(i in 1:ncol(x_train_scale)){

             if(nrow(x_train_scale)<numcut){
                        xcut_m[,i] <- sort(x_train_scale[,i])
             } else {
                        xcut_m[,i] <- seq(min(x_train_scale[,i]),
                                          max(x_train_scale[,i]),
                                          length.out = numcut+2)[-c(1,numcut+2)]
             }
     }


     # Scaling the y
     min_y <- apply(y_mat,2,min)
     max_y <- apply(y_mat,2,max)

     # Getting the min and max for each column
     min_x <- apply(x_train_scale,2,min)
     max_x <- apply(x_train_scale, 2, max)


     # Defining tau_mu_j
     tau_mu_j <- (4*n_tree*(kappa^2))/((max_y-min_y)^2)

     sigma_mu_j <- tau_mu_j^(-1/2)

     # Getting the naive sigma value
     nsigma <- apply(y_mat, 2, function(Y){naive_sigma(x = x_train_scale,y = Y)})

     # Define the ensity function
     phalft <- function(x, A, nu){
             f <- function(x){
                     2 * gamma((nu + 1)/2)/(gamma(nu/2)*sqrt(nu * pi * A^2)) * (1 + (x^2)/(nu * A^2))^(- (nu + 1)/2)
             }
             stats::integrate(f, lower = 0, upper = x)$value
     }

     # Define parameters
     nu <- df

     A_j <- numeric()

     for(i in 1:length(nsigma)){
             # Calculating lambda
             A_j[i] <- stats::optim(par = 0.01, f = function(A){(sigquant - phalft(nsigma[i], A, nu)^2)},
                           method = "Brent",lower = 0.00001,upper = 100)$par
     }

     # Calculating lambda
     qchi <- stats::qchisq(p = 1-sigquant,df = df,lower.tail = 1,ncp = 0)
     lambda <- (nsigma*nsigma*qchi)/df
     a_j_init <- (lambda*df)/2

     S_0_wish <- diag(1/a_j_init)/(2*nu)

     # Call the bart function
     Sigma_init <- diag(nsigma^2)
     mu_init <- apply(y_mat,2,mean)


     # Generating the BART obj
     bart_obj <- cppbart(x_train_scale,
                          y_mat,
                          x_test_scale,
                          xcut_m,
                          n_tree,
                          node_min_size,
                          n_mcmc,
                          n_burn,
                          Sigma_init,
                          mu_init,
                          sigma_mu_j,
                          alpha,beta,nu,
                          S_0_wish,
                          A_j,
                          a_j_init)


     # Returning the main components from the model
     y_train_post <- bart_obj[[1]]
     y_test_post <- bart_obj[[2]]
     Sigma_post <- bart_obj[[3]]
     all_Sigma_post <- bart_obj[[4]]


     # Getting the mean values for the Sigma and \y_hat and \y_hat_test
     Sigma_for <- matrix(0,nrow = nrow(Sigma_post), ncol = ncol(Sigma_post))
     y_train_for <- matrix(0,nrow = nrow(y_mat),ncol = ncol(y_mat))
     y_test_for <- matrix(0,nrow = nrow(x_test),ncol = ncol(y_mat))

     for(i in 1:(dim(Sigma_post)[3])){
             Sigma_for <- Sigma_for + Sigma_post[,,i]
             y_train_for <- y_train_for + y_train_post[,,i]
             y_test_for <- y_test_for + y_test_post[,,i]
     }

     Sigma_post_mean <- Sigma_for/dim(Sigma_post)[3]
     y_mat_mean <- y_train_for/dim(y_train_post)[3]
     y_mat_test_mean <- y_test_for/dim(y_test_post)[3]
     sigmas_mean <- sqrt(diag(Sigma_post_mean))

     # Return the list with all objects and parameters
     return(list(y_hat = y_train_post,
                 y_hat_test = y_test_post,
                 y_mat_mean = y_mat_mean,
                 y_mat_test_mean = y_mat_test_mean,
                 Sigma_post = Sigma_post,
                 Sigma_post_mean = Sigma_post_mean,
                 sigmas_mean = sigmas_mean,
                 all_Sigma_post = all_Sigma_post,
                 prior = list(n_tree = n_tree,
                              alpha = alpha,
                              beta = beta,
                              tau_mu_j = tau_mu_j,
                              df = df,
                              A_j = A_j,
                              mu_init = mu_init,
                              tree_proposal = bart_obj[[5]],
                              tree_acceptance = bart_obj[[6]]),
                 mcmc = list(n_mcmc = n_mcmc,
                             n_burn = n_burn),
                 data = list(x_train = x_train,
                             y_mat = y_mat,
                             x_test = x_test)))
}

