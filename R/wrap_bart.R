## Bart
#' @useDynLib mvnbart3
#' @importFrom Rcpp sourceCpp
#'


# Getting the BART wrapped function
#' @export
mvnbart3 <- function(x_train,
                  y_mat,
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
                  tau = 100,
                  scale_bool = TRUE,
                  stump = FALSE,
                  no_rotation_bool = FALSE,
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
             for(i in 1:length(dummy_x$facVars)){
                     # See if the levels of the test and train matches
                     if(!all(levels(x_train[[dummy_x$facVars[i]]])==levels(x_test[[dummy_x$facVars[i]]]))){
                        levels(x_test[[dummy_x$facVars[[i]]]]) <- levels(x_train[[dummy_x$facVars[[i]]]])
                     }
                     df_aux <- data.frame( x = x_train[,dummy_x$facVars[i]],y)
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

     # Calculating lambda
     A_j <- sapply(nsigma, function(sigma){
             stats::optim(par = 0.01, f = function(A){(sigquant - phalft(sigma, A, nu)^2)},
                   method = "Brent",lower = 0.000001,upper = 100)$par
     })

     a_j_init <- sapply(A_j,function(A_j_){1/stats::rgamma(n = 1,shape = 2,scale = A_j_^2)})

     S_0_wish <- 2*nu*diag(1/a_j_init)

     # Call the bart function
     Sigma_init <- diag(nsigma)

     mu_init <- apply(y_mat,2,mean)

     # Creating the vector that stores all trees
     all_tree_post <- vector("list",length = round(n_mcmc-n_burn))

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
                          a_j_init,
                          stump)


     if(scale_bool){
             # Tidying up the posterior elements
             y_train_post <- unnormalize_bart(z = bart_obj[[1]],a = min_y,b = max_y)
             y_test_post <- unnormalize_bart(z = bart_obj[[2]],a = min_y,b = max_y)

             Sigma <- bart_obj[[3]]/((max_y-min_y)^2)
             all_Sigma_post <- bart_obj[[4]]/((max_y-min_y)^2)
     } else {
             y_train_post <- bart_obj[[1]]
             y_test_post <- bart_obj[[2]]
             Sigma_post <- bart_obj[[3]]
             all_Sigma_post <- bart_obj[[4]]
     }

     par(mfrow=c(1,2))
     y_train_post %>% apply(c(1,2),mean) %>% .[,1] %>% plot(.,y_mat[,1])
     y_train_post %>% apply(c(1,2),mean) %>% .[,2] %>% plot(.,y_mat[,2])


     sigma_one <- sigma_two <- numeric()
     for(i in 1:(dim(Sigma_post)[3])){
             sigma_one[i] <- Sigma_post[1,1,i]
             sigma_two[i] <- Sigma_post[2,2,i]

     }
     sigma_one %>% plot(type = 'l')
     sigma_two %>% plot(type = 'l')

     # Return the list with all objects and parameters
     return(list(y_hat = y_train_post,
                 y_hat_test = y_test_post,
                 Sigma_post = Sigma_post,
                 all_Sigma_post = all_Sigma_post,
                 all_tree_post = all_tree_post,
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

