library(bayesplot)
# library(mvnbart3)
devtools::load_all()

# functions ####
f_true <- function(X){
      as.numeric(10*sin(pi*X[1]*X[2]) + 20*(X[3] - 0.5)^2 + 10*X[4] + 5*X[5])
}
set.seed(2)
# simulation - multiple covariates
n <- 100
p <- 10
# true covariance matrix for residuals
sigma_c <- 1
sigma_q <- 1
rho <- 0.9
Sigma <- matrix(c(sigma_c^2,sigma_c*sigma_q*rho,sigma_c*sigma_q*rho,sigma_q^2), nrow = 2)
Sigma_chol <- t(chol(Sigma))

X_train <- matrix(runif(n*p, 0, 1), nrow = n, ncol = p)
X_train <- as.data.frame(X_train)
Y_train <- data.frame(index = 1:n)
Y_train$Y1 <- NA
Y_train$EY1 <- NA
Y_train$Y2 <- NA
Y_train$EY2 <- NA

for (i in 1:n){
      resid <- Sigma_chol %*% rnorm(2)
      Y_train$EY1[i] <- f_true(X_train[i,1:5])
      Y_train$Y1[i] <- f_true(X_train[i,1:5]) + resid[1]
      Y_train$EY2[i] <- f_true(X_train[i,6:10])
      Y_train$Y2[i] <- f_true(X_train[i,6:10]) + resid[2]
}

X_test <- matrix(runif(n*p, 0, 1), nrow = n, ncol = p)
X_test <- as.data.frame(X_test)
Y_test <- data.frame(index = 1:n)
Y_test$Y1 <- NA
Y_test$EY1 <- NA
Y_test$Y2 <- NA
Y_test$EY2 <- NA


for (i in 1:n){
      resid <- Sigma_chol %*% rnorm(2)
      Y_test$EY1[i] <- f_true(X_test[i,1:5])
      Y_test$Y1[i] <- f_true(X_test[i,1:5]) + resid[1]
      Y_test$EY2[i] <- f_true(X_test[i,6:10])
      Y_test$Y2[i] <- f_true(X_test[i,6:10]) + resid[2]
}

# Y_train <- apply(Y_train,2,scale)
y_mat = as.matrix(Y_train[,c("Y1","Y2")])
x_train = X_train
x_test = X_test

mvnBart_fit <- mvnbart3(x_train = X_train,
                        y_mat = as.matrix(Y_train[,c("Y1","Y2")]),
                        x_test = X_test,
                        n_tree = 100,
                        n_mcmc = 2000,df = 3,
                        n_burn = 0,
                        Sigma_init = Sigma,
                        update_Sigma = TRUE,update_A_j = TRUE)


Sigma_post_df <- data.frame(
      sigma1 = sqrt(mvnBart_fit$Sigma_post[1,1,]),
      sigma2 = sqrt(mvnBart_fit$Sigma_post[2,2,]),
      rho = mvnBart_fit$Sigma_post[1,2,]/(sqrt(mvnBart_fit$Sigma_post[1,1,] * mvnBart_fit$Sigma_post[2,2,]))
)
mcmc_trace(Sigma_post_df)


y_hat_df <- data.frame(
      y1_hat = mvnBart_fit$y_hat[10,1,],
      y2_hat = mvnBart_fit$y_hat[10,2,]
)
mcmc_trace(y_hat_df)
