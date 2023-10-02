library(BART)
library(mvnbart)


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
sigma_c <- 1
sigma_q <- 2/3
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
     data_train$C[i] <- (f_true_C(data_train[i,1:4]) + resid[1]) * 10
     data_train$EQ[i] <- f_true_Q(data_train[i,1:4])
     data_train$Q[i] <- (f_true_Q(data_train[i,1:4]) + resid[2]) * 0.1
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
     data_test$C[i] <- (f_true_C(data_test[i,1:4]) + resid[1]) * 10
     data_test$EQ[i] <- f_true_Q(data_test[i,1:4])
     data_test$Q[i] <- (f_true_Q(data_test[i,1:4]) + resid[2]) * 0.1
}

s0 <- matrix(0,nrow = 10, ncol = 10)
for(i in 1:10){
        s0 <- s0 + crossprod(m1[i,,drop = FALSE]-m2[i,,drop = FALSE])
}
