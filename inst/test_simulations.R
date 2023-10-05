library(BART)
# library(mvnbart3)
rm(list=ls())
load_all()
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
sigma_q <- 1
rho <- 0.1
Sigma <- matrix(c(sigma_c^2,sigma_c*sigma_q*rho,sigma_c*sigma_q*rho,sigma_q^2), nrow = 2)
Sigma_chol <- t(chol(Sigma))

# sample size
N <- 500

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

# Getting y_mat element
y_mat <- cbind(data_train$C,data_train$Q)
x_train <- data_train[,1:4]
x_test <- data_test[,1:4]
colnames(y_mat) <- c("C","Q")

mvbart_mod <- mvnbart3(x_train = x_train,
                   y_mat = y_mat,
                   x_test = x_test,
                   n_tree = 200,
                   n_mcmc = 2500,df = 3,
                   n_burn = 500,Sigma_init = Sigma)


# mvbart_mod$Sigma_post_mean %>% sqrt
mvbart_mod$Sigma_post_mean[1,2]/((sqrt(mvbart_mod$Sigma_post_mean[1,1])*sqrt(mvbart_mod$Sigma_post_mean[2,2])))
plot(y_mat[,2],mvbart_mod$y_mat_mean[,2])
plot(y_mat[,1],mvbart_mod$y_mat_mean[,1])

# Getting univariate predictions and comparing it with BART
c_hat <- dbarts::bart(x.train = x_train,y.train = y_mat[,1],x.test = x_test,ntree = 100)
q_hat <- dbarts::bart(x.train = x_train,y.train = y_mat[,2],x.test = x_test,ntree = 100)

plot(q_hat$yhat.train.mean,y_mat[,2])
plot(c_hat$yhat.train.mean,mvbart_mod$y_mat_mean[,1])
plot(q_hat$yhat.train.mean,mvbart_mod$y_mat_mean[,2])
plot(y_mat[,2],mvbart_mod$y_mat_mean[,2])

rmse(c_hat$yhat.train.mean,mvbart_mod$y_mat_mean[,1])
rmse(q_hat$yhat.train.mean,mvbart_mod$y_mat_mean[,2])


crossprod((y_mat-mvbart_mod$y_mat_mean))
sum((y_mat[,1]-mvbart_mod$y_mat_mean[,1])*(y_mat[,2]-mvbart_mod$y_mat_mean[,2]))

sigma_one <- sigma_two <- numeric()
for(i in 1:dim(mvbart_mod$all_Sigma_post)[3]){
        sigma_one[i] <- sqrt(mvbart_mod$all_Sigma_post[1,1,i])
        sigma_two[i] <- sqrt(mvbart_mod$all_Sigma_post[2,2,i])

}

plot(sigma_one, type = "l")
plot(sigma_two, type = "l")
