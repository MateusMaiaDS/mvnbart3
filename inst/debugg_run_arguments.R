rm(list=ls())
library(purrr)
library(devtools)

devtools::load_all()
source("inst/test_simulations.R")
x_train <- data_train %>% dplyr::select(dplyr::starts_with("X"))
x_test <- data_test %>% dplyr::select(dplyr::starts_with("X"))
y_mat <- cbind(data_train$C,data_train$Q)
colnames(y_mat) <- c("C","Q")
n_tree = 200
n_mcmc = 2000
n_burn = 500
alpha = 0.9
beta = 2
df = 3
sigquant = 0.9
kappa = 2
scale_bool = FALSE
# Hyperparam for tau_b and tau_b_0
delta = 1
stump <- FALSE
numcut <- 100L
usequants <- FALSE
node_min_size <- 5
