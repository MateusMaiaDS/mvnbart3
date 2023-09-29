library(purrr)
x_train <- data_train %>% dplyr::select(dplyr::starts_with("X"))
x_test <- data_test %>% dplyr::select(dplyr::starts_with("X"))
y_mat <- cbind(data_train$C,data_train$Q)
colnames(y_mat) <- c("C","Q")
n_tree = 20
n_mcmc = 2000
n_burn = 500
alpha = 0.95
beta = 2
dif_order = 0
nIknots = 20
df = 3
sigquant = 0.9
kappa = 2
scale_bool = FALSE
# Hyperparam for tau_b and tau_b_0
nu = 2
delta = 1
a_delta = 0.0001
d_delta = 0.0001
df_tau_b = 3
prob_tau_b = 0.9
stump <- FALSE
numcut <- 100L
usequants <- FALSE
