############################################################
# FINAL CASE STUDY SCRIPT
# Synthetic beer fermentation + PCA-MSPC + XGBoost + SHAP
# Harmonized multipanel figure
############################################################

rm(list = ls())
graphics.off()

set.seed(12345)

############################################################
# PACKAGES
############################################################

library(dplyr)
library(qcc)
library(xgboost)
library(pROC)
library(corrplot)
library(ggplot2)
library(patchwork)
library(scales)
library(grid)
library(SHAPforxgboost)

############################################################
# 1) SYNTHETIC DATA FOR FERMENTATION PROCESS
############################################################

n_batches <- 500
n_samples <- 50
N <- n_batches * n_samples

# Fermentation time: 0-168 h (7 days)
tiempo_h <- rep(seq(0, 168, length.out = n_samples), n_batches)

# Batch ID
batch <- rep(1:n_batches, each = n_samples)

############################################################
# PROCESS SENSORS
############################################################

# Temperature: slight variability + noise
temperatura <- runif(N, 12, 15) + rnorm(N, 0, 0.5)

# pH: logistic decrease during fermentation
ph <- 4.5 - 0.3 * plogis(tiempo_h / 60) + rnorm(N, 0, 0.05)

# Degrees Plato: exponential decrease during fermentation
plato <- 13 - 10 * (1 - exp(-tiempo_h / 60)) + rnorm(N, 0, 0.1)

# CO2: increases during fermentation
co2_gL <- 4.7 + 0.2 * (1 - exp(-tiempo_h / 60)) + rnorm(N, 0, 0.1)

# Dissolved oxygen: decays with time + noise
oxigeno_ppb <- 80 * exp(-tiempo_h / 60) + 5 + rnorm(N, 0, 5)

############################################################
# FAULTY PROCESS
############################################################

fault_batches <- sample(1:n_batches, size = round(0.08 * n_batches))
fault_type <- sample(c("F1", "F2", "F3"), length(fault_batches), replace = TRUE)

for (i in seq_along(fault_batches)) {
  
  b <- fault_batches[i]
  idx <- batch == b
  
  if (fault_type[i] == "F1") {
    # Temperature deviation
    temperatura[idx] <- temperatura[idx] + rnorm(sum(idx), 3, 0.4)
  }
  
  if (fault_type[i] == "F2") {
    # Multivariate disturbance: pH, O2, CO2
    ph[idx] <- ph[idx] + rnorm(sum(idx), 0.15, 0.04)
    oxigeno_ppb[idx] <- oxigeno_ppb[idx] + runif(sum(idx), 15, 25)
    co2_gL[idx] <- co2_gL[idx] - runif(sum(idx), 0.2, 0.5)
  }
  
  if (fault_type[i] == "F3") {
    # Incomplete fermentation: high final Plato
    final_idx <- idx & tiempo_h > 100
    plato[final_idx] <- plato[final_idx] + rnorm(sum(final_idx), 1.5, 0.3)
  }
}

############################################################
# CHEMICAL AND SENSORY PARAMETERS
############################################################

alcohol_pct <- 4.5 + 0.3 * (13 - plato) + rnorm(N, 0, 0.15)

ibu_amargor <- runif(N, 0, 30) + 0.01 * (temperatura - 18)^2

proteinas_gL <- runif(N, 0.8, 1.5)

espuma_estable <- 300 -
  12 * proteinas_gL -
  7 * abs(temperatura - 18) -
  5 * abs(ph - 4.3) +
  rnorm(N, 0, 3)

sabor_score <- 85 +
  0.25 * ibu_amargor +
  2.5 * proteinas_gL +
  1.2 * co2_gL -
  1.8 * temperatura -
  0.5 * oxigeno_ppb +
  rnorm(N, 0, 1)

# Extra penalty in faulty batches
sabor_score[batch %in% fault_batches] <-
  sabor_score[batch %in% fault_batches] -
  abs(rnorm(sum(batch %in% fault_batches), 4, 1))

############################################################
# FULL TIME-SERIES DATASET
############################################################

dataset_cerveza <- data.frame(
  batch,
  tiempo_h,
  temperatura,
  ph,
  plato,
  co2_gL,
  oxigeno_ppb,
  alcohol_pct,
  ibu_amargor,
  proteinas_gL,
  espuma_estable,
  sabor_score
)

# Synthetic quality index
dataset_cerveza$quality_index <-
  0.4 * dataset_cerveza$sabor_score +
  0.3 * dataset_cerveza$espuma_estable +
  0.2 * dataset_cerveza$alcohol_pct -
  0.7 * dataset_cerveza$oxigeno_ppb -
  rnorm(nrow(dataset_cerveza), 0, 5)

threshold <- quantile(dataset_cerveza$quality_index, 0.70)
dataset_cerveza$calidad_ok <- ifelse(dataset_cerveza$quality_index > threshold, 1, 0)

dataset_cerveza$fault <- ifelse(batch %in% fault_batches, 1, 0)

############################################################
# BATCH-LEVEL DATASET
############################################################

beer_batches <- dataset_cerveza %>%
  group_by(batch) %>%
  arrange(tiempo_h, .by_group = TRUE) %>%
  summarise(
    temp_mean = mean(temperatura),
    temp_sd = sd(temperatura),
    plato_final = last(plato),
    alcohol_final = last(alcohol_pct),
    ibu_final = mean(ibu_amargor),
    ph_final = last(ph),
    proteinas_gL_final = last(proteinas_gL),
    oxigeno_final = last(oxigeno_ppb),
    co2_gL_final = last(co2_gL),
    espuma_final = last(espuma_estable),
    sabor_final = last(sabor_score),
    ph_drop = first(ph) - last(ph),
    attenuation_P = (first(plato) - last(plato)) / first(plato),
    co2_growth = last(co2_gL) - first(co2_gL),
    temp_range = max(temperatura) - min(temperatura),
    calidad = last(calidad_ok),
    .groups = "drop"
  )

beer_batches$fault <- ifelse(beer_batches$batch %in% fault_batches, 1, 0)
fault_df <- data.frame(batch = fault_batches, fault_type = fault_type)
beer_batches <- merge(beer_batches, fault_df, by = "batch", all.x = TRUE)

# Optional export
write.csv(dataset_cerveza, "fermentacion_cerveza_5000obs.csv", row.names = FALSE)
write.csv(beer_batches, "beer_batches_SPC_500batches.csv", row.names = FALSE)

############################################################
# CORRELATION PLOT
############################################################

corr_matrix <- cor(beer_batches[, c(
  "temp_range",
  "alcohol_final",
  "ph_final",
  "ibu_final",
  "proteinas_gL_final",
  "oxigeno_final",
  "co2_gL_final",
  "attenuation_P",
  "espuma_final",
  "sabor_final",
  "calidad"
)])

corrplot(
  corr_matrix,
  method = "color",
  tl.cex = 0.8,
  addCoef.col = "black",
  number.cex = 0.5
)

############################################################
# TRAIN-TEST SPLIT BEFORE PREPROCESSING OR MODEL FITTING
############################################################

set.seed(12345)

train_idx <- beer_batches %>%
  group_by(calidad) %>%
  slice_sample(prop = 0.80) %>%
  pull(batch)

train <- beer_batches %>%
  filter(batch %in% train_idx)

test <- beer_batches %>%
  filter(!batch %in% train_idx)

# In-control reference data used to fit the PCA-MSPC model
reference_train <- train %>%
  filter(fault == 0)

############################################################
# 2) MSPC: PCA + HOTELLING T2 + SPE(Q)
############################################################

predictors <- c(
  "temp_mean", "temp_sd", "temp_range",
  "ph_final", "ph_drop",
  "co2_gL_final", "co2_growth",
  "oxigeno_final",
  "espuma_final",
  "alcohol_final",
  "attenuation_P",
  "ibu_final",
  "proteinas_gL_final"
)

############################################################
# PCA-MSPC MODEL FITTED ONLY ON IN-CONTROL TRAINING BATCHES
############################################################

pca_model <- prcomp(
  reference_train[, predictors],
  center = TRUE,
  scale. = TRUE
)

explained_var <- cumsum(
  pca_model$sdev^2 / sum(pca_model$sdev^2)
)

plot(
  explained_var,
  type = "b",
  main = "Explained Variance PCA"
)

# Minimum number of PCs explaining at least 85% cumulative variance
ncomp <- which(explained_var >= 0.85)[1]

if (is.na(ncomp) || length(ncomp) == 0) {
  ncomp <- 4
}

print(paste("Number of retained principal components:", ncomp))
print(paste(
  "Cumulative explained variance:",
  round(explained_var[ncomp], 4)
))

P_loadings <- pca_model$rotation[, 1:ncomp, drop = FALSE]
Lambda <- diag(pca_model$sdev[1:ncomp]^2)

############################################################
# FUNCTION TO APPLY THE REFERENCE PCA MODEL
############################################################

apply_mspc_model <- function(new_data, pca_model, predictors,
                             ncomp, P_loadings, Lambda) {
  
  # Apply centre and scale estimated only from reference_train
  X_scaled <- scale(
    new_data[, predictors],
    center = pca_model$center,
    scale = pca_model$scale
  )
  
  # Project observations onto the retained PCA subspace
  T_scores <- X_scaled %*% P_loadings
  
  # Hotelling T2
  T2 <- apply(T_scores, 1, function(ti) {
    as.numeric(t(ti) %*% solve(Lambda) %*% ti)
  })
  
  # Reconstruct observations and calculate SPE/Q
  X_hat <- T_scores %*% t(P_loadings)
  residuals <- X_scaled - X_hat
  Q <- rowSums(residuals^2)
  
  data.frame(
    T2 = T2,
    Q = Q
  )
}

############################################################
# CONTROL LIMITS ESTIMATED ONLY FROM REFERENCE TRAINING DATA
############################################################

reference_scores <- apply_mspc_model(
  new_data = reference_train,
  pca_model = pca_model,
  predictors = predictors,
  ncomp = ncomp,
  P_loadings = P_loadings,
  Lambda = Lambda
)

# Hotelling T2 limit: F-distribution approximation, 95%
n_ref <- nrow(reference_train)
alpha_T2 <- 0.95

T2_limit <- (
  ncomp * (n_ref - 1) / (n_ref - ncomp)
) * qf(
  alpha_T2,
  ncomp,
  n_ref - ncomp
)

# SPE/Q limit: Jackson-Mudholkar approximation, 99%
residual_eigenvalues <- pca_model$sdev[
  (ncomp + 1):length(pca_model$sdev)
]^2

residual_eigenvalues <- residual_eigenvalues[
  !is.na(residual_eigenvalues)
]

theta1 <- sum(residual_eigenvalues)
theta2 <- sum(residual_eigenvalues^2)
theta3 <- sum(residual_eigenvalues^3)

h0 <- 1 - (
  2 * theta1 * theta3
) / (
  3 * theta2^2
)

z <- qnorm(0.99)

Q_limit <- theta1 * (
  (
    z * sqrt(2 * theta2 * h0^2) / theta1
  ) +
    1 +
    (
      theta2 * h0 * (h0 - 1)
    ) / theta1^2
)^(1 / h0)

############################################################
# APPLY THE FIXED PCA-MSPC MODEL TO TRAIN AND TEST
############################################################

train_mspc <- apply_mspc_model(
  new_data = train,
  pca_model = pca_model,
  predictors = predictors,
  ncomp = ncomp,
  P_loadings = P_loadings,
  Lambda = Lambda
)

test_mspc <- apply_mspc_model(
  new_data = test,
  pca_model = pca_model,
  predictors = predictors,
  ncomp = ncomp,
  P_loadings = P_loadings,
  Lambda = Lambda
)

train$T2 <- train_mspc$T2
train$Q <- train_mspc$Q

test$T2 <- test_mspc$T2
test$Q <- test_mspc$Q

train$T2_outlier <- as.integer(train$T2 > T2_limit)
train$Q_outlier <- as.integer(train$Q > Q_limit)
train$MSPC_outlier <- as.integer(
  train$T2_outlier == 1 | train$Q_outlier == 1
)

test$T2_outlier <- as.integer(test$T2 > T2_limit)
test$Q_outlier <- as.integer(test$Q > Q_limit)
test$MSPC_outlier <- as.integer(
  test$T2_outlier == 1 | test$Q_outlier == 1
)

# Reconstruct the complete dataset only for plotting
beer_batches <- bind_rows(train, test) %>%
  arrange(batch)

print(paste("Reference training batches:", n_ref))
print(paste("T2 control limit:", round(T2_limit, 4)))
print(paste("Q control limit:", round(Q_limit, 4)))

# MSPC flags
beer_batches$T2_outlier <- ifelse(beer_batches$T2 > T2_limit, 1, 0)
beer_batches$Q_outlier  <- ifelse(beer_batches$Q > Q_limit, 1, 0)
beer_batches$MSPC_outlier <- ifelse(
  beer_batches$T2_outlier == 1 | beer_batches$Q_outlier == 1, 1, 0
)

# Optional T2 chart
qcc(
  beer_batches$T2,
  type = "xbar.one",
  center = mean(beer_batches$T2),
  limits = c(0, T2_limit),
  title = "Hotelling T2 Control Chart (PCA)"
)

############################################################
# 3) XGBOOST: ROBUST AND REPRODUCIBLE
############################################################

features <- c(
  "temp_mean", "temp_sd", "temp_range",
  "ph_final", "ph_drop",
  "co2_gL_final", "co2_growth",
  "oxigeno_final",
  "attenuation_P",
  "ibu_final",
  "proteinas_gL_final",
  "T2", "Q", "MSPC_outlier"
)

set.seed(12345)


X_train <- as.matrix(train[, features])
X_test  <- as.matrix(test[, features])

y_train <- as.numeric(train$calidad)
y_test  <- as.numeric(test$calidad)

dtrain <- xgb.DMatrix(X_train, label = y_train)
dtest  <- xgb.DMatrix(X_test, label = y_test)

scale_pos_weight <- sum(y_train == 0) / sum(y_train == 1)

params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  scale_pos_weight = scale_pos_weight,
  eta = 0.05,
  max_depth = 4,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1,
  lambda = 1,
  alpha = 0,
  tree_method = "hist",
  nthread = 1,
  seed = 12345
)

set.seed(12345)

cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 200,
  nfold = 5,
  stratified = TRUE,
  early_stopping_rounds = 10,
  verbose = 0
)

best_nrounds <- NULL

if (!is.null(cv$early_stop) && !is.null(cv$early_stop$best_iteration)) {
  best_nrounds <- cv$early_stop$best_iteration
}

if (is.null(best_nrounds)) {
  best_nrounds <- attr(cv, "best_iteration")
}

if (is.null(best_nrounds) && !is.null(cv$evaluation_log)) {
  metric_col <- grep("^test.*logloss.*mean$", names(cv$evaluation_log), value = TRUE)
  if (length(metric_col) >= 1) {
    best_nrounds <- which.min(cv$evaluation_log[[metric_col[1]]])
  }
}

if (is.null(best_nrounds) || length(best_nrounds) == 0 || is.na(best_nrounds)) {
  best_nrounds <- nrow(cv$evaluation_log)
}

best_nrounds <- as.integer(best_nrounds)

set.seed(12345)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

pred <- predict(xgb_model, dtest)

ROC <- pROC::roc(
  response = y_test,
  predictor = pred,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

AUC <- as.numeric(pROC::auc(ROC))

decision_threshold <- 0.50

pred_class <- ifelse(
  pred >= decision_threshold,
  1,
  0
)

# Positive class:
# calidad = 1 means acceptable quality
# calidad = 0 means non-acceptable quality

conf_matrix <- table(pred_class, real = y_test)

TP <- sum(pred_class == 1 & y_test == 1)
TN <- sum(pred_class == 0 & y_test == 0)
FP <- sum(pred_class == 1 & y_test == 0)
FN <- sum(pred_class == 0 & y_test == 1)

accuracy <- mean(pred_class == y_test)
precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
TPR <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
TNR <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))
balanced_accuracy <- mean(c(TPR, TNR), na.rm = TRUE)
FPR <- ifelse((FP + TN) == 0, NA, FP / (FP + TN))
F1 <- ifelse(is.na(precision) || is.na(TPR) || (precision + TPR) == 0, NA,
             2 * (precision * TPR) / (precision + TPR))

mcc_den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
mcc <- ifelse(mcc_den == 0, NA, (TP * TN - FP * FN) / mcc_den)

print(conf_matrix)
print(accuracy)
print(AUC)
print(balanced_accuracy)
print(decision_threshold)
print(best_nrounds)
print(mcc)

############################################################
# 4) SHAP
############################################################

shap_values <- shap.values(xgb_model = xgb_model, X_train = X_train)
shap_df <- as.data.frame(shap_values$shap_score)
xtrain_df <- as.data.frame(X_train)

feature_names <- intersect(colnames(shap_df), colnames(xtrain_df))

shap_long_plot <- do.call(
  rbind,
  lapply(feature_names, function(f) {
    data.frame(
      Feature = f,
      SHAP = shap_df[[f]],
      FeatureValue = xtrain_df[[f]]
    )
  })
)

feature_order <- shap_long_plot %>%
  group_by(Feature) %>%
  summarise(mean_abs_shap = mean(abs(SHAP), na.rm = TRUE), .groups = "drop") %>%
  arrange(mean_abs_shap) %>%
  pull(Feature)

top_features <- tail(feature_order, 12)

shap_long_plot <- shap_long_plot %>%
  filter(Feature %in% top_features) %>%
  mutate(
    Feature = factor(Feature, levels = top_features),
    FeatureValueScaled = scales::rescale(FeatureValue)
  )

############################################################
# 5) HYBRID DECISION MAP
############################################################

test$pred_prob <- pred
test$pred_class <- pred_class

# Rebuild MSPC status directly on the test set
#test$MSPC_outlier <- ifelse(test$T2 > T2_limit | test$Q > Q_limit, 1, 0)

test$hybrid_state <- "OK"
test$hybrid_state[test$MSPC_outlier == 0 & test$pred_class == 0] <- "RISK"
test$hybrid_state[test$MSPC_outlier == 1 & test$pred_class == 0] <- "FAIL"
test$hybrid_state[test$MSPC_outlier == 1 & test$pred_class == 1] <- "ANOMALY"

test$hybrid_state <- factor(
  test$hybrid_state,
  levels = c("OK", "RISK", "FAIL", "ANOMALY")
)

print(table(test$hybrid_state, test$fault))

ncomp
explained_var[ncomp]
T2_limit
Q_limit
decision_threshold
best_nrounds
AUC
accuracy
balanced_accuracy
TPR
TNR
############################################################
# 6) HARMONIZED MULTIPANEL FIGURE
############################################################

theme_case <- theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0),
    axis.title = element_text(face = "bold", size = 12.5),
    axis.text = element_text(colour = "black", size = 10.5),
    panel.grid.major = element_line(linewidth = 0.25, colour = "grey90"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "grey60", fill = NA, linewidth = 0.5),
    plot.background = element_rect(fill = "white", colour = NA),
    panel.background = element_rect(fill = "white", colour = NA),
    legend.background = element_rect(fill = alpha("white", 0.92), colour = "grey70"),
    legend.key = element_blank(),
    legend.title = element_text(face = "bold", size = 10.5),
    legend.text = element_text(size = 10),
    plot.margin = margin(8, 12, 8, 10)
  )

# theme_case <- theme_bw(base_size = 11) +
#   theme(
#     plot.title = element_text(face = "bold", size = 11, hjust = 0),
#     axis.title = element_text(face = "bold"),
#     axis.text = element_text(colour = "black"),
#     panel.grid.major = element_line(linewidth = 0.25, colour = "grey90"),
#     panel.grid.minor = element_blank(),
#     panel.border = element_rect(colour = "grey60", fill = NA, linewidth = 0.5),
#     plot.background = element_rect(fill = "white", colour = NA),
#     panel.background = element_rect(fill = "white", colour = NA),
#     legend.background = element_rect(fill = alpha("white", 0.92), colour = "grey70"),
#     legend.key = element_blank(),
#     legend.title = element_text(face = "bold", size = 8.5),
#     legend.text = element_text(size = 8.5),
#     plot.margin = margin(8, 10, 8, 8)
#   )

col_fault <- c(
  "0" = "#9A9A9A",
  "1" = "#8C2D19"
)

col_state <- c(
  "OK"      = "#2F6F62",
  "RISK"    = "#B5882A",
  "FAIL"    = "#A3472A",
  "ANOMALY" = "#4C78A8"
)

# Panel A
pA <- ggplot(
  beer_batches,
  aes(x = T2, y = Q, colour = factor(fault))
) +
  geom_point(size = 2.2, alpha = 0.85) +
  geom_vline(
    xintercept = T2_limit,
    linetype = "dashed",
    linewidth = 0.5,
    colour = "#B44D3A"
  ) +
  geom_hline(
    yintercept = Q_limit,
    linetype = "dashed",
    linewidth = 0.5,
    colour = "#B44D3A"
  ) +
  scale_colour_manual(
    values = col_fault,
    labels = c("Normal batch", "Faulty batch"),
    name = NULL
  ) +
  guides(
    colour = guide_legend(
      nrow = 1,
      override.aes = list(size = 3.2, alpha = 1)
    )
  ) +
  labs(
    title = "(A) PCA-based MSPC monitoring",
    x = expression("Hotelling's " * T^2),
    y = "SPE (Q)"
  ) +
  theme_case +
  theme(
    legend.position = c(0.60, 0.35),
    legend.justification = c(0, 1),
    legend.direction = "horizontal"
  )

# Panel B
perf_label <- paste0(
  "AUC = ", sprintf("%.2f", AUC),
  "\nBalanced accuracy = ", sprintf("%.2f", balanced_accuracy),
  "\nAccuracy = ", sprintf("%.2f", accuracy),
  "\nRecall = ", sprintf("%.2f", TPR),
  "\nTNR = ", sprintf("%.2f", TNR)
)

pB <- pROC::ggroc(ROC, linewidth = 0.85, colour = "black", legacy.axes = TRUE) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    linewidth = 0.45,
    colour = "grey65"
  ) +
  annotate(
    "label",
    x = 0.34, y = 0.30,
    label = perf_label,
    size = 3.3,
    hjust = 0,
    vjust = 1,
    lineheight = 0.96,
    label.size = 0.25,
    fill = alpha("white", 0.94)
  ) +
  labs(
    title = "(B) ROC curve",
    x = "1 - Specificity",
    y = "Sensitivity"
  ) +
  coord_equal() +
  theme_case +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 13),
    axis.title = element_text(size = 12.5),
    axis.text = element_text(size = 10.5)
  )


# Panel C (SHAP)
feature_labels_en <- c(
  oxigeno_final = "Final dissolved oxygen",
  temp_mean = "Mean temperature",
  ph_final = "Final pH",
  co2_growth = "CO2 growth",
  attenuation_P = "Apparent attenuation",
  co2_gL_final = "Final CO2",
  temp_range = "Temperature range",
  proteinas_gL_final = "Final protein concentration",
  temp_sd = "Temperature SD",
  T2 = "Hotelling's T2",
  ibu_final = "Final bitterness (IBU)",
  Q = "SPE (Q)",
  ph_drop = "pH drop",
  MSPC_outlier = "MSPC outlier flag"
)

blue_pal <- c(
  "#DCEAF7",
  "#B9D7EE",
  "#8CBFE3",
  "#5AA2D6",
  "#2C7FB8",
  "#08519C"
)

pC <- ggplot(
  shap_long_plot,
  aes(x = SHAP, y = Feature, colour = FeatureValueScaled)
) +
  geom_vline(
    xintercept = 0,
    linewidth = 0.35,
    colour = "grey65"
  ) +
  geom_point(
    alpha = 0.70,
    size = 1.55,
    position = position_jitter(height = 0.15, width = 0)
  ) +
  scale_colour_gradientn(
    colours = blue_pal,
    values = scales::rescale(c(0, 0.2, 0.4, 0.6, 0.8, 1)),
    name = "Feature value",
    labels = c("Low", "High"),
    breaks = c(0, 1),
    guide = guide_colourbar(
      title.position = "top",
      title.hjust = 0.5,
      label.position = "bottom",
      ticks = FALSE,
      barwidth = unit(28, "mm"),
      barheight = unit(3.2, "mm")
    )
  ) +
  scale_y_discrete(labels = feature_labels_en) +
  labs(
    title = "(C) SHAP summary plot",
    x = "SHAP value",
    y = NULL
  ) +
  theme_case +
  theme(
    axis.text.y = element_text(size = 10.2),
    axis.text.x = element_text(size = 10.5),
    axis.title.x = element_text(size = 12.5),
    plot.title = element_text(size = 13),
    legend.position = c(0.79, 0.10),
    legend.justification = c(0.5, 0),
    legend.direction = "horizontal"
  )

# Panel D
pD <- ggplot(
  test,
  aes(x = T2, y = Q, colour = hybrid_state)
) +
  geom_point(size = 2.4, alpha = 0.88) +
  geom_vline(
    xintercept = T2_limit,
    linetype = "dashed",
    linewidth = 0.5,
    colour = "#B44D3A"
  ) +
  geom_hline(
    yintercept = Q_limit,
    linetype = "dashed",
    linewidth = 0.5,
    colour = "#B44D3A"
  ) +
  scale_colour_manual(
    values = col_state,
    name = "State"
  ) +
  guides(
    colour = guide_legend(
      nrow = 1,
      override.aes = list(size = 3.2, alpha = 1)
    )
  ) +
  labs(
    title = "(D) Hybrid decision map",
    x = expression("Hotelling's " * T^2),
    y = "SPE (Q)"
  ) +
  theme_case +
  theme(
    legend.position = c(0.60, 0.22),
    legend.justification = c(0.5, 0),
    legend.direction = "horizontal"
  )

panel_case_study <- (pA + pB) / (pC + pD)
panel_case_study

############################################################
# 7) EXPORT
############################################################

ggsave(
  filename = "panel_case_study_final.pdf",
  plot = panel_case_study,
  width = 11,
  height = 8.2,
  units = "in"
)

ggsave(
  filename = "panel_case_study_final.png",
  plot = panel_case_study,
  width = 11,
  height = 8.2,
  units = "in",
  dpi = 600
)

