library(readxl)


df = read_excel("D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\data.xlsx")

#df = read.csv("D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\data.csv")
#=========================================================
# ML pipeline for translate_binary (tidymodels)
# Models: logistic (glmnet), random forest, xgboost, SVM-RBF, kNN
# 5-fold CV tuning; hold-out test set for ROC/PR curves & metrics
# =========================================================

# 1) Packages ---------------------------------------------------------------
library(tidymodels)
library(xgboost)
library(doParallel)
set.seed(42)

# 2) 数据准备 ---------------------------------------------------------------
# 确保变量类型正确
df <- df |>
  dplyr::mutate(
    TransferProb = factor(TransferProb, levels = c(0, 1), labels = c("0", "1")),
    PublicationYear = as.numeric(PublicationYear),
    Transfer = as.numeric(Transfer),
    CitedNum = as.numeric(CitedNum),
    TitleSimilarity = as.numeric(TitleSimilarity),
    AuthorCount = as.numeric(AuthorCount),   # ✅ 数值变量
    PaperType = as.factor(PaperType),
    Funding = as.factor(Funding),
    English = as.factor(English),
    Final = as.factor(Final)
  )

# 训练/测试划分
set.seed(42)
data_split <- initial_split(df, prop = 0.8, strata = TransferProb)
train_df   <- training(data_split)
test_df    <- testing(data_split)

# 交叉验证 folds
set.seed(42)
cv_folds <- vfold_cv(train_df, v = 5, strata = TransferProb)

# 3) Recipe ---------------------------------------------------------------
rec_lm <- recipe(
  Transfer ~ PublicationYear + CitedNum + PaperType + Funding +
    AuthorCount + English + Final + TitleSimilarity,
  data = train_df
) |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# 4) Metrics 集合 -----------------------------------------------------------
metrics_all <- yardstick::metric_set(
  yardstick::roc_auc,
  yardstick::pr_auc,
  yardstick::accuracy,
  yardstick::sens,
  yardstick::spec,
  yardstick::f_meas
)

# 5) 定义模型与参数网格 -----------------------------------------------------

# Logistic (glmnet)
#log_spec <- logistic_reg(penalty = tune(), mixture = tune()) |> set_engine("glmnet")
#log_grid <- grid_regular(penalty(range = c(-4, 0)), mixture(), levels = c(10, 5))

# 线性回归

lm_spec <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")



# Random Forest
rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |>
  set_engine("ranger", importance = "impurity") |> set_mode("classification")
rf_grid <- grid_regular(mtry(range = c(2, 10)), min_n(range = c(2, 20)), levels = 5)

# XGBoost
xgb_spec <- boost_tree(
  trees = 1000, learn_rate = tune(), tree_depth = tune(),
  mtry = tune(), loss_reduction = tune(), min_n = tune()
) |> set_engine("xgboost") |> set_mode("classification")
xgb_grid <- grid_latin_hypercube(
  learn_rate(range = c(1e-3, 0.3)),
  tree_depth(range = c(2L, 10L)),
  mtry(range = c(2, 10)),
  loss_reduction(),
  min_n(),
  size = 25
)

# SVM-RBF
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) |> set_engine("kernlab") |> set_mode("classification")
svm_grid <- grid_regular(cost(range = c(-3, 3)), rbf_sigma(range = c(-3, 1)), levels = 7)

# kNN
knn_spec <- nearest_neighbor(neighbors = tune(), weight_func = tune()) |>
  set_engine("kknn") |> set_mode("classification")
knn_grid <- grid_regular(
  neighbors(range = c(3L, 75L)),
  weight_func(values = c("rectangular", "triangular", "epanechnikov", "gaussian")),
  levels = c(15, 4)
)


lm_fit <- list(
  name = "linear_regression_lpm",
  final_wf = workflow() |> add_model(lm_spec) |> add_recipe(rec_lm)
)



# 6) 调参函数 ---------------------------------------------------------------
tune_one <- function(model_spec, grid, model_name){
  wf <- workflow() |> add_model(model_spec) |> add_recipe(rec)
  res <- tune_grid(
    wf,
    resamples = cv_folds,
    grid = grid,
    control = control_grid(save_pred = TRUE, verbose = FALSE),
    metrics = metrics_all
  )
  best <- select_best(res, metric = "roc_auc")
  final_wf <- finalize_workflow(wf, best)
  list(name = model_name, res = res, best = best, final_wf = final_wf)
}

#install.packages("kknn")

library(kernlab)
library(kknn)

# 7) 调参各模型 -------------------------------------------------------------
#log_fit <- tune_one(log_spec, log_grid, "logistic_glmnet")
rf_fit  <- tune_one(rf_spec,  rf_grid,  "random_forest")
xgb_fit <- tune_one(xgb_spec, xgb_grid, "xgboost")
svm_fit <- tune_one(svm_spec, svm_grid, "svm_rbf")
knn_fit <- tune_one(knn_spec, knn_grid, "knn")

models <- list(lm_fit)

# 8) 测试集评估函数 ---------------------------------------------------------
evaluate_on_test <- function(fit_obj, test_df){
  mod_name <- fit_obj$name
  final_fit <- fit_obj$final_wf |> fit(train_df)
  
  # --- 如果是回归模型（lm），走 LPM 路线 ---
  if (inherits(extract_fit_engine(final_fit), "lm")) {
    # 连续预测
    pred_num <- predict(final_fit, test_df) |>
      dplyr::bind_cols(test_df |> dplyr::select(TransferProb)) |>
      dplyr::rename(.pred = .pred)
    
    # clip 到 [0,1] 当作“概率”
    pred_num <- pred_num |>
      dplyr::mutate(
        .pred_1 = pmin(pmax(.pred, 0), 1),
        .pred_class = factor(ifelse(.pred_1 >= 0.5, "1", "0"), levels = c("0","1"))
      )
    
    m_auc  <- yardstick::roc_auc(pred_num, truth = TransferProb, .pred_1, event_level = "second")
    m_pra  <- yardstick::pr_auc(pred_num,  truth = TransferProb, .pred_1, event_level = "second")
    m_sens <- yardstick::sens(pred_num,    truth = TransferProb, .pred_class, event_level = "second")
    m_spec <- yardstick::spec(pred_num,    truth = TransferProb, .pred_class, event_level = "second")
    m_f1   <- yardstick::f_meas(pred_num,  truth = TransferProb, .pred_class, event_level = "second")
    
    metrics_tbl <- dplyr::bind_rows(m_auc, m_pra, m_sens, m_spec, m_f1) |>
      dplyr::mutate(model = mod_name)
    
    roc_df <- yardstick::roc_curve(pred_num, truth = TransferProb, .pred_1, event_level = "second") |>
      dplyr::mutate(model = mod_name)
    pr_df  <- yardstick::pr_curve(pred_num, truth = TransferProb, .pred_1, event_level = "second") |>
      dplyr::mutate(model = mod_name)
    
    return(list(name = mod_name, metrics = metrics_tbl, roc_df = roc_df, pr_df = pr_df))
  }
  
  # --- 否则按你原来的分类模型路线 ---
  preds <- predict(final_fit, test_df, type = "prob") |>
    dplyr::bind_cols(predict(final_fit, test_df, type = "class")) |>
    dplyr::bind_cols(test_df |> dplyr::select(TransferProb)) |>
    dplyr::rename(.pred_class = .pred_class)
  
  m_auc  <- yardstick::roc_auc(preds, truth = TransferProb, .pred_1, event_level = "second")
  m_pra  <- yardstick::pr_auc(preds,  truth = TransferProb, .pred_1, event_level = "second")
  m_sens <- yardstick::sens(preds,    truth = TransferProb, .pred_class, event_level = "second")
  m_spec <- yardstick::spec(preds,    truth = TransferProb, .pred_class, event_level = "second")
  m_f1   <- yardstick::f_meas(preds,  truth = TransferProb, .pred_class, event_level = "second")
  
  metrics_tbl <- dplyr::bind_rows(m_auc, m_pra, m_sens, m_spec, m_f1) |>
    dplyr::mutate(model = mod_name)
  
  roc_df <- yardstick::roc_curve(preds, truth = TransferProb, .pred_1, event_level = "second") |>
    dplyr::mutate(model = mod_name)
  pr_df <- yardstick::pr_curve(preds, truth = TransferProb, .pred_1, event_level = "second") |>
    dplyr::mutate(model = mod_name)
  
  list(name = mod_name, metrics = metrics_tbl, roc_df = roc_df, pr_df = pr_df)
}


results <- purrr::map(models, evaluate_on_test, test_df = test_df)

# 9) 汇总指标 ---------------------------------------------------------------
all_metrics <- purrr::map_dfr(results, "metrics")
print(all_metrics)

write.table(all_metrics,"D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\metrics0.csv",sep=",")

set.seed(123)

df$Transfer=as.numeric(df$Transfer)
df$TitleSimilarity=as.numeric(df$TitleSimilarity)

n <- nrow(df)
idx <- split(seq_len(n), cut(seq_len(n), 6, labels = FALSE))
dfs <- lapply(idx, function(i) df[i, ])



# 得到 df1 ~ df5
df1 <- dfs[[1]]
df2 <- dfs[[2]]
df3 <- dfs[[3]]
df4 <- dfs[[4]]
df5 <- dfs[[5]]
df6 <- dfs[[6]]

#系数确定

PublicationYearMax <- quantile(df$PublicationYear, 0.9)
CitedNumMax <- quantile(df$CitedNum, 0.9)
AuthorCountMax <- quantile(df$AuthorCount, 0.9)
TitleSimilarityMax <- quantile(df$TitleSimilarity, 0.9)

PaperType_Book.NoteMax <- 1.5*sum(df$PaperType=="Note")/1301
PaperType_Conference.ReviewMax <- 1.5*sum(df$PaperType=="Review")/1301

PaperType_Book.ChapterMax <- 1.5*sum(df$PaperType=="Book Chapter")/1301
PaperType_Conference.PaperMax <- 1.5*sum(df$PaperType=="Conference_Paper")/1301

PaperType_EditorialMax <- 1.5*sum(df$PaperType=="Editorial")/1301
PaperType_LetterMax <- 1.5*sum(df$PaperType=="Letter")/1301

PaperType_SurveyMax <- 1.5*sum(df$PaperType=="Article")/1301

#table(df$PaperType)

FundingMax <- 1.5*sum(as.numeric(df$Funding))/1301

EnglishMax <- 1.5*sum(as.numeric(df$English))/1301

FinalMax <- 1.5*sum(as.numeric(df$Final))/1301




#提取线性回归系数

final_lm_fit <- lm_fit$final_wf |> fit(train_df)

lm_engine <- extract_fit_engine(final_lm_fit)

library(broom)

coef_tbl <- tidy(lm_engine)
print(coef_tbl)

coef(lm_engine)

coef_no_intercept <- coef(lm_engine)[-1]

# 线性回归
set.seed(123)

df1$TitleSimilarity=as.numeric(df1$TitleSimilarity)

fit <- lm(Transfer ~ PublicationYear + CitedNum + AuthorCount+as.numeric(TitleSimilarity)+PaperType
          + Funding+English+Final, data = df1)
summary(fit)

#线性回归重要性

#install.packages("relaimpo")
library(relaimpo)

imp_lmg <- calc.relimp(fit, type = "lmg", rela = TRUE)
imp_lmg$lmg |> sort(decreasing = TRUE)

plot(imp_lmg, sort = TRUE)

library(dplyr)

lmg_df <- data.frame(
  variable = names(imp_lmg$lmg),
  importance = as.numeric(imp_lmg$lmg)
) |>
  arrange(desc(importance))

library(ggplot2)

p1<-ggplot(lmg_df, aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(
    x = NULL,
    y = "Relative importance (LMG)",
    title = "Linear Regression relative importance of predictors (LMG)"
  ) +
  theme_minimal(base_size = 13)




set.seed(123)
boot_imp <- boot.relimp(fit, b = 1000, type = "lmg", rank = TRUE, rela = TRUE)
booteval.relimp(boot_imp)    # 重要性CI与稳定性




# 随机森林

#install.packages("randomForest")
library(randomForest)

set.seed(123)

fit_rf <- randomForest(
  Transfer ~ PublicationYear + CitedNum + AuthorCount +
    TitleSimilarity + PaperType + Funding + English + Final,
  data = df2,
  ntree = 500,
  na.action = na.omit,
  importance = TRUE
)

fit_rf
varImpPlot(fit_rf)

imp <- importance(fit_rf)

imp_df <- data.frame(
  variable = rownames(imp),
  importance = imp[, "%IncMSE"]
)

library(ggplot2)
library(dplyr)

p2<-imp_df |>
  arrange(importance) |>
  ggplot(aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(
    x = NULL,
    y = "Increase in MSE (%)",
    title = "Random Forest Variable importance"
  ) +
  theme_classic(base_size = 13)



#XGBoost

#install.packages("xgboost")
library(xgboost)

# 删除这些变量有缺失值的行
vars <- c(
  "Transfer",
  "PublicationYear",
  "CitedNum",
  "AuthorCount",
  "TitleSimilarity",
  "PaperType",
  "Funding",
  "English",
  "Final"
)

df1_clean <- df1[
  complete.cases(df1[, vars]),
  vars
]

df6 <- df6[
  complete.cases(df6[, vars]),
  vars]

X <- model.matrix(
  Transfer ~ PublicationYear + CitedNum + AuthorCount +
    TitleSimilarity + PaperType + Funding + English + Final,
  data = df1_clean
)[, -1]   # 去掉截距

X_df6 <- model.matrix(
  Transfer ~ PublicationYear + CitedNum + AuthorCount +
    TitleSimilarity + PaperType + Funding + English + Final,
  data = df5
)[, -1]   # 去掉截距

X_df6<-as.matrix(X_df6)

X <- as.matrix(X)

setdiff(colnames(X), colnames(X_df6))
setdiff(colnames(X_df6), colnames(X))


# 使用 xgb.DMatrix 创建测试数据的 DMatrix
dtest <- xgb.DMatrix(data = X, missing = NA)

y <- df1_clean$Transfer

fit_xgb <- xgboost(
  data = X,
  label = y,
  objective = "reg:squarederror",  # 回归任务，平方误差
  nrounds = 300,                   # 迭代次数
  eta = 0.05,                      # 学习率
  max_depth = 6,                   # 树的深度
  subsample = 0.8,                 # 样本子集比例
  colsample_bytree = 0.8,          # 每棵树的特征子集比例
  verbose = 0
  
)

df6_t <- xgb.DMatrix(data = X_df6, missing = NA)


library(xgboost)

imp_xgb <- xgb.importance(
  feature_names = colnames(X),
  model = fit_xgb
)

library(dplyr)

imp_df <- imp_xgb |>
  dplyr::select(Feature, Gain) |>
  arrange(desc(Gain))



library(ggplot2)

p3<-ggplot(imp_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(
    x = NULL,
    y = "Gain (relative importance)",
    title = "XGBoost feature importance"
  ) +
  theme_classic(base_size = 14)

colnames(imp_xgb)


#SVM

# install.packages("e1071")
library(e1071)

set.seed(123)

fit_svm <- svm(
  Transfer ~ PublicationYear + CitedNum + AuthorCount +
    TitleSimilarity + PaperType + Funding + English + Final,
  data = df2,
  type = "eps-regression",
  kernel = "radial"
)

fit_svm

vars <- c("Transfer","PublicationYear","CitedNum","AuthorCount",
          "TitleSimilarity","PaperType","Funding","English","Final")

df2_svm <- df2[complete.cases(df2[, vars]), vars]

df2_svm$Transfer = as.numeric(df2_svm$Transfer)

library(e1071)
set.seed(123)

fit_svm <- svm(
  Transfer ~ PublicationYear + CitedNum + AuthorCount +
    TitleSimilarity + PaperType + Funding + English + Final,
  data = df2_svm,
  type = "eps-regression",
  kernel = "radial"
)

rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

perm_importance_svm <- function(model, data, y = "Transfer", nrep = 20, seed = 123){
  set.seed(seed)
  y_true <- data[[y]]
  base_pred <- predict(model, data)
  base_rmse <- rmse(y_true, base_pred)
  
  feats <- setdiff(names(data), y)
  out <- lapply(feats, function(f){
    inc <- replicate(nrep, {
      dperm <- data
      dperm[[f]] <- sample(dperm[[f]])   # 打乱该变量
      pred <- predict(model, dperm)
      rmse(y_true, pred) - base_rmse
    })
    data.frame(feature = f, importance = mean(inc), sd = sd(inc))
  })
  do.call(rbind, out)
}


library(dplyr)
library(ggplot2)

imp_svm <- perm_importance_svm(fit_svm, df2_svm, nrep = 30) |>
  arrange(desc(importance))

p4<-ggplot(imp_svm, aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "skyblue")+
  coord_flip() +
  labs(
    x = NULL,
    y = "Permutation importance (ΔRMSE)",
    title = "SVM (RBF) feature importance"
  ) +
  theme_classic(base_size = 14)



#knn

# install.packages("caret")
library(caret)

set.seed(123)

vars <- c("Transfer","PublicationYear","CitedNum","AuthorCount",
          "TitleSimilarity","PaperType","Funding","English","Final")

df_knn <- df1[complete.cases(df1[, vars]), vars]

library(caret)

set.seed(123)
ctrl <- trainControl(method = "cv", number = 5)

fit_knn <- train(
  Transfer ~ PublicationYear + CitedNum + AuthorCount +
    TitleSimilarity + PaperType + Funding + English + Final,
  data = df_knn,
  method = "knn",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 10
)
fit_knn


rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

perm_importance_knn <- function(model, data, y = "Transfer", nrep = 30, seed = 123){
  set.seed(seed)
  y_true <- data[[y]]
  
  # 基线 RMSE（注意：caret 会自动使用训练时的 preProcess）
  base_pred <- predict(model, newdata = data)
  base_rmse <- rmse(y_true, base_pred)
  
  feats <- setdiff(names(data), y)
  
  out <- lapply(feats, function(f){
    inc <- replicate(nrep, {
      dperm <- data
      dperm[[f]] <- sample(dperm[[f]])   # 打乱该变量
      pred <- predict(model, newdata = dperm)
      rmse(y_true, pred) - base_rmse
    })
    data.frame(feature = f, importance = mean(inc), sd = sd(inc))
  })
  
  do.call(rbind, out)
}


library(dplyr)
library(ggplot2)

imp_knn <- perm_importance_knn(fit_knn, df_knn, nrep = 50) |>
  arrange(desc(importance))

p5<-ggplot(imp_knn, aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "skyblue")+
  coord_flip() +
  labs(
    x = NULL,
    y = "Permutation importance (ΔRMSE)",
    title = "kNN feature importance"
  ) +
  theme_classic(base_size = 14)


library(patchwork)

(p1+p2)/(p3+p4+p5)


ctrl <- trainControl(method = "cv", number = 5)

fit_knn <- train(
  Transfer ~ PublicationYear + CitedNum + AuthorCount +
    TitleSimilarity + PaperType + Funding + English + Final,
  data = df2,
  method = "knn",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 10,
  na.action = na.omit
)

fit_knn

vars <- c("Transfer","PublicationYear","CitedNum","AuthorCount",
          "TitleSimilarity","PaperType","Funding","English","Final")

df_knn <- df1[complete.cases(df1[, vars]), vars]

library(caret)

set.seed(123)
ctrl <- trainControl(method = "cv", number = 5)

fit_knn <- train(
  Transfer ~ PublicationYear + CitedNum + AuthorCount +
    TitleSimilarity + PaperType + Funding + English + Final,
  data = df_knn,
  method = "knn",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 10
)
fit_knn


rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

perm_importance_knn <- function(model, data, y = "Transfer", nrep = 30, seed = 123){
  set.seed(seed)
  y_true <- data[[y]]
  
  # 基线 RMSE（注意：caret 会自动使用训练时的 preProcess）
  base_pred <- predict(model, newdata = data)
  base_rmse <- rmse(y_true, base_pred)
  
  feats <- setdiff(names(data), y)
  
  out <- lapply(feats, function(f){
    inc <- replicate(nrep, {
      dperm <- data
      dperm[[f]] <- sample(dperm[[f]])   # 打乱该变量
      pred <- predict(model, newdata = dperm)
      rmse(y_true, pred) - base_rmse
    })
    data.frame(feature = f, importance = mean(inc), sd = sd(inc))
  })
  
  do.call(rbind, out)
}


library(dplyr)
library(ggplot2)

imp_knn <- perm_importance_knn(fit_knn, df_knn, nrep = 50) |>
  arrange(desc(importance))

ggplot(imp_knn, aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "grey30") +
  geom_errorbar(aes(ymin = importance - sd, ymax = importance + sd), width = 0.2) +
  coord_flip() +
  labs(
    x = NULL,
    y = "Permutation importance (ΔRMSE)",
    title = "kNN feature importance"
  ) +
  theme_classic(base_size = 14)


setdiff(colnames(df1), colnames(df6_t))
setdiff(colnames(df6_t), colnames(df1))



#计算RMSE

# 真实值
y_true <- as.numeric(df6$Transfer)

y_true1<-df5$Transfer

# 各模型预测
pred_lm  <- predict(fit, df6)
pred_rf  <- predict(fit_rf, df6)
pred_xgb <- predict(fit_xgb, df6_t)      # X 为 model.matrix
pred_svm <- predict(fit_svm, df6)
pred_knn <- predict(fit_knn, df6)

rmse_lm  <- sqrt(mean((y_true - pred_lm)^2,na.rm = T))
rmse_lm
rmse_rf  <- sqrt(mean((y_true - pred_rf)^2,na.rm = T))
rmse_rf
rmse_xgb <- sqrt(mean((y_true1 - pred_xgb)^2,na.rm = T))
rmse_xgb
rmse_svm <- sqrt(mean((y_true - pred_svm)^2,na.rm = T))
rmse_svm
rmse_knn <- sqrt(mean((y_true - pred_knn)^2,na.rm = T))
rmse_knn

#计算R方

library(yardstick)

rsq_lm <- rsq_vec(y_true, pred_lm)
rsq_lm

rsq_rf <- rsq_vec(y_true, pred_rf)
rsq_rf

rsq_xgb <- rsq_vec(y_true1, pred_xgb)
rsq_xgb

rsq_svm <- rsq_vec(y_true, pred_svm)
rsq_svm

rsq_knn <- rsq_vec(y_true, pred_knn)
rsq_knn

# (2) 计算MAE
mae_lm <- mean(abs(y_true - pred_lm))
mae_rf <- mean(abs(y_true - pred_rf))
mae_xgb <- mean(abs(y_true1 - pred_xgb))
mae_svm <- mean(abs(y_true - pred_svm))
mae_knn <- mean(abs(y_true - pred_knn))

mae_lm
mae_rf
mae_xgb
mae_svm
mae_knn


coef(fit)

coef_no_intercept <- coef(fit)[-1]

#install.packages("lpSolve")
library(lpSolve)

#coef_no_intercept <- coef_no_intercept / sum(abs(coef_no_intercept))
n <- ncol(A)

# 哪些变量要是 0/1
bin_idx <- 5:n

lower <- rep(0, n)
upper <- rep(Inf, n)   # 前4个可以连续、无上界（建议你也给上界，避免无界）
upper[bin_idx] <- 1    # 二元变量上界=1

int_vec <- bin_idx



# 目标函数系数
obj <- coef_no_intercept

A <- rbind(
  c(0, 1,1,0,0,0,0,0,0,0,0,0,0,0),
  c(1, 0,0,0,0,0,0,0,0,0,0,0,0,0),
  c(0,0,0,1,0,0,0,0,0,0,0,0,0,0),
  c(0,0,0,0,1,1,1,1,1,1,1,0,0,0),
  c(0,0,0,0,0,0,0,0,0,0,0,1,1,1)
  
)

# 约束方向与右端项
dir <- c("<=","<=","<=","<=","<=")
rhs <- c(AuthorCountMax+CitedNumMax, PublicationYearMax,TitleSimilarityMax, 
         
         PaperType_Book.NoteMax+PaperType_Conference.ReviewMax+
         
         PaperType_Book.ChapterMax+
         PaperType_Conference.PaperMax+
         
         PaperType_EditorialMax+
         PaperType_LetterMax+
         
         PaperType_SurveyMax,
         
         FundingMax+EnglishMax+FinalMax)

res <- lp(
  direction = "max",
  objective.in = obj,      # 你的目标函数系数向量，长度必须=14
  const.mat = A,
  const.dir = dir,
  const.rhs = rhs,
  int.vec = int_vec,       # 只让第5~14个变量为整数
  all.int = FALSE,
  binary.vec = int_vec,    # 如果你的 lpSolve 版本支持这个参数，会更明确
)

res$status   # 0=最优；2=无可行解；3=无界
res$objval+as.numeric(coef(fit)[1])
res$solution

#df1:0.4897, df2:0.2454, df3:0.3832, df4: 0.2863, df5: 0.2636

#计算linear regression RMSE

y_true <- df6$Transfer

y_pred <- predict(fit, df6)

rmse <- sqrt(mean((y_true - y_pred)^2,na.rm = T))

# 迭代
obj <- c(2023,884,0,0.461,0,1,0,0,0,0,0,0,0,0)

A <- rbind(
  c(2023,0,0,0,0,0,0,0,0,0,0,0,0,0),
  c(0,884,0,0,0,0,0,0,0,0,0,0,0,0),
  c(0,0,0,0.461,0,0,0,0,0,0,0,0,0,0),
  c(0,0,0,0,0,1,0,0,0,0,0,0,0,0)
)

coef(fit)

dir <- c("<=","<=","<=","<=")
rhs <- c(PublicationYearMax*0.00206,CitedNumMax*0.000058707, 
         TitleSimilarityMax*0.16875,PaperType_Conference.PaperMax*0.021906)

res <- lp(
  direction = "max",
  objective.in = obj,      # 你的目标函数系数向量，长度必须=14
  const.mat = A,
  const.dir = dir,
  const.rhs = rhs,
  #int.vec = int_vec,       # 只让第5~14个变量为整数
  all.int = FALSE,
  #binary.vec = int_vec,    # 如果你的 lpSolve 版本支持这个参数，会更明确
)

res$status   # 0=最优；2=无可行解；3=无界
res$objval-3.975
res$solution





#### 云浩不用看以下内容


# 10) 绘制 ROC 和 PR 曲线 ---------------------------------------------------
roc_all <- purrr::map_dfr(results, "roc_df")
pr_all  <- purrr::map_dfr(results, "pr_df")

p_roc <- ggplot(roc_all, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = 2) +
  coord_equal() +
  labs(title = "ROC Curves on Test Set", x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)")+
  theme_minimal()

p_pr <- ggplot(pr_all, aes(x = recall, y = precision, color = model)) +
  geom_path(linewidth = 1) +
  labs(title = "Precision-Recall Curves on Test Set", x = "Recall", y = "Precision")+
  theme_minimal()

print(p_roc)
print(p_pr)

p_roc+p_pr


#rf 
library(patchwork)
# =========================================================
# 11) 变量重要性图（使用随机森林模型）
# =========================================================
library(vip)    # 可视化变量重要性
library(ggplot2)

# 用最佳参数拟合随机森林
rf_final_fit <- log_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p1=vip(rf_final_fit, num_features = 20, aesthetics = list(fill = "aliceblue")) +
  ggtitle("Logistic") +
  theme_minimal() +
  coord_flip() 


# 用最佳参数拟合随机森林
rf_final_fit <- rf_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p2=vip(rf_final_fit, num_features = 20, aesthetics = list(fill = "blue")) +
  ggtitle("Random Forest") +
  theme_minimal() +
  coord_flip()

rf_final_fit <- xgb_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p3=vip(rf_final_fit, num_features = 20, aesthetics = list(fill = "lightblue")) +
  ggtitle("XGBoost") +
  theme_minimal() +
  coord_flip()

rf_final_fit <- svm_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p4=vip(rf_final_fit, num_features = 20) +
  ggtitle("Support Vector Machine") +
  theme_minimal() +
  coord_flip()

rf_final_fit <- knn_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p5=vip(rf_final_fit, num_features = 20) +
  ggtitle("KNN") +
  theme_minimal() +
  coord_flip()

(p1+p2+p3)




# =========================================================
# 12) 主成分分析 PCA
# =========================================================
# 从 recipe 里准备处理过的数据矩阵
prep_rec <- prep(rec, training = train_df)
train_processed <- bake(prep_rec, new_data = train_df)

# 去掉目标变量
X <- train_processed |> dplyr::select(-TransferProb)

# PCA
pca_res <- prcomp(X, scale. = TRUE)

# 前两个主成分的解释度
summary(pca_res)

# 把 PCA 得分和类别拼在一起
pca_df <- as.data.frame(pca_res$x[, 1:2]) |>
  dplyr::mutate(TransferProb = train_processed$TransferProb)

# 画 PCA 二维图
ggplot(pca_df, aes(x = PC1, y = PC2, color = TransferProb)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(title = "PCA Plot (First Two Components)") +
  theme_minimal()

# 查看前两个主成分的载荷
loadings <- pca_res$rotation[, 1:2]

# 按绝对值排序，找出贡献最大的变量
pc1_top <- sort(abs(loadings[, "PC1"]), decreasing = TRUE)
pc2_top <- sort(abs(loadings[, "PC2"]), decreasing = TRUE)

cat("PC1主要由以下变量贡献:\n")
print(head(pc1_top, 5))

cat("PC2主要由以下变量贡献:\n")
print(head(pc2_top, 5))

library(ggplot2)

loadings_df <- as.data.frame(loadings) |>
  tibble::rownames_to_column("variable")

ggplot(loadings_df, aes(x = PC1, y = PC2, label = variable)) +
  geom_point(color = "orange",size=5) +
  geom_text(vjust = 1.5, size = 3) +
  labs(title = "PCA Loadings Plot",
       x = "PC1 Loadings",
       y = "PC2 Loadings") +
  theme_minimal()



library(ggplot2)
library(dplyr)
library(tidyr)

# 1. PublicationYear 柱状图
p1=ggplot(df, aes(x = PublicationYear)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "white") +
  labs(title = "Distribution of PublicationYear", x = "PublicationYear", y = "Count") +
  theme_minimal()

# 2. CitedNum 柱状图
p2=ggplot(df, aes(x = CitedNum)) +
  geom_histogram(binwidth = 5, fill = "orange", color = "white") +
  labs(title = "Distribution of CitedNum", x = "CitedNum", y = "Count") +
  theme_minimal() +
  xlim(0, 200)+
  ylim(0,250)

# 3. PaperType 饼图
p3=df %>%
  count(PaperType) %>%
  ggplot(aes(x = "", y = n, fill = PaperType)) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Distribution of PaperType") +
  theme_void()

# 4. Transfer 密度分布图

df$Transfer = as.numeric(df$Transfer)

p4=ggplot(df, aes(x = Transfer)) +
  geom_density(fill = "red", alpha = 0.5) +
  labs(title = "Density of Transfer", x = "Transfer", y = "Density") +
  theme_minimal()

# 5. AuthorCount 柱状图
p5=ggplot(df, aes(x = AuthorCount)) +
  geom_histogram(binwidth = 1, fill = "purple", color = "white") +
  labs(title = "Distribution of AuthorCount", x = "AuthorCount", y = "Count") +
  theme_minimal()

# 6. Funding, English, Final, TransferProb 堆叠柱状图
p6=df %>%
  select(Funding, English, Final, TransferProb) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable, Value) %>%
  summarise(n = n(), .groups = "drop") %>%
  ggplot(aes(x = Variable, y = n, fill = as.factor(Value))) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "darkblue")) +
  labs(title = "Stacked Bar Chart of Funding, English, Final, TransferProb",
       x = "Variable", y = "Count", fill = "Value") +
  theme_minimal()


(p1+p2)/(p3+p4)/(p5+p6)


#Summary Table

library(dplyr)
library(tidyr)
library(readr)

# 1. 连续变量 summary
numeric_summary <- df %>%
  select(PublicationYear, CitedNum, TitleSimilarity, AuthorCount) %>%
  summarise(across(everything(),
                   list(mean = ~mean(. , na.rm = TRUE),
                        sd   = ~sd(. , na.rm = TRUE)))) %>%
  pivot_longer(
    everything(),
    names_to = c("Variable", ".value"),
    names_sep = "_"
  ) %>%
  mutate(
    Category = NA_character_,
    Count = NA_integer_,
    Percent = NA_real_
  ) %>%
  rename(Mean = mean, SD = sd)

# 2. 分类变量 summary
categorical_summary <- df %>%
  select(TransferProb, PaperType, Funding, English, Final) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Category") %>%
  group_by(Variable, Category) %>%
  summarise(Count = n(), .groups = "drop_last") %>%
  mutate(
    Percent = round(100 * Count / sum(Count), 1),
    Mean = NA_real_,
    SD = NA_real_
  )

# 3. 统一列顺序
numeric_summary <- numeric_summary %>%
  select(Variable, Category, Mean, SD, Count, Percent)

categorical_summary <- categorical_summary %>%
  select(Variable, Category, Mean, SD, Count, Percent)

# 4. 合并
summary_table <- bind_rows(numeric_summary, categorical_summary)

# 5. 导出 CSV
write_csv(summary_table, "D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\summary_table.csv")

print("✅ summary_table.csv 已生成")


# 计算最优化问题的系数

mean(df0$Funding)*1301

# 1130

median(df0$AuthorCount)

# 3903

as.numeric(median(df0$TitleSimilarity))*1301


df0$
# 

# 第二个数据库

# commercialized
library(readr)
data <- read_csv("data.csv")


data_second = read.csv2("D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\11461587\\_patent_paper_pairs_plus.csv",sep=",",encoding="gb18030")

# 训练/测试划分rm(list = ls())

df = data_second

library(dplyr)

df <- df %>%
  filter(grepl("Medicine", concepts, ignore.case = TRUE))

df = subset(df,df$concepts=="Medicine")

mean(df$patentuniv)*15955

mean(df$papergovn)*15955

mean(df$paperfirm)*15955

df$commercialized = as.factor(df$commercialized)

set.seed(42)
data_split <- initial_split(df, prop = 0.8, strata = commercialized)
train_df   <- training(data_split)
test_df    <- testing(data_split)

# 交叉验证 folds
set.seed(42)
cv_folds <- vfold_cv(train_df, v = 5, strata = commercialized)

rec <- recipe(
  commercialized ~ ppp_score+daysdiffcont+citationbased+paperuniv+papergovn+paperfirm+paperunk+patentuniv
  +patentgovn+patentlone,
  data = train_df
) |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# 4) Metrics 集合 -----------------------------------------------------------
metrics_all <- yardstick::metric_set(
  yardstick::roc_auc,
  yardstick::pr_auc,
  yardstick::accuracy,
  yardstick::sens,
  yardstick::spec,
  yardstick::f_meas
)

# 5) 定义模型与参数网格 -----------------------------------------------------

# Logistic (glmnet)
log_spec <- logistic_reg(penalty = tune(), mixture = tune()) |> set_engine("glmnet")
log_grid <- grid_regular(penalty(range = c(-4, 0)), mixture(), levels = c(10, 5))

# Random Forest
rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |>
  set_engine("ranger", importance = "impurity") |> set_mode("classification")
rf_grid <- grid_regular(mtry(range = c(2, 10)), min_n(range = c(2, 20)), levels = 5)

# XGBoost
xgb_spec <- boost_tree(
  trees = 1000, learn_rate = tune(), tree_depth = tune(),
  mtry = tune(), loss_reduction = tune(), min_n = tune()
) |> set_engine("xgboost") |> set_mode("classification")
xgb_grid <- grid_latin_hypercube(
  learn_rate(range = c(1e-3, 0.3)),
  tree_depth(range = c(2L, 10L)),
  mtry(range = c(2, 10)),
  loss_reduction(),
  min_n(),
  size = 25
)

# SVM-RBF
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) |> set_engine("kernlab") |> set_mode("classification")
svm_grid <- grid_regular(cost(range = c(-3, 3)), rbf_sigma(range = c(-3, 1)), levels = 7)

# kNN
knn_spec <- nearest_neighbor(neighbors = tune(), weight_func = tune()) |>
  set_engine("kknn") |> set_mode("classification")
knn_grid <- grid_regular(
  neighbors(range = c(3L, 75L)),
  weight_func(values = c("rectangular", "triangular", "epanechnikov", "gaussian")),
  levels = c(15, 4)
)

# 6) 调参函数 ---------------------------------------------------------------
tune_one <- function(model_spec, grid, model_name){
  wf <- workflow() |> add_model(model_spec) |> add_recipe(rec)
  res <- tune_grid(
    wf,
    resamples = cv_folds,
    grid = grid,
    control = control_grid(save_pred = TRUE, verbose = FALSE),
    metrics = metrics_all
  )
  best <- select_best(res, metric = "roc_auc")
  final_wf <- finalize_workflow(wf, best)
  list(name = model_name, res = res, best = best, final_wf = final_wf)
}

#install.packages("kknn")

library(kernlab)
library(kknn)

# 7) 调参各模型 -------------------------------------------------------------
log_fit <- tune_one(log_spec, log_grid, "logistic_glmnet")
rf_fit  <- tune_one(rf_spec,  rf_grid,  "random_forest")
xgb_fit <- tune_one(xgb_spec, xgb_grid, "xgboost")
svm_fit <- tune_one(svm_spec, svm_grid, "svm_rbf")
knn_fit <- tune_one(knn_spec, knn_grid, "knn")

models <- list(log_fit, rf_fit, xgb_fit, svm_fit, knn_fit)

# 8) 测试集评估函数 ---------------------------------------------------------
evaluate_on_test <- function(fit_obj, test_df){
  mod_name <- fit_obj$name
  
  # 🔧 1. 在训练集上拟合最终模型
  final_fit <- fit_obj$final_wf |> fit(train_df)
  
  # 🔧 2. 预测概率（注意：不再直接用 type="class"）
  preds <- predict(final_fit, test_df, type = "prob") %>%
    dplyr::bind_cols(test_df %>% dplyr::select(commercialized))
  
  # ✅ 3. 计算 ROC 曲线，寻找最佳 cutoff
  roc_data <- yardstick::roc_curve(preds, truth = commercialized, .pred_1, event_level = "second")
  
  # 避免 NA 值（有时模型未覆盖所有阈值）
  roc_data <- roc_data[!is.na(roc_data$sensitivity) & !is.na(roc_data$specificity), ]
  
  best_cutoff <- roc_data$.threshold[which.max(roc_data$sensitivity + roc_data$specificity - 1)]
  cat("模型:", mod_name, "| 最佳阈值 =", round(best_cutoff, 4), "\n")
  
  # ✅ 4. 重新生成类别预测（使用最佳阈值）
  preds <- preds %>%
    mutate(
      .pred_class = ifelse(.pred_1 >= best_cutoff, "1", "0"),
      .pred_class = factor(.pred_class, levels = levels(test_df$commercialized))  # 保持因子一致
    )
  
  # ✅ 5. 计算各项指标
  m_auc  <- yardstick::roc_auc(preds, truth = commercialized, .pred_1, event_level = "second")
  m_pra  <- yardstick::pr_auc(preds,  truth = commercialized, .pred_1, event_level = "second")
  m_sens <- yardstick::sens(preds,    truth = commercialized, .pred_class, event_level = "second")
  m_spec <- yardstick::spec(preds,    truth = commercialized, .pred_class, event_level = "second")
  m_f1   <- yardstick::f_meas(preds,  truth = commercialized, .pred_class, event_level = "second")
  
  # ✅ 6. 汇总结果
  metrics_tbl <- dplyr::bind_rows(m_auc, m_pra, m_sens, m_spec, m_f1) %>%
    dplyr::mutate(model = mod_name)
  
  roc_df <- roc_data %>% dplyr::mutate(model = mod_name)
  pr_df  <- yardstick::pr_curve(preds, truth = commercialized, .pred_1, event_level = "second") %>%
    dplyr::mutate(model = mod_name)
  
  # ✅ 7. 输出（包括 metrics + ROC + PR）
  list(name = mod_name, metrics = metrics_tbl, roc_df = roc_df, pr_df = pr_df)
}


results <- purrr::map(models, evaluate_on_test, test_df = test_df)

# 9) 汇总指标 ---------------------------------------------------------------
all_metrics <- purrr::map_dfr(results, "metrics")
print(all_metrics)

write.table(all_metrics,"D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\metrics_2.csv",sep=",")


table(train_df$commercialized)
prop.table(table(train_df$commercialized))

# =========================================================
# 12) 主成分分析 PCA
# =========================================================
# 从 recipe 里准备处理过的数据矩阵
prep_rec <- prep(rec, training = train_df)
train_processed <- bake(prep_rec, new_data = train_df)

# 去掉目标变量
X <- train_processed |> dplyr::select(-commercialized)

# PCA
pca_res <- prcomp(X, scale. = TRUE)

# 前两个主成分的解释度
summary(pca_res)

# 把 PCA 得分和类别拼在一起
pca_df <- as.data.frame(pca_res$x[, 1:2]) |>
  dplyr::mutate(commercialized = train_processed$commercialized)

# 画 PCA 二维图
ggplot(pca_df, aes(x = PC1, y = PC2, color = commercialized)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(title = "PCA Plot (First Two Components)") +
  theme_minimal()

# 查看前两个主成分的载荷
loadings <- pca_res$rotation[, 1:2]

# 按绝对值排序，找出贡献最大的变量
pc1_top <- sort(abs(loadings[, "PC1"]), decreasing = TRUE)
pc2_top <- sort(abs(loadings[, "PC2"]), decreasing = TRUE)

cat("PC1主要由以下变量贡献:\n")
print(head(pc1_top, 5))

cat("PC2主要由以下变量贡献:\n")
print(head(pc2_top, 5))

library(ggplot2)

loadings_df <- as.data.frame(loadings) |>
  tibble::rownames_to_column("variable")

ggplot(loadings_df, aes(x = PC1, y = PC2, label = variable)) +
  geom_point(color = "orange",size=5) +
  geom_text(vjust = 1.5, size = 3) +
  labs(title = "PCA Loadings Plot",
       x = "PC1 Loadings",
       y = "PC2 Loadings") +
  theme_minimal()


#Summary Table

library(dplyr)
library(tidyr)
library(readr)

# 1. 连续变量 summary
numeric_summary <- df %>%
  select(daysdiffcont) %>%
  summarise(across(everything(),
                   list(mean = ~mean(. , na.rm = TRUE),
                        sd   = ~sd(. , na.rm = TRUE)))) %>%
  pivot_longer(
    everything(),
    names_to = c("Variable", ".value"),
    names_sep = "_"
  ) %>%
  mutate(
    Category = NA_character_,
    Count = NA_integer_,
    Percent = NA_real_
  ) %>%
  rename(Mean = mean, SD = sd)

df$commercialized = as.numeric(df$commercialized)

# 2. 分类变量 summary
categorical_summary <- df %>%
  select(ppp_score, citationbased, paperuniv, papergovn, paperfirm, paperunk,patentgovn,patentuniv,patentfirm,patentgovn,patentlone,commercialized,commercialized,prerecommercialized) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Category") %>%
  group_by(Variable, Category) %>%
  summarise(Count = n(), .groups = "drop_last") %>%
  mutate(
    Percent = round(100 * Count / sum(Count), 1),
    Mean = NA_real_,
    SD = NA_real_
  )

# 3. 统一列顺序
numeric_summary <- numeric_summary %>%
  select(Variable, Category, Mean, SD, Count, Percent)

categorical_summary <- categorical_summary %>%
  select(Variable, Category, Mean, SD, Count, Percent)

# 4. 合并
summary_table <- bind_rows(numeric_summary, categorical_summary)

# 5. 导出 CSV
#write_csv(summary_table, "D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\summary_table.csv")

write.csv(numeric_summary, "D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\numeric_summary.csv", row.names = FALSE)
write.csv(categorical_summary, "D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\categorical_summary.csv", row.names = FALSE)

print("✅ summary_table.csv 已生成")

# 6. Funding, English, Final, TransferProb 堆叠柱状图

df$commercialized <- ifelse(df$commercialized == 1, 0, 1)

p6=df %>%
  select(citationbased,paperuniv,papergovn,paperfirm,paperunk,patentuniv
         ,patentfirm,patentgovn,patentlone,commercialized,commercialized,prerecommercialized) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable, Value) %>%
  summarise(n = n(), .groups = "drop") %>%
  ggplot(aes(x = Variable, y = n, fill = as.factor(Value))) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "darkblue")) +
  labs(title = "Stacked Bar Chart of Category Variables",
       x = "Category Variables", y = "Count", fill = "Value") +
  theme_minimal()

p1=ggplot(df, aes(x = ppp_score)) +
  geom_histogram(binwidth = 1, fill = "lightgreen", color = "white") +
  labs(title = "Distribution of PPP Score", x = "PPP Score", y = "Count") +
  theme_minimal()

# 2. CitedNum 柱状图
p2=ggplot(df, aes(x = daysdiffcont)) +
  geom_density(binwidth = 5, fill = "orange", color = "orange") +
  labs(title = "Distribution of DaysDiffCont", x = "DaysDiffCont", y = "Density") +
  theme_minimal()

install.packages("patchwork")

library(patchwork)

(p1+p2)/p6


# AUC Plot

# 10) 绘制 ROC 和 PR 曲线 ---------------------------------------------------
roc_all <- purrr::map_dfr(results, "roc_df")
#pr_all  <- purrr::map_dfr(results, "pr_df")

p_roc <- ggplot(roc_all, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = 2) +
  coord_equal() +
  labs(title = "ROC Curves on Test Set", x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)")+
  theme_minimal()

p_roc


library(patchwork)
# =========================================================
# 11) 变量重要性图（使用随机森林模型）
# =========================================================
library(vip)    # 可视化变量重要性
library(ggplot2)

train_df$reassigned = as.factor(train_df$reassigned)

# 用最佳参数拟合随机森林
rf_final_fit <- log_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p1=vip(rf_final_fit, num_features = 20, aesthetics = list(fill = "aliceblue")) +
  ggtitle("Logistic") +
  theme_minimal() +
  coord_flip() 


# 用最佳参数拟合随机森林
rf_final_fit <- rf_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p2=vip(rf_final_fit, num_features = 20, aesthetics = list(fill = "blue")) +
  ggtitle("Random Forest") +
  theme_minimal() +
  coord_flip()

rf_final_fit <- xgb_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p3=vip(rf_final_fit, num_features = 20, aesthetics = list(fill = "lightblue")) +
  ggtitle("XGBoost") +
  theme_minimal() +
  coord_flip()

rf_final_fit <- svm_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p4=vip(rf_final_fit, num_features = 20) +
  ggtitle("Support Vector Machine") +
  theme_minimal() +
  coord_flip()

rf_final_fit <- knn_fit$final_wf |> fit(train_df)

# 提取并绘制变量重要性（横向柱状图）
p5=vip(rf_final_fit, num_features = 20) +
  ggtitle("KNN") +
  theme_minimal() +
  coord_flip()

(p1+p2+p3)







# 第三个数据库 commercialized

data_second = read.csv2("D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\11461587\\_patent_paper_pairs_plus.csv",sep=",",encoding="gb18030")

# 训练/测试划分
df = data_second



library(dplyr)

df <- df %>%
  filter(grepl("Medicine", concepts, ignore.case = TRUE))

df = subset(df,df$concepts=="Medicine")


df$commercialized = as.factor(df$commercialized)

set.seed(42)
data_split <- initial_split(df, prop = 0.8, strata = commercialized)
train_df   <- training(data_split)
test_df    <- testing(data_split)

# 交叉验证 folds
set.seed(42)
cv_folds <- vfold_cv(train_df, v = 5, strata = commercialized)

rec <- recipe(
  commercialized ~ ppp_score+daysdiffcont+citationbased+paperuniv+papergovn+paperfirm+paperunk+patentuniv
  +patentfirm+patentgovn+patentlone,
  data = train_df
) |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# 4) Metrics 集合 -----------------------------------------------------------
metrics_all <- yardstick::metric_set(
  yardstick::roc_auc,
  yardstick::pr_auc,
  yardstick::accuracy,
  yardstick::sens,
  yardstick::spec,
  yardstick::f_meas
)

# 5) 定义模型与参数网格 -----------------------------------------------------

# Logistic (glmnet)
log_spec <- logistic_reg(penalty = tune(), mixture = tune()) |> set_engine("glmnet")
log_grid <- grid_regular(penalty(range = c(-4, 0)), mixture(), levels = c(10, 5))

# Random Forest
rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |>
  set_engine("ranger", importance = "impurity") |> set_mode("classification")
rf_grid <- grid_regular(mtry(range = c(2, 10)), min_n(range = c(2, 20)), levels = 5)

# XGBoost
xgb_spec <- boost_tree(
  trees = 1000, learn_rate = tune(), tree_depth = tune(),
  mtry = tune(), loss_reduction = tune(), min_n = tune()
) |> set_engine("xgboost") |> set_mode("classification")
xgb_grid <- grid_latin_hypercube(
  learn_rate(range = c(1e-3, 0.3)),
  tree_depth(range = c(2L, 10L)),
  mtry(range = c(2, 10)),
  loss_reduction(),
  min_n(),
  size = 25
)

# SVM-RBF
svm_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) |> set_engine("kernlab") |> set_mode("classification")
svm_grid <- grid_regular(cost(range = c(-3, 3)), rbf_sigma(range = c(-3, 1)), levels = 7)

# kNN
knn_spec <- nearest_neighbor(neighbors = tune(), weight_func = tune()) |>
  set_engine("kknn") |> set_mode("classification")
knn_grid <- grid_regular(
  neighbors(range = c(3L, 75L)),
  weight_func(values = c("rectangular", "triangular", "epanechnikov", "gaussian")),
  levels = c(15, 4)
)

# 6) 调参函数 ---------------------------------------------------------------
tune_one <- function(model_spec, grid, model_name){
  wf <- workflow() |> add_model(model_spec) |> add_recipe(rec)
  res <- tune_grid(
    wf,
    resamples = cv_folds,
    grid = grid,
    control = control_grid(save_pred = TRUE, verbose = FALSE),
    metrics = metrics_all
  )
  best <- select_best(res, metric = "roc_auc")
  final_wf <- finalize_workflow(wf, best)
  list(name = model_name, res = res, best = best, final_wf = final_wf)
}

#install.packages("kknn")

library(kernlab)
library(kknn)

# 7) 调参各模型 -------------------------------------------------------------
log_fit_c <- tune_one(log_spec, log_grid, "logistic_glmnet")
rf_fit_c  <- tune_one(rf_spec,  rf_grid,  "random_forest")
xgb_fit_c <- tune_one(xgb_spec, xgb_grid, "xgboost")
svm_fit_c <- tune_one(svm_spec, svm_grid, "svm_rbf")
knn_fit_c <- tune_one(knn_spec, knn_grid, "knn")

models <- list(log_fit_c, rf_fit_c, xgb_fit_c, svm_fit_c, knn_fit_c)

# 8) 测试集评估函数 ---------------------------------------------------------
evaluate_on_test <- function(fit_obj, test_df){
  mod_name <- fit_obj$name
  
  # 🔧 1. 在训练集上拟合最终模型
  final_fit <- fit_obj$final_wf |> fit(train_df)
  
  # 🔧 2. 预测概率（注意：不再直接用 type="class"）
  preds <- predict(final_fit, test_df, type = "prob") %>%
    dplyr::bind_cols(test_df %>% dplyr::select(commercialized))
  
  # ✅ 3. 计算 ROC 曲线，寻找最佳 cutoff
  roc_data <- yardstick::roc_curve(preds, truth = commercialized, .pred_1, event_level = "second")
  
  # 避免 NA 值（有时模型未覆盖所有阈值）
  roc_data <- roc_data[!is.na(roc_data$sensitivity) & !is.na(roc_data$specificity), ]
  
  best_cutoff <- roc_data$.threshold[which.max(roc_data$sensitivity + roc_data$specificity - 1)]
  cat("模型:", mod_name, "| 最佳阈值 =", round(best_cutoff, 4), "\n")
  
  # ✅ 4. 重新生成类别预测（使用最佳阈值）
  preds <- preds %>%
    mutate(
      .pred_class = ifelse(.pred_1 >= best_cutoff, "1", "0"),
      .pred_class = factor(.pred_class, levels = levels(test_df$commercialized))  # 保持因子一致
    )
  
  # ✅ 5. 计算各项指标
  m_auc  <- yardstick::roc_auc(preds, truth = commercialized, .pred_1, event_level = "second")
  m_pra  <- yardstick::pr_auc(preds,  truth = commercialized, .pred_1, event_level = "second")
  m_sens <- yardstick::sens(preds,    truth = commercialized, .pred_class, event_level = "second")
  m_spec <- yardstick::spec(preds,    truth = commercialized, .pred_class, event_level = "second")
  m_f1   <- yardstick::f_meas(preds,  truth = commercialized, .pred_class, event_level = "second")
  
  # ✅ 6. 汇总结果
  metrics_tbl <- dplyr::bind_rows(m_auc, m_pra, m_sens, m_spec, m_f1) %>%
    dplyr::mutate(model = mod_name)
  
  roc_df <- roc_data %>% dplyr::mutate(model = mod_name)
  pr_df  <- yardstick::pr_curve(preds, truth = commercialized, .pred_1, event_level = "second") %>%
    dplyr::mutate(model = mod_name)
  
  # ✅ 7. 输出（包括 metrics + ROC + PR）
  list(name = mod_name, metrics = metrics_tbl, roc_df = roc_df, pr_df = pr_df)
}


results <- purrr::map(models, evaluate_on_test, test_df = test_df)

# 9) 汇总指标 ---------------------------------------------------------------
all_metrics <- purrr::map_dfr(results, "metrics")
print(all_metrics)

write.table(all_metrics,"D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\metrics_2.csv",sep=",")


table(train_df$commercialized)
prop.table(table(train_df$commercialized))

# =========================================================
# 12) 主成分分析 PCA
# =========================================================
# 从 recipe 里准备处理过的数据矩阵
prep_rec <- prep(rec, training = train_df)
train_processed <- bake(prep_rec, new_data = train_df)

# 去掉目标变量
X <- train_processed |> dplyr::select(-commercialized)

# PCA
pca_res <- prcomp(X, scale. = TRUE)

# 前两个主成分的解释度
summary(pca_res)

# 把 PCA 得分和类别拼在一起
pca_df <- as.data.frame(pca_res$x[, 1:2]) |>
  dplyr::mutate(commercialized = train_processed$commercialized)

# 画 PCA 二维图
ggplot(pca_df, aes(x = PC1, y = PC2, color = commercialized)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(title = "PCA Plot (First Two Components)") +
  theme_minimal()

# 查看前两个主成分的载荷
loadings <- pca_res$rotation[, 1:2]

# 按绝对值排序，找出贡献最大的变量
pc1_top <- sort(abs(loadings[, "PC1"]), decreasing = TRUE)
pc2_top <- sort(abs(loadings[, "PC2"]), decreasing = TRUE)

cat("PC1主要由以下变量贡献:\n")
print(head(pc1_top, 5))

cat("PC2主要由以下变量贡献:\n")
print(head(pc2_top, 5))

library(ggplot2)

loadings_df <- as.data.frame(loadings) |>
  tibble::rownames_to_column("variable")

ggplot(loadings_df, aes(x = PC1, y = PC2, label = variable)) +
  geom_point(color = "orange",size=5) +
  geom_text(vjust = 1.5, size = 3) +
  labs(title = "PCA Loadings Plot",
       x = "PC1 Loadings",
       y = "PC2 Loadings") +
  theme_minimal()


#Summary Table

library(dplyr)
library(tidyr)
library(readr)

# 1. 连续变量 summary
numeric_summary <- df %>%
  select(daysdiffcont) %>%
  summarise(across(everything(),
                   list(mean = ~mean(. , na.rm = TRUE),
                        sd   = ~sd(. , na.rm = TRUE)))) %>%
  pivot_longer(
    everything(),
    names_to = c("Variable", ".value"),
    names_sep = "_"
  ) %>%
  mutate(
    Category = NA_character_,
    Count = NA_integer_,
    Percent = NA_real_
  ) %>%
  rename(Mean = mean, SD = sd)

df$commercialized = as.numeric(df$commercialized)

# 2. 分类变量 summary
categorical_summary <- df %>%
  select(ppp_score, citationbased, paperuniv, papergovn, paperfirm, paperunk,patentgovn,patentuniv,patentfirm,patentgovn,patentlone,commercialized,commercialized,prerecommercialized) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Category") %>%
  group_by(Variable, Category) %>%
  summarise(Count = n(), .groups = "drop_last") %>%
  mutate(
    Percent = round(100 * Count / sum(Count), 1),
    Mean = NA_real_,
    SD = NA_real_
  )

# 3. 统一列顺序
numeric_summary <- numeric_summary %>%
  select(Variable, Category, Mean, SD, Count, Percent)

categorical_summary <- categorical_summary %>%
  select(Variable, Category, Mean, SD, Count, Percent)

# 4. 合并
summary_table <- bind_rows(numeric_summary, categorical_summary)

# 5. 导出 CSV
#write_csv(summary_table, "D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\summary_table.csv")

write.csv(numeric_summary, "D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\numeric_summary.csv", row.names = FALSE)
write.csv(categorical_summary, "D:\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\categorical_summary.csv", row.names = FALSE)

print("✅ summary_table.csv 已生成")

# 6. Funding, English, Final, TransferProb 堆叠柱状图

df$commercialized <- ifelse(df$commercialized == 1, 0, 1)

p6=df %>%
  select(citationbased,paperuniv,papergovn,paperfirm,paperunk,patentuniv
         ,patentfirm,patentgovn,patentlone,commercialized,commercialized,prerecommercialized) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable, Value) %>%
  summarise(n = n(), .groups = "drop") %>%
  ggplot(aes(x = Variable, y = n, fill = as.factor(Value))) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "darkblue")) +
  labs(title = "Stacked Bar Chart of Category Variables",
       x = "Category Variables", y = "Count", fill = "Value") +
  theme_minimal()

p1=ggplot(df, aes(x = ppp_score)) +
  geom_histogram(binwidth = 1, fill = "lightgreen", color = "white") +
  labs(title = "Distribution of PPP Score", x = "PPP Score", y = "Count") +
  theme_minimal()

# 2. CitedNum 柱状图
p2=ggplot(df, aes(x = daysdiffcont)) +
  geom_density(binwidth = 5, fill = "orange", color = "orange") +
  labs(title = "Distribution of DaysDiffCont", x = "DaysDiffCont", y = "Density") +
  theme_minimal()

library(patchwork)

(p1+p2)/p6


# AUC Plot

# 10) 绘制 ROC 和 PR 曲线 ---------------------------------------------------
roc_all <- purrr::map_dfr(results, "roc_df")
#pr_all  <- purrr::map_dfr(results, "pr_df")

p_roc <- ggplot(roc_all, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = 2) +
  coord_equal() +
  labs(title = "ROC Curves on Test Set", x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)")+
  theme_minimal()

p_roc


# 导入绘图包
library(ggplot2)
library(reshape2)

# 构建数据框
data <- data.frame(
  Model = rep(c("Transformer", "RNN"), each = 12),
  Metric = rep(c("AUC", "Sensitivity", "Specificity", "Accuracy"), times = 6),
  Category = rep(c("Irregular", "Regular", "Irregular_Delta"), each = 4, times = 2),
  Value = c(
    
    0.70, 0.71, 0.69, 0.71,
    0.63, 0.7, 0.51, 0.55,   
    0.73, 0.68, 0.69, 0.69,  
    
    0.72, 0.60, 0.57, 0.57,
    0.75, 0.73, 0.69, 0.70,  
    0.70, 0.74, 0.71, 0.71  
  )
)

# 绘制柱状图
ggplot(data, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~ Category) +
  labs(
    title = "Comparison of Transformer and RNN Models Across Different Conditions(All Cause Mortality)",
    x = "Performance Metric",
    y = "Mean Value"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)
  ) +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B"))


# Inhospital

# 导入绘图包
library(ggplot2)
library(reshape2)

# 构建数据框
data_inhospital <- data.frame(
  Model = rep(c("Transformer", "RNN"), each = 12),
  Metric = rep(c("AUC", "Sensitivity", "Specificity", "Accuracy"), times = 6),
  Category = rep(c("Irregular", "Regular", "Irregular(Delta)"), each = 4, times = 2),
  Value = c(
    # Transformer
    0.770, 0.600, 0.880, 0.890,
    0.623, 0.756, 0.520, 0.526,  # Irregular
    0.820, 0.830, 0.840, 0.830,  # Regular
      # Irregular(Delta)
    # RNN
    0.596, 0.662, 0.836, 0.835,
    0.866, 0.872, 0.769, 0.772,  # Irregular
    0.766, 0.819, 0.784, 0.784  # Regular
       # Irregular(Delta)
  )
)

# 绘制柱状图（横坐标旋转45度）
ggplot(data_inhospital, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  facet_wrap(~ Category) +
  labs(
    title = "Comparison of Transformer and RNN Models Across Different Conditions(Inhospital Mortality)",
    x = "Performance Metric",
    y = "Mean Value"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)
  ) +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B"))



# 第二个数据库重要性分布

library(readr)
data <- read_csv("data.csv")


data_second = read.csv2("D:\\backup_20260414\\Project\\Work\\postdoctor\\project\\research\\hanlu\\data\\csv\\11461587\\_patent_paper_pairs_plus.csv",sep=",",encoding="gb18030")

# 训练/测试划分rm(list = ls())

df = data_second

library(dplyr)

df <- df %>%
  filter(grepl("Medicine", concepts, ignore.case = TRUE))

df = subset(df,df$concepts=="Medicine")

mean(df$patentuniv)*15955

mean(df$papergovn)*15955

mean(df$paperfirm)*15955

df$commercialized = as.numeric(df$commercialized)

# 线性回归
set.seed(123)

n <- nrow(df)
idx <- split(seq_len(n), cut(seq_len(n), 6, labels = FALSE))
dfs <- lapply(idx, function(i) df[i, ])

df1 <- dfs[[1]]
df2 <- dfs[[2]]
df3 <- dfs[[3]]
df4 <- dfs[[4]]
df5 <- dfs[[5]]
df6 <- dfs[[6]]

ppp_scoreMax <- quantile(df$ppp_score, 0.9)
daysdiffcontMax <- quantile(df$daysdiffcont, 0.9)

#table(df$PaperType)

citationbasedMax <- 1.5*sum(as.numeric(df$citationbased))/15955

paperunivMax <- 1.5*sum(as.numeric(df$paperuniv))/15955

papergovnMax <- 1.5*sum(as.numeric(df$papergovn))/15955

paperfirmMax <- 1.5*sum(as.numeric(df$paperfirm))/15955

paperunkMax <- 1.5*sum(as.numeric(df$paperunk))/15955

patentunivMax <- 1.5*sum(as.numeric(df$patentuniv))/15955

patentgovnMax <- 1.5*sum(as.numeric(df$patentgovn))/15955

patentfirmMax <- 1.5*sum(as.numeric(df$patentfirm))/15955

patentloneMax <- 1.5*sum(as.numeric(df$patentlone))/15955


#df1:0.589,df2:0.350,df3:0.484,df4:0.713, df5:0.568


fit <- lm(as.numeric(commercialized) ~ ppp_score+daysdiffcont+citationbased+paperuniv+papergovn+paperfirm+paperunk+patentuniv
          +patentgovn+patentlone,data = df)
summary(fit)

coef(fit)

coef_no_intercept <- coef(fit)[-1]

#install.packages("lpSolve")
library(lpSolve)

# 目标函数系数
obj <- coef_no_intercept  # 3x + 2y

A <- rbind(
  c(1,0,0,0,0,0,0,0,0,0,0),
  c(0,1,0,0,0,0,0,0,0,0,0),
  c(0,0,1,1,1,1,1,1,1,1,1)
  
)

n <- ncol(A)

bin_idx <- 3:n

lower <- rep(0, n)
upper <- rep(Inf, n)   # 前4个可以连续、无上界（建议你也给上界，避免无界）
upper[bin_idx] <- 1    # 二元变量上界=1

int_vec <- bin_idx

# 约束方向与右端项
dir <- c("<=","<=","<=")
rhs <- c(ppp_scoreMax, daysdiffcontMax, 
         
         paperunivMax+
         
         papergovnMax+
         
         paperfirmMax+
         
         paperunkMax+
         
         patentunivMax+
         
         patentgovnMax+
         
         patentfirmMax+
         
         patentloneMax)

res <- lp(
  direction = "max",
  objective.in = obj,      # 你的目标函数系数向量，长度必须=14
  const.mat = A,
  const.dir = dir,
  const.rhs = rhs,
  int.vec = int_vec,       # 只让第5~14个变量为整数
  all.int = FALSE,
  binary.vec = int_vec,    # 如果你的 lpSolve 版本支持这个参数，会更明确
)

res$status   # 0=最优；2=无可行解；3=无界
res$objval-as.numeric(coef(fit)[1])
res$solution

#

#计算RMSE

# 真实值
y_true <- as.numeric(df6$commercialized)

y_true1<-df1_clean$Transfer

# 各模型预测
pred_lm  <- predict(fit, df6)
pred_rf  <- predict(fit_rf, df6)
pred_xgb <- predict(fit_xgb, df6_t)      # X 为 model.matrix
pred_svm <- predict(fit_svm, df6)
pred_knn <- predict(fit_knn, df6)

rmse_lm  <- sqrt(mean((y_true - pred_lm)^2,na.rm = T))
rmse_lm
rmse_rf  <- sqrt(mean((y_true - pred_rf)^2,na.rm = T))
rmse_rf
rmse_xgb <- sqrt(mean((y_true - pred_xgb)^2,na.rm = T))
rmse_xgb
rmse_svm <- sqrt(mean((y_true - pred_svm)^2,na.rm = T))
rmse_svm
rmse_knn <- sqrt(mean((y_true - pred_knn)^2,na.rm = T))
rmse_knn

#计算R方

library(yardstick)

rsq_lm <- rsq_vec(y_true, pred_lm)
rsq_lm

rsq_rf <- rsq_vec(y_true, pred_rf)
rsq_rf

rsq_xgb <- rsq_vec(y_true, pred_xgb)
rsq_xgb

rsq_svm <- rsq_vec(y_true, pred_svm)
rsq_svm

rsq_knn <- rsq_vec(y_true, pred_knn)
rsq_knn

# (2) 计算MAE
mae_lm <- mean(abs(y_true - pred_lm))
mae_rf <- mean(abs(y_true - pred_rf))
mae_xgb <- mean(abs(y_true - pred_xgb))
mae_svm <- mean(abs(y_true - pred_svm))
mae_knn <- mean(abs(y_true - pred_knn))

mae_lm
mae_rf
mae_xgb
mae_svm
mae_knn

# 4) MSE
mse_lm <- mean((y_true - pred_lm)^2)
mse_rf <- mean((y_true - pred_rf)^2)
mse_xgb <- mean((y_true - pred_xgb)^2)
mse_svm <- mean((y_true - pred_svm)^2)
mse_knn <- mean((y_true - pred_knn)^2)

mse_lm
mse_rf
mse_xgb
mse_svm
mse_knn



#线性回归重要性

#install.packages("relaimpo")
library(relaimpo)

imp_lmg <- calc.relimp(fit, type = "lmg", rela = TRUE)
imp_lmg$lmg |> sort(decreasing = TRUE)

plot(imp_lmg, sort = TRUE)

library(dplyr)

lmg_df <- data.frame(
  variable = names(imp_lmg$lmg),
  importance = as.numeric(imp_lmg$lmg)
) |>
  arrange(desc(importance))

library(ggplot2)

p1<-ggplot(lmg_df, aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(
    x = NULL,
    y = "Relative importance (LMG)",
    title = "Linear Regression relative importance of predictors (LMG)"
  ) +
  theme_minimal(base_size = 13)




set.seed(123)
boot_imp <- boot.relimp(fit, b = 1000, type = "lmg", rank = TRUE, rela = TRUE)
booteval.relimp(boot_imp)    # 重要性CI与稳定性




# 随机森林

#install.packages("randomForest")
library(randomForest)

set.seed(123)

fit_rf <- randomForest(
  as.numeric(commercialized) ~ ppp_score+daysdiffcont+citationbased+paperuniv+papergovn+paperfirm+paperunk+patentuniv
  +patentgovn+patentlone,
  data = df,
  ntree = 500,
  na.action = na.omit,
  importance = TRUE
)

fit_rf
varImpPlot(fit_rf)

imp <- importance(fit_rf)

imp_df <- data.frame(
  variable = rownames(imp),
  importance = imp[, "%IncMSE"]
)

library(ggplot2)
library(dplyr)

p2<-imp_df |>
  arrange(importance) |>
  ggplot(aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(
    x = NULL,
    y = "Increase in MSE (%)",
    title = "Random Forest Variable importance"
  ) +
  theme_classic(base_size = 13)



#XGBoost

#install.packages("xgboost")
library(xgboost)

# 删除这些变量有缺失值的行
df4_clean <- df[complete.cases(df[, c("commercialized","ppp_score","daysdiffcont","citationbased","paperuniv","papergovn","paperfirm","paperunk","patentuniv",
                                        "patentfirm","patentgovn","patentlone")]), ]


X <- model.matrix(
  as.numeric(commercialized) ~ ppp_score+daysdiffcont+citationbased+paperuniv+papergovn+paperfirm+paperunk+patentuniv
  +patentgovn+patentlone,
  data = df
)[, -1]   # 去掉截距

X_df6 <- model.matrix(
  as.numeric(commercialized) ~ ppp_score+daysdiffcont+citationbased+paperuniv+papergovn+paperfirm+paperunk+patentuniv
  +patentgovn+patentlone,
  data = df
)[, -1]   # 去掉截距

X_df6<-as.matrix(X_df6)

X <- as.matrix(X)

# 使用 xgb.DMatrix 创建测试数据的 DMatrix
dtest <- xgb.DMatrix(data = X, missing = NA)

y <- df$commercialized

fit_xgb <- xgboost(
  data = X,
  label = y,
  objective = "reg:squarederror",  # 回归任务，平方误差
  nrounds = 300,                   # 迭代次数
  eta = 0.05,                      # 学习率
  max_depth = 6,                   # 树的深度
  subsample = 0.8,                 # 样本子集比例
  colsample_bytree = 0.8,          # 每棵树的特征子集比例
  verbose = 0
  
)

df6_t <- xgb.DMatrix(data = X_df6, missing = NA)


library(xgboost)

imp_xgb <- xgb.importance(
  feature_names = colnames(X),
  model = fit_xgb
)

library(dplyr)

imp_df <- imp_xgb |>
  dplyr::select(Feature, Gain) |>
  arrange(desc(Gain))



library(ggplot2)

p3<-ggplot(imp_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  labs(
    x = NULL,
    y = "Gain (relative importance)",
    title = "XGBoost feature importance"
  ) +
  theme_classic(base_size = 14)

colnames(imp_xgb)


#SVM


vars <- c("commercialized","ppp_score","daysdiffcont","citationbased","paperuniv","papergovn","paperfirm","paperunk","patentuniv",
          "patentfirm","patentgovn","patentlone")

df2_svm <- df[complete.cases(df[, vars]), vars]

df2_svm$commercialized = as.numeric(df2_svm$commercialized)

library(e1071)
set.seed(123)

fit_svm <- svm(
  as.numeric(commercialized) ~ ppp_score+daysdiffcont+citationbased+paperuniv+papergovn+paperfirm+paperunk+patentuniv
  +patentgovn+patentlone,
  data = df2_svm,
  type = "eps-regression",
  kernel = "radial"
)

rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

perm_importance_svm <- function(model, data, y = "commercialized", nrep = 20, seed = 123){
  set.seed(seed)
  y_true <- data[[y]]
  base_pred <- predict(model, data)
  base_rmse <- rmse(y_true, base_pred)
  
  feats <- setdiff(names(data), y)
  out <- lapply(feats, function(f){
    inc <- replicate(nrep, {
      dperm <- data
      dperm[[f]] <- sample(dperm[[f]])   # 打乱该变量
      pred <- predict(model, dperm)
      rmse(y_true, pred) - base_rmse
    })
    data.frame(feature = f, importance = mean(inc), sd = sd(inc))
  })
  do.call(rbind, out)
}


library(dplyr)
library(ggplot2)

imp_svm <- perm_importance_svm(fit_svm, df2_svm, nrep = 30) |>
  arrange(desc(importance))

p4<-ggplot(imp_svm, aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "skyblue")+
  coord_flip() +
  labs(
    x = NULL,
    y = "Permutation importance (ΔRMSE)",
    title = "SVM (RBF) feature importance"
  ) +
  theme_classic(base_size = 14)



#knn

# install.packages("caret")
library(caret)

set.seed(123)

vars <- c("commercialized","ppp_score","daysdiffcont","citationbased","paperuniv","papergovn","paperfirm","paperunk","patentuniv",
          "patentfirm","patentgovn","patentlone")

df_knn <- df[complete.cases(df[, vars]), vars]

df_knn$commercialized = as.numeric(df_knn$commercialized)

library(caret)

set.seed(123)
ctrl <- trainControl(method = "cv", number = 5)

fit_knn <- train(
  as.numeric(commercialized) ~ ppp_score+daysdiffcont+citationbased+paperuniv+papergovn+paperfirm+paperunk+patentuniv
  +patentgovn+patentlone,
  data = df_knn,
  method = "knn",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 10
)
fit_knn


rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))



perm_importance_knn <- function(model, data, y = "commercialized", nrep = 30, seed = 123){
  set.seed(seed)
  y_true <- data[[y]]
  
  # 基线 RMSE（注意：caret 会自动使用训练时的 preProcess）
  base_pred <- predict(model, newdata = data)
  base_rmse <- rmse(y_true, base_pred)
  
  feats <- setdiff(names(data), y)
  
  out <- lapply(feats, function(f){
    inc <- replicate(nrep, {
      dperm <- data
      dperm[[f]] <- sample(dperm[[f]])   # 打乱该变量
      pred <- predict(model, newdata = dperm)
      rmse(y_true, pred) - base_rmse
    })
    data.frame(feature = f, importance = mean(inc), sd = sd(inc))
  })
  
  do.call(rbind, out)
}


library(dplyr)
library(ggplot2)

imp_knn <- perm_importance_knn(fit_knn, df_knn, nrep = 50) |>
  arrange(desc(importance))

p5<-ggplot(imp_knn, aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "skyblue")+
  coord_flip() +
  labs(
    x = NULL,
    y = "Permutation importance (ΔRMSE)",
    title = "kNN feature importance"
  ) +
  theme_classic(base_size = 14)


library(patchwork)

(p1+p2)/(p3+p4+p5)

# 查看 imp_svm 数据框的结构
str(imp_svm)

# 打印前几行数据，确保有 "feature" 和 "importance"
head(imp_svm)



# 第一个数据库重做PCA

vars <- c("Transfer","PublicationYear","CitedNum","AuthorCount",
          "TitleSimilarity","PaperType","Funding","English","Final")

df1_clean <- df1[complete.cases(df1[, vars]), vars]

# 结果变量（用于上色）
y <- df1_clean$Transfer

# 自变量
xvars <- setdiff(vars, "Transfer")
X_raw <- df1_clean[, xvars]

X <- model.matrix(~ . - 1, data = X_raw)  # -1 表示不含截距
pca_fit <- prcomp(X, center = TRUE, scale. = TRUE)

evar <- (pca_fit$sdev^2) / sum(pca_fit$sdev^2)

plot(evar, type="b", pch=19,
     xlab="Principal Component", ylab="Proportion of Variance Explained",
     main="Scree Plot")

cumvar <- cumsum(evar)

plot(cumvar, type="b", pch=19,
     xlab="Principal Component", ylab="Cumulative Proportion",
     main="Cumulative Explained Variance")
abline(h=0.80, lty=2)

scores <- as.data.frame(pca_fit$x[, 1:2, drop=FALSE])
scores$Transfer <- factor(y)

plot(scores$PC1, scores$PC2,
     pch=19,
     col=ifelse(as.numeric(scores$Transfer)>0.1,"red","black"),
     xlab="PC1", ylab="PC2",
     main="PC1-PC2 (colored by Transfer)")
legend("topright", legend=c("0","1"), col=c("black","red"), pch=19)

load1 <- sort(pca_fit$rotation[, "PC1"], decreasing = TRUE)
barplot(load1, las=2, main="Loadings on PC1", ylab="Loading")
abline(h=0)

load2 <- sort(pca_fit$rotation[, "PC2"], decreasing = TRUE)
barplot(load2, las=2, main="Loadings on PC2", ylab="Loading")
abline(h=0)

biplot(pca_fit, scale = 0, cex=0.7)



#ggplot2

library(ggplot2)

# 解释方差
evar <- (pca_fit$sdev^2) / sum(pca_fit$sdev^2)
cumvar <- cumsum(evar)

df_evar <- data.frame(
  PC = factor(paste0("PC", seq_along(evar)), levels = paste0("PC", seq_along(evar))),
  PC_num = seq_along(evar),
  evar = evar,
  cumvar = cumvar
)

# 样本得分（PC1/PC2）
scores <- as.data.frame(pca_fit$x[, 1:2, drop = FALSE])
scores$Transfer <- as.numeric(df1_clean$Transfer)  # 如果你是 commercialized，就换成 df4$commercialized 对齐后的那个

# 变量载荷（PC1/PC2）
loadings <- as.data.frame(pca_fit$rotation[, 1:2, drop = FALSE])
loadings$feature <- rownames(loadings)

p_scree <- ggplot(df_evar, aes(x = PC, y = evar, group = 1)) +
  geom_point(size = 2) +
  geom_line() +
  labs(x = "Principal Component", y = "Proportion of Variance Explained",
       title = "Scree Plot") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p_scree


p_cum <- ggplot(df_evar, aes(x = PC, y = cumvar, group = 1)) +
  geom_point(size = 2) +
  geom_line() +
  geom_hline(yintercept = 0.80, linetype = 2) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Principal Component", y = "Cumulative Proportion",
       title = "Cumulative Explained Variance") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p_scatter <- ggplot(scores, aes(x = PC1, y = PC2, color = Transfer)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "PCA Scores (PC1 vs PC2)", x = "PC1", y = "PC2") +
  theme_classic()

p_scatter


df_load1 <- data.frame(
  feature = loadings$feature,
  loading = loadings$PC1
)
df_load1 <- df_load1[order(abs(df_load1$loading), decreasing = TRUE), ]
df_load1$feature <- factor(df_load1$feature, levels = df_load1$feature)

p_load1 <- ggplot(df_load1, aes(x = feature, y = loading,fill="purple")) +
  geom_col() +
  coord_flip() +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  labs(title = "Loadings on PC1", x = NULL, y = "Loading") +
  theme_classic()

p_load1

df_load2 <- data.frame(
  feature = loadings$feature,
  loading = loadings$PC2
)
df_load2 <- df_load2[order(abs(df_load2$loading), decreasing = TRUE), ]
df_load2$feature <- factor(df_load2$feature, levels = df_load2$feature)

p_load2 <- ggplot(df_load2, aes(x = feature, y = loading,fill="yellow")) +
  geom_col() +
  coord_flip() +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  labs(title = "Loadings on PC2", x = NULL, y = "Loading") +
  theme_classic()

p_load2

# 缩放箭头长度：让箭头在scores坐标范围内可见
arrow_scale <- min(
  (max(scores$PC1) - min(scores$PC1)) / (max(loadings$PC1) - min(loadings$PC1)),
  (max(scores$PC2) - min(scores$PC2)) / (max(loadings$PC2) - min(loadings$PC2))
) * 0.2

df_arrows <- data.frame(
  feature = loadings$feature,
  xend = loadings$PC1 * arrow_scale,
  yend = loadings$PC2 * arrow_scale
)

p_biplot <- ggplot(scores, aes(PC1, PC2, color = Transfer)) +
  geom_point(size = 2, alpha = 0.8) +
  geom_segment(data = df_arrows,
               aes(x = 0, y = 0, xend = xend, yend = yend),
               inherit.aes = FALSE,
               arrow = arrow(length = unit(0.2, "cm"))) +
  geom_text(data = df_arrows,
            aes(x = xend, y = yend, label = feature),
            inherit.aes = FALSE,
            size = 3, vjust = 1) +
  labs(title = "PCA Biplot (ggplot2)", x = "PC1", y = "PC2") +
  theme_classic()

p_biplot

(p_scree+p_cum+p_scatter)/(p_load1+p_load2)+
  plot_annotation(title = "PCA Visualization(IIDS)")


# 第二个数据ggplot2做主成分分析

library(ggplot2)
library(patchwork)

# 安装 remotes
#install.packages("remotes")

library(remotes)

# 安装旧版本 patchwork
remotes::install_version(
  "patchwork",
  version = "1.1.3",
  repos = "https://cloud.r-project.org",
  upgrade = "never"
)

#install.packages("patchwork")

# ========== 1) 变量 ==========
yvar <- "commercialized"
xvars <- c("ppp_score","daysdiffcont","citationbased",
           "paperuniv","papergovn","paperfirm","paperunk",
           "patentuniv","patentgovn","patentlone")

vars <- c(yvar, xvars)

# ========== 2) 清洗：只保留这些变量且不缺失 ==========
df2_clean <- df[complete.cases(df[, vars]), vars]

y <- df2_clean[[yvar]]
X_raw <- df2_clean[, xvars]

# 如果自变量里有 factor/字符，先 dummy 展开（这里多数是0/1，但写成通用更稳）
X <- model.matrix(~ . - 1, data = X_raw)

# ========== 3) PCA ==========
pca_fit <- prcomp(X, center = TRUE, scale. = TRUE)

# ========== 4) 解释方差数据 ==========
evar <- (pca_fit$sdev^2) / sum(pca_fit$sdev^2)
cumvar <- cumsum(evar)

df_evar <- data.frame(
  PC = factor(paste0("PC", seq_along(evar)), levels = paste0("PC", seq_along(evar))),
  evar = evar,
  cumvar = cumvar
)

# ========== 5) scores（PC1/PC2） ==========
scores <- as.data.frame(pca_fit$x[, 1:2, drop = FALSE])

# 上色：如果 commercialized 是 0/1 或 factor，都统一成 factor
scores$commercialized <- factor(as.numeric(y))

# ========== 6) loadings（PC1/PC2） ==========
loadings <- as.data.frame(pca_fit$rotation[, 1:2, drop = FALSE])
loadings$feature <- rownames(loadings)

df_load1 <- data.frame(feature = loadings$feature, loading = loadings$PC1)
df_load1 <- df_load1[order(abs(df_load1$loading), decreasing = TRUE), ]
df_load1$feature <- factor(df_load1$feature, levels = df_load1$feature)

df_load2 <- data.frame(feature = loadings$feature, loading = loadings$PC2)
df_load2 <- df_load2[order(abs(df_load2$loading), decreasing = TRUE), ]
df_load2$feature <- factor(df_load2$feature, levels = df_load2$feature)

# ========== 7) ggplot2 图 ==========
p_scree <- ggplot(df_evar, aes(x = PC, y = evar, group = 1)) +
  geom_point(size = 2) +
  geom_line() +
  labs(x = "Principal Component",
       y = "Proportion of Variance Explained",
       title = "Scree Plot") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p_cum <- ggplot(df_evar, aes(x = PC, y = cumvar, group = 1)) +
  geom_point(size = 2) +
  geom_line() +
  geom_hline(yintercept = 0.80, linetype = 2) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Principal Component",
       y = "Cumulative Proportion",
       title = "Cumulative Explained Variance") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p_scatter <- ggplot(scores, aes(x = PC1, y = PC2, color = commercialized)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Scores (PC1 vs PC2) colored by commercialized",
       x = "PC1", y = "PC2", color = "commercialized") +
  theme_classic()

p_load1 <- ggplot(df_load1, aes(x = feature, y = loading)) +
  geom_col(fill = "gold") +
  coord_flip() +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  labs(title = "Loadings on PC1", x = NULL, y = "Loading") +
  theme_classic()

p_load2 <- ggplot(df_load2, aes(x = feature, y = loading)) +
  geom_col(fill = "gold") +
  coord_flip() +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  labs(title = "Loadings on PC2", x = NULL, y = "Loading") +
  theme_classic()

# ========== 8) ggplot2 biplot（可选） ==========
arrow_scale <- min(
  (max(scores$PC1) - min(scores$PC1)) / (max(loadings$PC1) - min(loadings$PC1)),
  (max(scores$PC2) - min(scores$PC2)) / (max(loadings$PC2) - min(loadings$PC2))
) * 0.2

df_arrows <- data.frame(
  feature = loadings$feature,
  xend = loadings$PC1 * arrow_scale,
  yend = loadings$PC2 * arrow_scale
)

p_biplot <- ggplot(scores, aes(PC1, PC2, color = commercialized)) +
  geom_point(size = 2, alpha = 0.8) +
  geom_segment(data = df_arrows,
               aes(x = 0, y = 0, xend = xend, yend = yend),
               inherit.aes = FALSE,
               arrow = arrow(length = unit(0.2, "cm"))) +
  geom_text(data = df_arrows,
            aes(x = xend, y = yend, label = feature),
            inherit.aes = FALSE,
            size = 3, vjust = 1) +
  labs(title = "PCA Biplot", x = "PC1", y = "PC2", color = "commercialized") +
  theme_classic()

# ========== 9) patchwork 拼图 ==========
((p_scree + p_cum + p_scatter) / (p_load1 + p_load2)) +
  plot_annotation(title = "PCA Visualization(RoS)")

# 如果你还想单独显示 biplot：
# p_biplot



#第二个数据库修正


# ============================================================
# 1. 初始化环境
# ============================================================


install.packages(
  "rsample",
  repos = "https://cloud.r-project.org"
)


# 设置CRAN镜像
options(repos = c(
  CRAN = "https://cloud.r-project.org"
))

# 检查R版本
R.version.string

# 1. 设置CRAN镜像
options(repos = c(
  CRAN = "https://cloud.r-project.org"
))

# 2. 检查rlang版本
packageVersion("rlang")

# 3. 安装remotes
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

# 4. 安装tune 1.3.0
remotes::install_version(
  "tune",
  version = "1.3.0",
  repos = "https://cloud.r-project.org",
  upgrade = "never",
  dependencies = c(
    "Depends",
    "Imports",
    "LinkingTo"
  )
)

options(repos = c(
  CRAN = "https://cloud.r-project.org"
))

# 安装S7
install.packages("S7", type = "source")

# 验证
packageVersion("S7")
library(S7)


# 设置CRAN镜像
options(repos = c(
  CRAN = "https://cloud.r-project.org"
))

# 如果没有remotes，先安装
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

# 安装tune 1.1.2
remotes::install_version(
  "tune",
  version = "1.1.2",
  repos = "https://cloud.r-project.org",
  upgrade = "never",
  dependencies = FALSE
)

# 设置CRAN镜像

rm(list = ls())

#install.packages("lava")

library(dplyr)
library(tidyr)
library(purrr)

library(rsample)
library(recipes)
library(parsnip)
library(workflows)
library(tune)
library(dials)
library(yardstick)
library(hardhat)

library(glmnet)
library(ranger)
library(xgboost)
library(kernlab)
library(kknn)
library(igraph)
library(ggplot2)

set.seed(42)

# ============================================================
# 2. 读取数据
# ============================================================

input_file <- paste0(
  "D:/backup_20260414/Project/Work/postdoctor/project/research/hanlu/data/csv/11461587/_patent_paper_pairs_plus.csv"
)

output_dir <- paste0(
  "D:/backup_20260414/Project/Work/postdoctor/project/research/",
  "hanlu/data/csv/"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

data_second <- read.csv(
  input_file,
  sep = ",",
  fileEncoding = "GB18030",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

df <- data_second

# 筛选Medicine领域
df <- df %>%
  filter(
    !is.na(concepts),
    grepl("Medicine", concepts, ignore.case = TRUE)
  )

df = subset(df,df$concepts=="Medicine")


# 如果只保留完全等于Medicine的记录：
# df <- df %>% filter(concepts == "Medicine")

# 分类标签
df <- df %>%
  filter(!is.na(commercialized))

df$commercialized <- factor(
  as.character(df$commercialized),
  levels = c("0", "1")
)

if (anyNA(df$commercialized)) {
  stop("commercialized存在非0/1的取值，请检查数据。")
}

if (length(unique(df$commercialized)) != 2) {
  stop("筛选后commercialized必须同时包含0和1。")
}

cat("\n========== 样本分布 ==========\n")

print(table(df$commercialized))
print(prop.table(table(df$commercialized)))

# ============================================================
# 3. 自动识别专利ID和论文ID
# ============================================================

# 如果知道实际列名，直接填写：
# patent_id_col <- "patent_id"
# paper_id_col  <- "paper_id"

patent_id_col <- NULL
paper_id_col  <- NULL

patent_candidates <- c(
  "patent_id",
  "patentid",
  "patent_ID",
  "patent_number",
  "patent_no",
  "patent_num",
  "publication_number",
  "patent_publication_number",
  "patent"
)

paper_candidates <- c(
  "paper_id",
  "paperid",
  "paper_ID",
  "paper_number",
  "paper_no",
  "paper_num",
  "openalex_id",
  "work_id",
  "doi",
  "paper"
)

detect_column <- function(
  columns,
  candidates,
  manually_selected = NULL
) {
  
  if (!is.null(manually_selected)) {
    if (!(manually_selected %in% columns)) {
      stop(
        "指定的ID列不存在：",
        manually_selected
      )
    }
    return(manually_selected)
  }
  
  idx <- match(
    tolower(candidates),
    tolower(columns)
  )
  
  idx <- idx[!is.na(idx)]
  
  if (length(idx) == 0) {
    return(NULL)
  }
  
  columns[idx[1]]
}

patent_id_col <- detect_column(
  names(df),
  patent_candidates,
  patent_id_col
)

paper_id_col <- detect_column(
  names(df),
  paper_candidates,
  paper_id_col
)

if (
  is.null(patent_id_col) ||
  is.null(paper_id_col)
) {
  
  cat("\nCSV文件中的全部字段名：\n")
  print(names(df))
  
  stop(
    paste0(
      "无法自动识别专利ID或论文ID。\n",
      "请在第3部分手动指定",
      "patent_id_col和paper_id_col。"
    )
  )
}

if (patent_id_col == paper_id_col) {
  stop("专利ID列与论文ID列不能相同。")
}

cat("\n专利ID字段:", patent_id_col)
cat("\n论文ID字段:", paper_id_col, "\n")

# ============================================================
# 4. 根据专利-论文关联网络创建分组
# ============================================================

# 核心思想：
# 同一专利或同一论文对应的配对记录必须归为一组。
# 如果不同记录之间存在间接关联，也归为同一组。
#
# 使用igraph连通分量计算分组。

normalize_id <- function(x) {
  
  x <- trimws(as.character(x))
  
  x[
    is.na(x) |
      tolower(x) %in% c(
        "",
        "na",
        "nan",
        "null",
        "none"
      )
  ] <- NA_character_
  
  return(x)
}

df$.patent_id <- normalize_id(
  df[[patent_id_col]]
)

df$.paper_id <- normalize_id(
  df[[paper_id_col]]
)

missing_patent <- sum(is.na(df$.patent_id))
missing_paper <- sum(is.na(df$.paper_id))

cat("\n专利ID缺失记录数:", missing_patent)
cat("\n论文ID缺失记录数:", missing_paper, "\n")

# 对缺失ID构造仅属于本行的虚拟节点。
# 不会把所有NA错误地归为同一个组。
#
# 前缀P:/A:用于区分专利与论文命名空间。

n <- nrow(df)

patent_node <- ifelse(
  is.na(df$.patent_id),
  paste0("P:MISSING_ROW_", seq_len(n)),
  paste0("P:", df$.patent_id)
)

paper_node <- ifelse(
  is.na(df$.paper_id),
  paste0("A:MISSING_ROW_", seq_len(n)),
  paste0("A:", df$.paper_id)
)

# 每一条配对记录构成一条边
edge_df <- data.frame(
  from = patent_node,
  to = paper_node,
  stringsAsFactors = FALSE
)

# 构建二部图
g <- igraph::graph_from_data_frame(
  edge_df,
  directed = FALSE
)

# 连通分量
comp <- igraph::components(g)

# 每条边的起点对应的连通分量
node_membership <- comp$membership

df$leakage_group <- as.character(
  unname(node_membership[patent_node])
)

# 删除中间图对象以减少内存
rm(g, comp, node_membership, edge_df)
gc()

cat(
  "\n========== 关联分组信息 ==========\n"
)

cat("总记录数:", nrow(df), "\n")

cat(
  "连通分量数量:",
  n_distinct(df$leakage_group),
  "\n"
)

group_sizes <- df %>%
  count(
    leakage_group,
    name = "n"
  ) %>%
  arrange(desc(n))

cat("最大连通分量记录数:", max(group_sizes$n), "\n")

cat(
  "最大连通分量占比:",
  round(max(group_sizes$n) / nrow(df), 4),
  "\n"
)

# 保存分组规模
write.csv(
  group_sizes,
  file.path(
    output_dir,
    "group_sizes.csv"
  ),
  row.names = FALSE
)

# ============================================================
# 5. 定义ID重叠检查函数
# ============================================================

check_overlap <- function(
  data_a,
  data_b,
  label = "数据集"
) {
  
  patent_a <- unique(
    na.omit(data_a$.patent_id)
  )
  
  patent_b <- unique(
    na.omit(data_b$.patent_id)
  )
  
  paper_a <- unique(
    na.omit(data_a$.paper_id)
  )
  
  paper_b <- unique(
    na.omit(data_b$.paper_id)
  )
  
  patent_overlap <- intersect(
    patent_a,
    patent_b
  )
  
  paper_overlap <- intersect(
    paper_a,
    paper_b
  )
  
  group_overlap <- intersect(
    unique(data_a$leakage_group),
    unique(data_b$leakage_group)
  )
  
  if (
    length(patent_overlap) > 0 ||
    length(paper_overlap) > 0 ||
    length(group_overlap) > 0
  ) {
    
    stop(
      paste0(
        label,
        "存在数据泄漏！",
        "专利ID重叠=",
        length(patent_overlap),
        "；论文ID重叠=",
        length(paper_overlap),
        "；关联组重叠=",
        length(group_overlap)
      )
    )
  }
  
  return(TRUE)
}

# ============================================================
# 6. 按关联分组划分训练集和测试集
# ============================================================

# 不能使用普通initial_split()
#
# 这里使用group_initial_split()
# 同一个连通分量只能进入训练集或测试集。
#
# 由于分组大小和标签比例可能不均衡，
# 尝试不同随机种子，寻找满足类别要求、
# 且测试集比例接近20%的划分。
#
# 所有选择仅基于组大小与标签分布，
# 不使用模型预测结果。

# 初始化
valid_splits <- list()
split_quality <- numeric()

# 尝试10次不同随机种子
for (seed_i in 1:100) {
  
  set.seed(seed_i)
  
  candidate <- group_initial_split(
    df,
    group = leakage_group,
    prop = 0.8
  )
  
  tr <- training(candidate)
  te <- testing(candidate)
  
  # 检查训练集和测试集是否均包含0和1
  tr_tab <- table(tr$commercialized)
  te_tab <- table(te$commercialized)
  
  if (
    all(tr_tab > 0) &&
    all(te_tab > 0)
  ) {
    
    # 测试集比例尽量接近20%
    ratio_error <- abs(
      nrow(te) / nrow(df) - 0.2
    )
    
    # 整体正类比例
    overall_rate <- mean(
      df$commercialized == "1"
    )
    
    # 训练集正类比例
    train_rate <- mean(
      tr$commercialized == "1"
    )
    
    # 测试集正类比例
    test_rate <- mean(
      te$commercialized == "1"
    )
    
    # 计算划分质量
    quality <- ratio_error +
      abs(train_rate - overall_rate) +
      abs(test_rate - overall_rate)
    
    # 保存有效划分
    valid_splits[[length(valid_splits) + 1]] <- candidate
    
    # 保存划分质量
    split_quality <- c(
      split_quality,
      quality
    )
  }
  
  # ==========================================
  # 每完成10次划分，显示一次进度
  # ==========================================
  if (seed_i %% 10 == 0) {
    cat(
      "已完成", seed_i, "/100 次划分",
      "| 有效划分数:", length(valid_splits),
      "\n"
    )
    flush.console()
  }
}

# 检查是否找到有效划分
if (length(valid_splits) == 0) {
  stop(
    "无法找到有效的训练集/测试集分组划分，请检查关联分组和类别分布。"
  )
}

# 选取质量最好的划分
best_split_idx <- which.min(split_quality)

data_split <- valid_splits[[best_split_idx]]

# 提取训练集和测试集
train_df <- training(data_split)
test_df <- testing(data_split)

# 输出结果
cat("\n========== 最终分组结果 ==========\n")

cat("训练集样本数:", nrow(train_df), "\n")
cat("测试集样本数:", nrow(test_df), "\n")

cat(
  "训练集占比:",
  round(nrow(train_df) / nrow(df), 4),
  "\n"
)

cat(
  "测试集占比:",
  round(nrow(test_df) / nrow(df), 4),
  "\n"
)

cat("\n训练集类别分布:\n")
print(table(train_df$commercialized))

cat("\n测试集类别分布:\n")
print(table(test_df$commercialized))

# 检查关联分组是否重叠
group_overlap <- intersect(
  unique(train_df$leakage_group),
  unique(test_df$leakage_group)
)

cat(
  "\n训练集和测试集重复关联分组数:",
  length(group_overlap),
  "\n"
)

stopifnot(length(group_overlap) == 0)

cat("\n分组划分成功，无关联分组重叠！\n")

if (length(valid_splits) == 0) {
  stop(
    paste0(
      "无法找到同时包含两类样本的分组训练/测试划分。",
      "可能存在过大的关联分量。",
      "请检查group_sizes.csv。"
    )
  )
}

best_split_idx <- which.min(split_quality)

data_split <- valid_splits[[best_split_idx]]

train_df <- training(data_split)
test_df <- testing(data_split)

rm(valid_splits)
gc()

cat(
  "\n========== 分组训练/测试划分 ==========\n"
)

cat(
  "训练集样本数:",
  nrow(train_df),
  "\n"
)

cat(
  "测试集样本数:",
  nrow(test_df),
  "\n"
)

cat(
  "实际训练集占比:",
  round(nrow(train_df) / nrow(df), 4),
  "\n"
)

cat("\n训练集类别分布:\n")
print(table(train_df$commercialized))

cat("\n测试集类别分布:\n")
print(table(test_df$commercialized))

# 严格检查两者是否存在ID交叉
stopifnot(
  check_overlap(
    train_df,
    test_df,
    "训练集/测试集"
  )
)

cat(
  "\n训练/测试集专利和论文ID重叠检查通过！\n"
)

# ============================================================
# 7. 分组五折交叉验证
# ============================================================

# 每一折也需要使用同样的关联分组。
#
# balance="observations"
# 尽量使每折样本数量接近。
#
# 只接受每一折分析集和验证集都包含两类样本、
# 且各折不存在共享专利或论文ID的划分。

if (
  n_distinct(train_df$leakage_group) < 5
) {
  stop("训练集独立关联分组不足5个，无法进行5折分组交叉验证。")
}

cv_folds <- NULL

for (seed_i in 1:100) {
  
  set.seed(seed_i)
  
  candidate_cv <- group_vfold_cv(
    train_df,
    group = leakage_group,
    v = 5,
    balance = "observations"
  )
  
  cv_valid <- TRUE
  
  for (i in seq_len(nrow(candidate_cv))) {
    
    analysis_df <- analysis(
      candidate_cv$splits[[i]]
    )
    
    assessment_df <- assessment(
      candidate_cv$splits[[i]]
    )
    
    if (
      any(table(analysis_df$commercialized) == 0) ||
      any(table(assessment_df$commercialized) == 0)
    ) {
      cv_valid <- FALSE
      break
    }
    
    if (
      length(intersect(
        unique(analysis_df$leakage_group),
        unique(assessment_df$leakage_group)
      )) > 0
    ) {
      cv_valid <- FALSE
      break
    }
  }
  
  if (cv_valid) {
    cv_folds <- candidate_cv
    break
  }
}

if (is.null(cv_folds)) {
  stop(
    paste0(
      "100次尝试后仍无法生成每折均包含两类样本的",
      "5折分组交叉验证。",
      "请检查关联分组与类别分布。"
    )
  )
}

cat(
  "\n========== 五折分组交叉验证 ==========\n"
)

for (i in seq_len(nrow(cv_folds))) {
  
  a <- analysis(cv_folds$splits[[i]])
  b <- assessment(cv_folds$splits[[i]])
  
  stopifnot(
    check_overlap(
      a,
      b,
      paste0("CV Fold ", i)
    )
  )
  
  cat(
    "Fold", i,
    "| 训练:", nrow(a),
    "| 验证:", nrow(b),
    "| 验证正类:", sum(b$commercialized == "1"),
    "\n"
  )
}

# ============================================================
# 8. 特征预处理
# ============================================================

feature_names <- c(
  "ppp_score",
  "daysdiffcont",
  "citationbased",
  "paperuniv",
  "papergovn",
  "paperfirm",
  "paperunk",
  "patentuniv",
  #"patentfirm",
  "patentgovn",
  "patentlone"
)

missing_features <- setdiff(
  feature_names,
  names(train_df)
)

if (length(missing_features) > 0) {
  stop(
    "缺少预测特征：",
    paste(missing_features, collapse = ", ")
  )
}

# ID和关联组都不会进入模型特征。
#
# 显式指定11个预测变量，
# 避免模型意外使用专利ID或论文ID。

rec <- recipe(
  commercialized ~
    ppp_score +
    daysdiffcont +
    citationbased +
    paperuniv +
    papergovn +
    paperfirm +
    paperunk +
    patentuniv +
    #patentfirm +
    patentgovn +
    patentlone,
  data = train_df
) |>
  step_impute_median(
    all_numeric_predictors()
  ) |>
  step_impute_mode(
    all_nominal_predictors()
  ) |>
  step_dummy(
    all_nominal_predictors()
  ) |>
  step_zv(
    all_predictors()
  ) |>
  step_normalize(
    all_numeric_predictors()
  )

# ============================================================
# 9. 定义分类评价指标
# ============================================================

# 正类是commercialized = 1

auc_positive <- yardstick::metric_tweak(
  "auc_positive",
  yardstick::roc_auc,
  event_level = "second"
)

sens_positive <- yardstick::metric_tweak(
  "sens_positive",
  yardstick::sens,
  event_level = "second"
)

spec_positive <- yardstick::metric_tweak(
  "spec_positive",
  yardstick::spec,
  event_level = "second"
)

metrics_all <- yardstick::metric_set(
  auc_positive,
  sens_positive,
  spec_positive,
  yardstick::accuracy
)

# ============================================================
# 10. 定义五种模型与超参数网格
# ============================================================

# -------------------------
# 10.1 Logistic Regression
# -------------------------

log_spec <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) |>
  set_engine("glmnet") |>
  set_mode("classification")

log_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(),
  levels = c(10, 5)
)

# -------------------------
# 10.2 Random Forest
# -------------------------

rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000
) |>
  set_engine(
    "ranger",
    importance = "impurity"
  ) |>
  set_mode("classification")

rf_grid <- grid_regular(
  mtry(
    range = c(2L, 10L)
  ),
  min_n(
    range = c(2L, 20L)
  ),
  levels = 5
)

# -------------------------
# 10.3 XGBoost
# -------------------------

xgb_spec <- boost_tree(
  trees = 1000,
  learn_rate = tune(),
  tree_depth = tune(),
  mtry = tune(),
  loss_reduction = tune(),
  min_n = tune()
) |>
  set_engine("xgboost") |>
  set_mode("classification")

xgb_grid <- grid_latin_hypercube(
  learn_rate(
    range = c(-3, log10(0.3))
  ),
  tree_depth(
    range = c(2L, 10L)
  ),
  mtry(
    range = c(2L, 10L)
  ),
  loss_reduction(),
  min_n(),
  size = 25
)

# -------------------------
# 10.4 SVM-RBF
# -------------------------

svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) |>
  set_engine("kernlab") |>
  set_mode("classification")

svm_grid <- grid_regular(
  cost(
    range = c(-3, 3)
  ),
  rbf_sigma(
    range = c(-3, 1)
  ),
  levels = 7
)

# ============================================================
# 10.5 kNN
# ============================================================

knn_spec <- parsnip::nearest_neighbor(
  neighbors = tune::tune(),
  weight_func = tune::tune()
) |>
  parsnip::set_engine("kknn") |>
  parsnip::set_mode("classification")

knn_grid <- dials::grid_regular(
  dials::neighbors(
    range = c(3L, 75L)
  ),
  dials::weight_func(
    values = c(
      "rectangular",
      "triangular",
      "epanechnikov",
      "gaussian"
    )
  ),
  levels = c(15, 4)
)

print(knn_grid)

# ============================================================
# 11. 超参数调优函数（修复指标重复传参）
# ============================================================

# 调参阶段仅计算原生指标，彻底绕开旧的 sens_positive()
# 由于 0/1 两类别互补，roc_auc 对类别 0 使用 .pred_0
# 与对类别 1 使用 .pred_1 得到相同的二分类 AUC。
# 以类别 1 为阳性的敏感度/特异度在第13部分单独计算。
tuning_metrics <- yardstick::metric_set(
  yardstick::roc_auc,
  yardstick::accuracy
)

tune_one <- function(model_spec, grid, model_name) {
  
  cat("\n========== 开始调参:", model_name, "==========\n")
  
  wf <- workflows::workflow() |>
    workflows::add_model(model_spec) |>
    workflows::add_recipe(rec)
  
  set.seed(42)
  res <- tune::tune_grid(
    wf,
    resamples = cv_folds,
    grid = grid,
    metrics = tuning_metrics,  # 关键：不能再使用 metrics_all
    control = tune::control_grid(
      save_pred = TRUE,
      verbose = FALSE
    )
  )
  
  notes <- tune::collect_notes(res)
  if (nrow(notes) > 0) {
    cat("\n调参提示/报错详情：\n")
    print(notes)
  }
  
  available_metrics <- tune::collect_metrics(res)
  auc_rows <- available_metrics |>
    dplyr::filter(.metric == "roc_auc", is.finite(mean))
  if (nrow(auc_rows) == 0) {
    stop("模型 ", model_name,
         " 未得到有效 ROC-AUC，请检查上面的 show_notes()。")
  }
  
  best <- tune::select_best(res, metric = "roc_auc")
  final_wf <- tune::finalize_workflow(wf, best)
  
  cat("\n最佳参数：\n")
  print(best)
  
  list(
    name = model_name,
    res = res,
    best = best,
    final_wf = final_wf
  )
}

# ============================================================
# 12. 五种模型调参
# ============================================================

# 保持前面既有数据划分及 cv_folds 不变，统一检查标签
expected_levels <- c("0", "1")
stopifnot(
  identical(levels(train_df$commercialized), expected_levels),
  identical(levels(test_df$commercialized), expected_levels)
)

for (i in seq_len(nrow(cv_folds))) {
  a <- rsample::analysis(cv_folds$splits[[i]])
  b <- rsample::assessment(cv_folds$splits[[i]])
  stopifnot(
    identical(levels(a$commercialized), expected_levels),
    identical(levels(b$commercialized), expected_levels)
  )
}

cat("\n========== 因变量类别检查 ==========\n")
print(table(train_df$commercialized))
print(table(test_df$commercialized))

set.seed(42)
log_fit <- tune_one(log_spec, log_grid, "Logistic Regression")
rf_fit  <- tune_one(rf_spec, rf_grid, "Random Forest")
xgb_fit <- tune_one(xgb_spec, xgb_grid, "XGBoost")
svm_fit <- tune_one(svm_spec, svm_grid, "SVM-RBF")
knn_fit <- tune_one(knn_spec, knn_grid, "kNN")

models <- list(log_fit, rf_fit, xgb_fit, svm_fit, knn_fit)
names(models) <- c("Logistic Regression", "Random Forest",
                   "XGBoost", "SVM-RBF", "kNN")

cat("\n五种模型调参全部完成！\n")


# ============================================================
# 13. 测试集评价函数
# ============================================================

evaluate_on_test <- function(
  fit_obj,
  test_df
) {
  
  mod_name <- fit_obj$name
  
  cat(
    "\n========== 测试模型:",
    mod_name,
    "==========\n"
  )
  
  # ----------------------------------------
  # 13.1 完整训练集训练最终模型
  # ----------------------------------------
  
  final_fit <- generics::fit(
    fit_obj$final_wf,
    data = train_df
  )
  
  # ----------------------------------------
  # 13.2 获取交叉验证 OOF 预测
  # ----------------------------------------
  
  cv_preds <- tune::collect_predictions(
    fit_obj$res,
    parameters = fit_obj$best
  )
  
  if (nrow(cv_preds) == 0) {
    stop(
      "OOF预测结果为空：",
      mod_name
    )
  }
  
  if (!".pred_1" %in% names(cv_preds)) {
    stop(
      "OOF预测结果缺少 .pred_1：",
      mod_name
    )
  }
  
  # 确保标签水平一致
  cv_preds <- cv_preds %>%
    dplyr::mutate(
      commercialized = factor(
        as.character(commercialized),
        levels = c("0", "1")
      )
    )
  
  # ----------------------------------------
  # 13.3 通过OOF预测确定最佳分类阈值
  # ----------------------------------------
  
  cv_roc <- yardstick::roc_curve(
    cv_preds,
    truth = commercialized,
    .pred_1,
    event_level = "second"
  )
  
  cv_roc <- cv_roc %>%
    dplyr::filter(
      is.finite(.threshold),
      !is.na(sensitivity),
      !is.na(specificity)
    ) %>%
    dplyr::mutate(
      youden = sensitivity +
        specificity - 1
    )
  
  if (nrow(cv_roc) == 0) {
    stop(
      "无法确定分类阈值：",
      mod_name
    )
  }
  
  # Youden Index最大化
  best_cutoff <- cv_roc$.threshold[
    which.max(cv_roc$youden)
  ]
  
  if (
    !is.finite(best_cutoff) ||
    is.na(best_cutoff)
  ) {
    stop(
      "最佳阈值异常：",
      mod_name
    )
  }
  
  cat(
    "训练阶段最佳阈值:",
    round(best_cutoff, 4),
    "\n"
  )
  
  # ----------------------------------------
  # 13.4 测试集预测概率
  # ----------------------------------------
  
  preds <- predict(
    final_fit,
    new_data = test_df,
    type = "prob"
  ) %>%
    dplyr::bind_cols(
      test_df %>%
        dplyr::select(commercialized)
    )
  
  preds <- preds %>%
    dplyr::mutate(
      commercialized = factor(
        as.character(commercialized),
        levels = c("0", "1")
      )
    )
  
  # ----------------------------------------
  # 13.5 根据最佳阈值确定预测类别
  # ----------------------------------------
  
  preds <- preds %>%
    dplyr::mutate(
      .pred_class = dplyr::if_else(
        .pred_1 >= best_cutoff,
        "1",
        "0"
      ),
      .pred_class = factor(
        .pred_class,
        levels = c("0", "1")
      )
    )
  
  # ----------------------------------------
  # 13.6 AUC
  # ----------------------------------------
  
  auc_value <- yardstick::roc_auc(
    preds,
    truth = commercialized,
    .pred_1,
    event_level = "second"
  )$.estimate
  
  # ----------------------------------------
  # 13.7 Sensitivity
  # ----------------------------------------
  
  sensitivity_value <- yardstick::sens(
    preds,
    truth = commercialized,
    estimate = .pred_class,
    event_level = "second"
  )$.estimate
  
  # ----------------------------------------
  # 13.8 Specificity
  # ----------------------------------------
  
  specificity_value <- yardstick::spec(
    preds,
    truth = commercialized,
    estimate = .pred_class,
    event_level = "second"
  )$.estimate
  
  # ----------------------------------------
  # 13.9 Accuracy
  # ----------------------------------------
  
  accuracy_value <- yardstick::accuracy(
    preds,
    truth = commercialized,
    estimate = .pred_class
  )$.estimate
  
  # ----------------------------------------
  # 13.10 汇总模型评价指标
  # ----------------------------------------
  
  metrics_tbl <- tibble::tibble(
    Model = mod_name,
    AUC = auc_value,
    Sensitivity = sensitivity_value,
    Specificity = specificity_value,
    Accuracy = accuracy_value,
    Threshold = best_cutoff
  )
  
  print(
    metrics_tbl,
    width = Inf
  )
  
  # ----------------------------------------
  # 13.11 ROC曲线
  # ----------------------------------------
  
  roc_df <- yardstick::roc_curve(
    preds,
    truth = commercialized,
    .pred_1,
    event_level = "second"
  ) %>%
    dplyr::mutate(
      Model = mod_name
    )
  
  # ----------------------------------------
  # 13.12 混淆矩阵
  # ----------------------------------------
  
  confusion_matrix <- yardstick::conf_mat(
    preds,
    truth = commercialized,
    estimate = .pred_class
  )
  
  print(confusion_matrix)
  
  # ----------------------------------------
  # 13.13 保存单模型预测结果
  # ----------------------------------------
  
  safe_model_name <- gsub(
    "[^A-Za-z0-9_]",
    "_",
    mod_name
  )
  
  write.csv(
    preds,
    file.path(
      output_dir,
      paste0(
        safe_model_name,
        "_test_predictions.csv"
      )
    ),
    row.names = FALSE
  )
  
  # ----------------------------------------
  # 13.14 返回结果
  # ----------------------------------------
  
  return(
    list(
      name = mod_name,
      metrics = metrics_tbl,
      roc_df = roc_df,
      predictions = preds,
      confusion_matrix = confusion_matrix,
      threshold = best_cutoff,
      final_fit = final_fit
    )
  )
}


# ============================================================
# 14. 五种模型独立测试集评价
# ============================================================

results <- purrr::map(
  models,
  evaluate_on_test,
  test_df = test_df
)

cat(
  "\n五种模型测试集评价完成！\n"
)


# ============================================================
# 15. 汇总四项分类评价指标
# ============================================================

all_metrics <- purrr::map_dfr(
  results,
  "metrics"
) %>%
  dplyr::arrange(
    dplyr::desc(AUC)
  )

metrics_final <- all_metrics %>%
  dplyr::select(
    Model,
    AUC,
    Sensitivity,
    Specificity,
    Accuracy
  )

cat(
  "\n========== 最终模型评价 ==========\n"
)

print(
  metrics_final,
  width = Inf
)

# ----------------------------------------
# 15.1 保存四项评价指标
# ----------------------------------------

write.csv(
  metrics_final,
  file.path(
    output_dir,
    "metrics_2.csv"
  ),
  row.names = FALSE
)

# ----------------------------------------
# 15.2 保存包含阈值的评价指标
# ----------------------------------------

write.csv(
  all_metrics,
  file.path(
    output_dir,
    "metrics_with_threshold.csv"
  ),
  row.names = FALSE
)


# ============================================================
# 16. 绘制五种模型ROC曲线
# ============================================================

all_roc <- purrr::map_dfr(
  results,
  "roc_df"
)

roc_plot <- ggplot2::ggplot(
  all_roc,
  ggplot2::aes(
    x = 1 - specificity,
    y = sensitivity,
    color = Model
  )
) +
  ggplot2::geom_line(
    linewidth = 1.1
  ) +
  ggplot2::geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "gray60"
  ) +
  ggplot2::coord_equal() +
  ggplot2::labs(
    title = "ROC Curves of Five Models",
    x = "1 - Specificity",
    y = "Sensitivity",
    color = "Model"
  ) +
  ggplot2::theme_bw(
    base_size = 13
  ) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(
      hjust = 0.5,
      face = "bold"
    ),
    legend.position = "bottom"
  )

print(roc_plot)

ggplot2::ggsave(
  filename = file.path(
    output_dir,
    "ROC_five_models.png"
  ),
  plot = roc_plot,
  width = 8,
  height = 6,
  dpi = 600
)


# ============================================================
# 17. 保存最终模型
# ============================================================

for (i in seq_along(results)) {
  
  model_name <- gsub(
    "[^A-Za-z0-9_]",
    "_",
    results[[i]]$name
  )
  
  saveRDS(
    results[[i]]$final_fit,
    file = file.path(
      output_dir,
      paste0(
        model_name,
        ".rds"
      )
    )
  )
}

cat(
  "\n最终模型保存完成！\n"
)


# ============================================================
# 18. 保存分组信息与训练测试划分
# ============================================================

split_info <- df %>%
  dplyr::select(
    dplyr::all_of(
      c(
        patent_id_col,
        paper_id_col
      )
    ),
    leakage_group,
    commercialized
  ) %>%
  dplyr::mutate(
    dataset = ifelse(
      leakage_group %in%
        train_df$leakage_group,
      "Train",
      "Test"
    )
  )

write.csv(
  split_info,
  file.path(
    output_dir,
    "patent_paper_group_split.csv"
  ),
  row.names = FALSE
)


# ============================================================
# 19. 最终数据泄漏检查
# ============================================================

cat(
  "\n========== 最终数据泄漏检查 ==========\n"
)

# ----------------------------------------
# 19.1 训练集和测试集分组检查
# ----------------------------------------

stopifnot(
  check_overlap(
    train_df,
    test_df,
    "最终训练/测试集"
  )
)

# ----------------------------------------
# 19.2 交叉验证分组检查
# ----------------------------------------

for (i in seq_len(nrow(cv_folds))) {
  
  stopifnot(
    check_overlap(
      rsample::analysis(
        cv_folds$splits[[i]]
      ),
      rsample::assessment(
        cv_folds$splits[[i]]
      ),
      paste0(
        "最终CV Fold ",
        i
      )
    )
  )
}

cat(
  "\n所有训练/测试和交叉验证分组检查通过！\n"
)

cat(
  "\n全部模型训练、评价及文件保存完成！\n"
)


