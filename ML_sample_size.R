# Script owner: Marie Pittet
# Date: 13.01.2026

# This script was written in te context of sample size planning for
# predictive modeling with hierarchical / repeated-measures data.
#
# It simulates: N participants × K items (long format), with mixed predictor types
# (person-level covariates, item-level/task-like features, and noise variables),
# then evaluates out-of-sample performance using a participant-level split to
# prevent leakage.
#
# Note: All data are synthetic; parameter values are illustrative.


# Set-up ------------------------------------------------------------------
library(pacman)
pacman::p_load(tidyverse, glmnet, ranger, xgboost, nnet, scales, ggplot)


# Generating datasets (function)-------------------------------------------
gen_data <- function(n_subs, n_items=24) {
  
  # Person predictors (10 traits/states per person)
  person_traits <- matrix(rnorm(n_subs * 10), ncol=10)
  colnames(person_traits) <- paste0("Trait_", 1:10)
  subjects <- data.frame(ID = 1:n_subs, person_traits)
  
  # Item predictors (each person sees n items)
  grid <- expand.grid(ID = subjects$ID, Item = 1:n_items)
  
  # Simulating 20 task metrics that are actually predictive (like RTs, accuracy, etc)
  item_metrics <- matrix(rnorm(nrow(grid) * 20), ncol=20)
  colnames(item_metrics) <- paste0("Task_", 1:20)
  
  # Binding everything together
  df <- bind_cols(grid, as.data.frame(item_metrics)) %>%
    left_join(subjects, by = "ID")
  
  # We add 50 noise random variables (which is often the case in psychological studies), a blend of unuseful personality and task metrics)
  noise_matrix <- matrix(rnorm(nrow(df) * 50), ncol=50)
  colnames(noise_matrix) <- paste0("Noise_", 1:50)
  df <- bind_cols(df, as.data.frame(noise_matrix))
  
  # Starting to build y from 4 predictors (main and interactions) as would realistically
  # be the case in real-life. 
  signal <- 0.5 * df$Task_1 +         # Main driver
    0.3 * df$Task_2 +                 # Secondary driver
    0.4 * (df$Trait_1 * df$Task_1) +  # Interaction
    0.2 * df$Trait_2
  
  # Building y from the signal + some random static (sd=1.2) because humans are never 100% consistent. 
  # sd=1.2 puts us in the 0.20 R2 range
  df$y_raw <- signal + rnorm(nrow(df), sd=1.2)
  
  # Scaling y to have the relative y outcome and not absolute. Removes personal bias.
  df <- df %>% 
    group_by(ID) %>% 
    mutate(y = as.numeric(scale(y_raw, scale=FALSE))) %>% 
    ungroup() %>%
    select(-y_raw)
  
  return(df)
}


# Simulation loop -----------------------------------------------------------
  run_sim <- function(n) {
    
    # generating the data for n
    df <- gen_data(n)
  
  # Train-test split (py people to avoid people leakage)
  train_ids <- sample(unique(df$ID), 0.8 * n)
  train <- df %>% filter(ID %in% train_ids)
  test  <- df %>% filter(!ID %in% train_ids)
  
  x_cols <- setdiff(names(train), c("ID", "Item", "y"))
  X_train <- as.matrix(train[, x_cols])
  y_train <- train$y
  X_test  <- as.matrix(test[, x_cols])
  y_test  <- test$y
  
  # MODELS
  # Elastic Net
  fit_en <- cv.glmnet(X_train, y_train, alpha=0.5)
  pred_en <- predict(fit_en, X_test, s="lambda.min")
  r2_en <- cor(y_test, pred_en)^2
  
  # Random Forest
  fit_rf <- ranger(y ~ ., data = train %>% select(-ID, -Item), num.trees=100)
  pred_rf <- predict(fit_rf, test)$predictions
  r2_rf <- cor(y_test, pred_rf)^2
  
  # XGBoost
  dtrain <- xgb.DMatrix(X_train, label=y_train)
  params <- list(objective = "reg:squarederror", max_depth = 4, learning_rate = 0.1)
  fit_xgb <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
  pred_xgb <- predict(fit_xgb, X_test)
  r2_xgb <- cor(y_test, pred_xgb)^2
  
  # Simple Neural Net 
  fit_nn <- nnet::nnet(X_train, y_train, size=5, decay=0.1, linout=TRUE, maxit=200, trace=FALSE)
  pred_nn <- predict(fit_nn, X_test)
  r2_nn <- cor(y_test, pred_nn)^2
  
  # Baseline (shuffled the outcome to have a chance benchmark that other models have to beat)
  y_shuffled <- sample(y_train)
  fit_null <- cv.glmnet(X_train, y_shuffled, alpha=0.5)
  pred_null <- predict(fit_null, X_test, s="lambda.min")
  r2_null <- cor(y_test, pred_null)^2
  if(is.na(r2_null)) r2_null <- 0 
  
  return(data.frame(N=n, 
                    ElasticNet = r2_en,
                    RandomForest = r2_rf, 
                    XGBoost = r2_xgb,
                    NeuralNetwork = r2_nn,
                    Baseline_Shuffled = r2_null))
  }


# Execution ---------------------------------------------------------------
print("Running the power analysis. Grab a coffee or something.")
n_steps <- c(50, 100, 200, 400, 800) 

# Increase reps to 30 for smoothness
results <- map_dfr(n_steps, function(n) {
  map_dfr(1:30, ~run_sim(n))
})


# Visualization -----------------------------------------------------------
results_long <- results %>% 
  pivot_longer(-N, names_to="Model", values_to="R2") %>%
  mutate(Model = case_when(
    Model == "lambda.min" ~ "ElasticNet",
    TRUE ~ Model
  )) %>%
  filter(Model %in% c("ElasticNet", "RandomForest", "XGBoost", "NeuralNetwork", "Baseline_Shuffled"))

ggplot(results_long, aes(x=N, y=R2, color=Model)) +
  stat_summary(fun=mean, geom="line", size=1.5) +
  stat_summary(fun=mean, geom="point", size=3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 10) +
  
  scale_color_manual(values = c(
    "ElasticNet"        = "#d62728", 
    "RandomForest"      = "#2ca02c", 
    "XGBoost"           = "#1f77b4", 
    "NeuralNetwork"     = "#9467bd", 
    "Baseline_Shuffled" = "gray70"
  )) +
  
  geom_hline(yintercept = 0, linetype="dashed", color="gray50") +
  labs(title = "Learning curves (simulated data with 80 predictors)",
       subtitle = "Models stabilize at N=400 with R² ~ 0.20",
       y = "Predictive Accuracy (R²)", 
       x = "Number of Participants") +
  theme_minimal() +

  theme(legend.position = "right")
