##LAPTOP PRICE ANALYSIS

#install necessary packages and libraries
library(tidyverse)
library(dplyr)
library(caret)
library(ggplot2)
library(randomForest)
library(gbm)
library(Metrics)
library(lubridate)
library(corrplot)

#load the data

laptop_data <- read.csv("C:/Users/abdul/Downloads/laptop_prices.csv")

#Data Preview

head(laptop_data)                #overview of first few rows
colSums(is.na(laptop_data))      #check any missing value
dim(laptop_data)                 #data dimention
str(laptop_data)                 #show each column and its contetnt
summary(laptop_data)             #dataset summary
sum(duplicated(laptop_data))     #check duplicate values

#EDA

#Laptop price distribution
ggplot(laptop_data, aes(x = Price_euros)) +
  geom_histogram(aes(fill = ..count..), bins = 30) +
  scale_fill_gradient(low = "#90CAF9", high = "#0D47A1") +
  labs(
    title = "Distribution of Laptop Prices",
    x = "Price (Euros)",
    y = "Count"
  ) +
  theme_minimal()

#Laptops by company
ggplot(laptop_data, aes(x = Company, fill = Company)) +
  geom_bar(show.legend = FALSE) +
  labs(
    title = "Number of Laptops by Manufacturer",
    x = "Company",
    y = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Laptop RAM distribution
ggplot(laptop_data, aes(x = factor(Ram), fill = factor(Ram))) +
  geom_bar(show.legend = FALSE) +
  labs(
    title = "Distribution of RAM Sizes",
    x = "RAM (GB)",
    y = "Count"
  ) +
  theme_minimal()

#OS Distribution
ggplot(laptop_data, aes(x = OS, fill = OS)) +
  geom_bar(show.legend = FALSE) +
  labs(
    title = "Operating System Distribution",
    x = "Operating System",
    y = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Price vs Ram
ggplot(laptop_data, aes(x = factor(Ram), y = Price_euros, fill = factor(Ram))) +
  geom_boxplot(show.legend = FALSE) +
  labs(
    title = "Laptop Price vs RAM",
    x = "RAM (GB)",
    y = "Price (Euros)"
  ) +
  theme_minimal()

#Price vs OS
ggplot(laptop_data, aes(x = OS, y = Price_euros, fill = OS)) +
  geom_boxplot(show.legend = FALSE) +
  labs(
    title = "Laptop Price vs Operating System",
    x = "Operating System",
    y = "Price (Euros)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Price vs Touchscreen
ggplot(laptop_data, aes(x = factor(Touchscreen), y = Price_euros, fill = factor(Touchscreen))) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_manual(values = c("#FFCC80", "#FF7043")) +
  labs(
    title = "Laptop Price vs Touchscreen Availability",
    x = "Touchscreen (0 = No, 1 = Yes)",
    y = "Price (Euros)"
  ) +
  theme_minimal()

#Price vs Primary storage type
ggplot(laptop_data, aes(x = PrimaryStorageType, y = Price_euros, fill = PrimaryStorageType)) +
  geom_boxplot(show.legend = FALSE) +
  labs(
    title = "Laptop Price vs Primary Storage Type",
    x = "Primary Storage Type",
    y = "Price (Euros)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Price vs CPU company
ggplot(laptop_data, aes(x = CPU_company, y = Price_euros, fill = CPU_company)) +
  geom_boxplot(show.legend = FALSE) +
  labs(
    title = "Laptop Price vs CPU Manufacturer",
    x = "CPU Company",
    y = "Price (Euros)"
  ) +
  theme_minimal()

#Outlier Visualization
ggplot(laptop_data, aes(y = Price_euros)) +
  geom_boxplot(fill = "#81C784", color = "#1B5E20") +
  labs(
    title = "Outliers in Laptop Prices",
    y = "Price (Euros)"
  ) +
  theme_minimal()

#Feature Engineering
#do required conversions and mutations
laptop_data <- laptop_data %>%
  mutate(ScreenResolution = ScreenW * ScreenH)
laptop_data <- laptop_data %>%
  mutate(TotalStorage = PrimaryStorage + SecondaryStorage)
laptop_data <- laptop_data %>%
  mutate(
    Touchscreen   = ifelse(Touchscreen == 1, 1, 0),
    IPSpanel      = ifelse(IPSpanel == 1, 1, 0),
    RetinaDisplay = ifelse(RetinaDisplay == 1, 1, 0)
  )
factor_cols <- c(
  "Company", "TypeName", "OS", "Screen",
  "CPU_company", "CPU_model",
  "PrimaryStorageType", "SecondaryStorageType",
  "GPU_company", "GPU_model"
)

laptop_data[factor_cols] <- lapply(
  laptop_data[factor_cols],
  as.factor
)

#remove redundant columns
laptop_data <- laptop_data %>%
  select(-ScreenW, -ScreenH)

#create test-train split
set.seed(123)

train_index <- createDataPartition(
  laptop_data$Price_euros,
  p = 0.8,
  list = FALSE
)

train_data <- laptop_data[train_index, ]
test_data  <- laptop_data[-train_index, ]

#separate features and target variables
x_train <- train_data %>% select(-Price_euros)
y_train <- train_data$Price_euros

x_test  <- test_data %>% select(-Price_euros)
y_test  <- test_data$Price_euros

#scaling numerical variables
numeric_cols <- c(
  "Inches", "Ram", "Weight", "CPU_freq",
  "PrimaryStorage", "SecondaryStorage",
  "ScreenResolution", "TotalStorage"
)

valid_numeric_cols <- numeric_cols[
  numeric_cols %in% colnames(x_train) &
    sapply(x_train[numeric_cols], is.numeric)
]
train_scaled <- scale(x_train[valid_numeric_cols])

train_center <- attr(train_scaled, "scaled:center")
train_scale  <- attr(train_scaled, "scaled:scale")

# Apply to train
x_train[valid_numeric_cols] <- train_scaled

# Apply to test using TRAIN parameters
x_test[valid_numeric_cols] <- scale(
  x_test[valid_numeric_cols],
  center = train_center,
  scale  = train_scale
)
preproc <- preProcess(
  x_train[valid_numeric_cols],
  method = c("center", "scale")
)

x_train[valid_numeric_cols] <- predict(preproc, x_train[valid_numeric_cols])
x_test[valid_numeric_cols]  <- predict(preproc, x_test[valid_numeric_cols])

#remoeve columns with high cardinality
high_cardinality <- c("Product", "CPU_model", "GPU_model")

x_train <- x_train %>% select(-all_of(high_cardinality))
x_test  <- x_test %>% select(-all_of(high_cardinality))


#Model building and evaluation

#train-control setup
set.seed(123)

train_control <- trainControl(
  method = "cv",
  number = 5
)

#LRM Baseline
set.seed(123)

lm_model <- train(
  x = x_train,
  y = y_train,
  method = "lm",
  trControl = train_control
)
head(x_train)
lm_model
#predictions and metrics
lm_pred <- predict(lm_model, x_test)

lm_rmse <- rmse(y_test, lm_pred)
lm_mae  <- mae(y_test, lm_pred)
lm_r2   <- R2(lm_pred, y_test)

#RF Model
set.seed(123)

rf_model <- train(
  x = x_train,
  y = y_train,
  method = "rf",
  trControl = train_control,
  tuneLength = 5,
  importance = TRUE
)

rf_model
#predictions and metrics
rf_pred <- predict(rf_model, x_test)

rf_rmse <- rmse(y_test, rf_pred)
rf_mae  <- mae(y_test, rf_pred)
rf_r2   <- R2(rf_pred, y_test)

#Gradient boost model
set.seed(123)

gbm_model <- train(
  x = x_train,
  y = y_train,
  method = "gbm",
  trControl = train_control,
  verbose = FALSE
)
gbm_model
#predictions and metrics
gbm_pred <- predict(gbm_model, x_test)

gbm_rmse <- rmse(y_test, gbm_pred)
gbm_mae  <- mae(y_test, gbm_pred)
gbm_r2   <- R2(gbm_pred, y_test)

#Model Performance Comparision
model_results <- data.frame(
  Model = c("Linear Regression", "Random Forest", "Gradient Boosting"),
  RMSE  = c(lm_rmse, rf_rmse, gbm_rmse),
  MAE   = c(lm_mae, rf_mae, gbm_mae),
  R2    = c(lm_r2, rf_r2, gbm_r2)
)

model_results

#rmse comparison plot
ggplot(model_results, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(RMSE, 1)), vjust = -0.5, size = 4) +
  labs(
    title = "Model Performance Comparison (RMSE)",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

#MAE comparison PLot
ggplot(model_results, aes(x = Model, y = MAE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(MAE, 1)), vjust = -0.5, size = 4) +
  labs(
    title = "Model Performance Comparison (MAE)",
    x = "Model",
    y = "MAE"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

#R square comparioson plot
ggplot(model_results, aes(x = Model, y = R2, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(R2, 3)), vjust = -0.5, size = 4) +
  labs(
    title = "Model Performance Comparison (R²)",
    x = "Model",
    y = "R²"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

#Feature Impoortance

rf_importance <- varImp(rf_model, scale = TRUE)

rf_importance

rf_imp_df <- rf_importance$importance
rf_imp_df$Feature <- rownames(rf_imp_df)

ggplot(rf_imp_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Feature Importance from RF Model",
    x = "Features",
    y = "Importance Score"
  ) +
  theme_minimal(base_size = 13)

## Project Completed