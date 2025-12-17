# Laptop-Price-Analysis
Developed an end-to-end machine learning solution in R to analyze and predict laptop prices based on hardware and system specifications. Implemented EDA, feature engineering, and ensemble regression models, with Random Forest delivering the best performance (R² ≈ 0.84) and revealing key pricing drivers.

# 💻 Laptop Price Analysis & Prediction (R)

## 📌 Project Overview
This project focuses on analyzing and predicting laptop prices using machine learning techniques implemented in **R**. By leveraging detailed laptop hardware specifications, the study aims to identify the key factors influencing laptop prices and to build an accurate and reliable predictive model. The project follows a complete end-to-end data science workflow, from data preprocessing and exploratory analysis to model development, evaluation, and interpretation.

---

## 🎯 Objectives
- Analyze laptop pricing trends using exploratory data analysis (EDA)
- Perform feature engineering to convert raw specifications into meaningful predictors
- Build and compare multiple regression-based machine learning models
- Identify the most influential factors affecting laptop prices
- Develop a robust model capable of accurately predicting laptop prices

---

## 📂 Dataset Description
The dataset contains information on laptop hardware and system features, including:
- Brand and laptop type
- Screen size and resolution
- RAM capacity and storage configuration
- CPU and GPU manufacturer details
- Operating system and physical attributes

High-cardinality identifiers such as specific product, CPU model, and GPU model names were excluded during modeling to improve generalization.

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was conducted to understand data distributions, detect outliers, and analyze relationships between features and laptop prices. Key insights include:
- Higher RAM, CPU frequency, and SSD storage are strongly associated with higher prices
- Laptops with high-resolution displays and dedicated GPUs tend to be priced higher
- Pricing patterns exhibit non-linear relationships, motivating the use of ensemble models

Visualizations were created using `ggplot2` to support analytical findings.

---

## ⚙️ Feature Engineering
Feature engineering steps included:
- Creation of binary indicators for touchscreen, IPS panel, and Retina display
- Aggregation of total storage from primary and secondary storage components
- Transformation of screen resolution into numerical form
- Scaling of numerical features after train-test splitting
- Removal of high-cardinality categorical variables to prevent overfitting

---

## 🤖 Models Implemented
The following regression models were developed and evaluated using **5-fold cross-validation**:

- **Linear Regression** (Baseline Model)
- **Gradient Boosting Regression (GBM)**
- **Random Forest Regression**

The `caret` framework was used for model training, hyperparameter tuning, and performance comparison.

---

## 📊 Model Performance Summary

| Model               | RMSE   | MAE    | R²   |
|---------------------|--------|--------|------|
| Linear Regression   | 336.07 | 238.67 | 0.78 |
| Gradient Boosting   | 304.05 | 210.81 | 0.82 |
| **Random Forest**   | **291.29** | **191.39** | **0.84** |

Random Forest Regression achieved the best overall performance and was selected as the final model.

---

## 🧠 Feature Importance Insights
Feature importance analysis from the Random Forest model revealed that laptop prices are primarily influenced by:
- RAM capacity
- CPU frequency
- Total storage and storage type
- Screen resolution
- GPU manufacturer

Performance-oriented hardware specifications were found to be more influential than brand-level identifiers.

---

## ✅ Model Validation
Model validation through residual analysis and actual vs predicted price plots showed:
- Strong alignment between actual and predicted prices
- Residuals centered around zero with no major bias
- Good generalization performance on unseen data

---

## 🛠 Tools & Technologies
- **Programming Language:** R  
- **Libraries:** tidyverse, dplyr, ggplot2, caret, randomForest, gbm, Metrics  
- **Techniques:** EDA, feature engineering, regression modeling, ensemble learning, cross-validation

---

## 🚀 Conclusion
This project demonstrates the effective application of machine learning techniques to predict laptop prices using structured hardware data. Ensemble-based models, particularly Random Forest, significantly outperformed linear approaches, highlighting the importance of non-linear modeling for pricing problems. The insights derived from this study can support data-driven pricing strategies and product analysis.

---

## 🔮 Future Scope
- Incorporate additional features such as customer ratings and market demand
- Explore advanced ensemble techniques and hyperparameter optimization
- Deploy the model as a web-based pricing recommendation tool
- Extend the analysis to other consumer electronics categories

---
## 📄 Author
**Abdul Razzaq**  
Master’s in Statistics | Aspiring Data Scientist
