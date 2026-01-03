# Credit-fraud-detection
Credit Card Fraud Detection (Imbalanced Classification)

Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning under extreme class imbalance conditions. Fraudulent transactions represent only ~0.17% of all observations, making accuracy an unreliable metric.

The goal is to build and evaluate models that prioritize the minority (fraud) class, using appropriate preprocessing, resampling strategies, and evaluation metrics.

Dataset

Source: Kaggle – Credit Card Fraud Detection

Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Description:

European cardholder transactions from 2013

284,807 transactions

492 fraud cases (highly imbalanced)

Features V1–V28 are PCA-transformed for confidentiality

Time and Amount are raw, unscaled features

Note: The dataset is not included in this repository due to size and licensing constraints.

Key Challenges

Severe class imbalance (~1 fraud per 600 transactions)

High cost of false negatives (missed fraud)

Traditional accuracy metrics are misleading

Distance-based and linear models are sensitive to feature scaling

Methodology
1. Exploratory Data Analysis (EDA)
Examined class distribution to quantify imbalance
Visualized Time and Amount distributions
Identified need for robust scaling due to skew and outliers

2. Feature Scaling

Applied RobustScaler to Time and Amount
PCA features were already standardized by dataset design

3. Baseline Modeling

Logistic Regression with class_weight='balanced'
Established a reference point for recall and precision on fraud cases

4. Handling Class Imbalance

Compared multiple strategies:
Class-weighted Logistic Regression
SMOTE (Synthetic Minority Oversampling) + Logistic Regression
Random Forest with balanced subsampling

5. Model Evaluation

Used metrics appropriate for imbalanced classification:

Precision
Recall (Sensitivity)
F1-score
PR-AUC (Primary metric)
ROC-AUC

6. Threshold Optimization

Tuned classification thresholds based on F1 score
Demonstrated trade-off between recall and precision for fraud detection


## Results

| Model | PR-AUC |
|:------|--------:|
| Random Forest (balanced) | **0.86** |
| SMOTE + Logistic Regression | 0.73 |
| Logistic Regression (class-weighted) | 0.72 |

## Key Insights

Accuracy alone is misleading for fraud detection
PR-AUC is more informative than ROC-AUC under heavy imbalance
Class weighting is a strong baseline and often competitive with resampling
SMOTE can help linear models but may introduce noise in high dimensions
Threshold tuning is critical for real-world deployment