# Airline Passenger Satisfaction Prediction

A machine learning project that predicts whether an airline passenger is **satisfied** or
**neutral/dissatisfied** based on demographic data, travel profile, in-flight service ratings,
and delay information.

> Project 17 — Data Analysis with Python
> National Economics University, Hanoi — Faculty of Data Science and Artificial Intelligence

## Overview

The commercial airline industry is highly competitive, and customer satisfaction directly drives
retention, loyalty, and brand reputation. This project frames passenger satisfaction as a **binary
classification problem** and builds a full pipeline — from raw survey data to a tuned, explainable
predictive model — to identify which service factors matter most.

**Dataset:** [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) (Kaggle)
- Train set: 103,904 rows × 25 columns
- Test set: 25,976 rows × 25 columns
- 22 features across four domains: demographics, travel profile, service ratings (0–5 scale),
  and operational delays.

## Pipeline

```
Raw Data → Cleaning → EDA → Preprocessing → Modeling → Evaluation → Insights
```

1. **Data Cleaning** — column normalization, duplicate checks, outlier capping (IQR / 99th
   percentile), and type-appropriate missing value imputation (median for continuous,
   mode for ratings and categoricals). Cleaning is applied identically but independently to
   train and test splits to prevent data leakage.
2. **Exploratory Data Analysis** — descriptive statistics, target distribution, and five key
   behavioral insights (see below) supported by correlation and multicollinearity analysis.
3. **Preprocessing** — One-Hot Encoding for nominal categorical variables, binary target
   encoding, and `StandardScaler` applied only for Logistic Regression.
4. **Modeling** — three classifiers trained and compared: Logistic Regression (baseline),
   Random Forest, and XGBoost.
5. **Evaluation** — train/test performance comparison, confusion matrices, ROC-AUC analysis,
   feature importance, and an explicit overfitting diagnosis for Random Forest.

## Key Findings

| # | Insight |
|---|---|
| 1 | Cabin class × travel purpose is the strongest discriminator of satisfaction — Business travelers in Business class reach **71.4%** satisfaction, while personal-travel Economy passengers fall below **10%**. |
| 2 | Digital touchpoints (Online Boarding, Inflight Entertainment) show the largest satisfaction gaps, far outweighing physical/logistical factors like Gate Location. |
| 3 | Online Boarding rating has a near-linear relationship with satisfaction, rising from **13.8%** (rating 1) to **87.1%** (rating 5) — the single highest-leverage service touchpoint. |
| 4 | Loyalty ≠ satisfaction: despite higher satisfaction than disloyal customers, **52.4%** of loyal customers still report neutral/dissatisfied experiences. |
| 5 | Flight delays are *not* a major driver of dissatisfaction — both satisfied and dissatisfied cohorts share a median delay of 0 minutes. |

## Model Performance

| Model | Train Acc | Test Acc | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.8648 | 0.8638 | 0.8464 | 0.9241 |
| Random Forest | 1.0000 | 0.9592 | 0.9528 | 0.9924 |
| **XGBoost** | 0.9657 | **0.9596** | **0.9533** | **0.9932** |

**XGBoost** was selected as the final model — it achieves the best generalization (smallest
train-test gap among high-performing models), the highest F1-score and ROC-AUC, and provides
gain-based feature importance for interpretability.

### Top predictive features (averaged across Random Forest & XGBoost)

1. Online Boarding
2. Inflight Wifi Service
3. Type of Travel (Business)
4. Class (Business)
5. Inflight Entertainment

## Practical Recommendations

- **Prioritize digital boarding infrastructure** over physical cabin upgrades.
- **Segment service recovery programs** by cabin class and travel purpose.
- **Redesign loyalty incentives** around service quality, not just travel frequency.
- **Redirect operational focus** from punctuality metrics toward service consistency.

## Limitations

- Static, single-wave survey data — no temporal or seasonal dimension.
- Self-reported Likert-scale ratings are subject to response bias.
- No pricing or competitor context, so value-for-money cannot be assessed.
- Hyperparameters were not exhaustively tuned (no grid/Bayesian search).
- Airline identity and survey period are undisclosed by the dataset source.

## Future Improvements

- SHAP-based explainability for instance-level predictions.
- Stratified k-fold cross-validation for more robust performance estimates.
- Classification threshold tuning to better balance false negatives vs. false positives.
- Validation across multiple airlines and time periods.

