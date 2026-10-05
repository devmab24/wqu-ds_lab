# Financial Hardship Prediction --- End-to-End Machine Learning Competition Project

## Overview

This project is an end-to-end machine learning solution for a Zindi
financial hardship prediction competition.

The objective is to predict whether a customer will experience
**liquidity stress within the next 30 days**. The target is binary:

-   `0` --- no liquidity stress
-   `1` --- liquidity stress

The competition metric combines:

-   **Log Loss --- 60%**
-   **ROC-AUC --- 40%**

The project was developed as a practical ML engineering workflow rather
than as a single modeling notebook. It covers data preparation, feature
engineering, feature selection, hyperparameter optimization,
cross-validation, model evaluation, artifact management, ensemble
learning, and leaderboard validation.

------------------------------------------------------------------------

## Competition Outcome

### Best individual model

**LightGBM**

  Metric                              Result
  -------------------------- ---------------
  Local Log Loss                  `0.257963`
  Local ROC-AUC                   `0.900549`
  Local Competition Score         `0.194558`
  Public Leaderboard Score     `0.700118576`

### XGBoost

The final XGBoost model was optimized with a 200-trial Optuna study.

  Metric                                Result
  ------------------------- ------------------
  Local Competition Score           `0.195188`
  Public submission           Used in ensemble

### LightGBM + XGBoost ensemble

The two-model ensemble was optimized using out-of-fold predictions.

The final ensemble improved the public leaderboard score to:

**`0.700050246`**

This was an improvement over the LightGBM-only submission:

``` text
0.700118576 → 0.700050246
Improvement: 0.000068330
```

The leaderboard rank at the time of the latest submission was **43rd**.

> Competition rankings are dynamic and can change as other participants
> submit improved solutions.

------------------------------------------------------------------------

# 1. Problem Understanding

The task is a binary classification problem where the model estimates
the probability of future financial hardship.

The evaluation metric rewards both:

1.  well-calibrated probability predictions through Log Loss;
2.  good class discrimination through ROC-AUC.

The competition score used throughout local validation was:

``` text
Competition Score =
    0.60 × LogLoss
    + 0.40 × (1 − ROC_AUC)
```

Lower is better.

This made probability quality especially important. Optimizing only
accuracy or ROC-AUC would not have aligned with the competition
objective.

------------------------------------------------------------------------

# 2. Data Understanding

The training data contained approximately:

-   **40,000 observations**
-   **184 original features**

Initial data types included:

-   numerical variables;
-   categorical variables;
-   identifier information.

The target distribution was approximately:

``` text
Class 0: 85%
Class 1: 15%
```

The dataset was therefore imbalanced, which was considered during model
development.

------------------------------------------------------------------------

# 3. Feature Engineering

A major part of the project was transforming the original financial
transaction variables into a richer feature representation.

Feature engineering expanded the feature space substantially by creating
derived variables from monthly financial behavior.

Examples of engineered feature groups included:

-   Paybill activity
-   Merchant payments
-   Bank transfers
-   Mobile-money transfers
-   Received funds
-   Deposits
-   Withdrawals
-   Daily average balance

The engineered representation was subsequently reduced through
correlation analysis and feature selection.

This process resulted in a selected feature set of:

## **Top 137 encoded features**

The selected features were stored separately so that the exact same
feature space could be reproduced during final training and inference.

------------------------------------------------------------------------

# 4. Model Development

The project evaluated multiple gradient-boosting approaches.

## LightGBM

LightGBM was used as the primary model and tuned using Optuna.

The final model was evaluated using 5-fold cross-validation.

Final local result:

``` text
LogLoss           : 0.257963
ROC-AUC           : 0.900549
Competition Score : 0.194558
```

The LightGBM model became the strongest individual local model and
produced the initial competitive public submission.

------------------------------------------------------------------------

## XGBoost

XGBoost was subsequently optimized using Optuna.

A 200-trial optimization study was used, with pruning enabled.

The optimization produced the following best parameters:

``` python
{
    "learning_rate": 0.014962779909537652,
    "n_estimators": 1703,
    "max_depth": 9,
    "min_child_weight": 5.015957391190683,
    "gamma": 0.9014218177128371,
    "subsample": 0.7917087626295457,
    "colsample_bytree": 0.8500729945267632,
    "reg_alpha": 11.48700705819626,
    "reg_lambda": 7.900214710998145,
    "scale_pos_weight": 1.1405025746396893
}
```

Local competition score:

``` text
0.195188
```

Although XGBoost was slightly weaker than LightGBM individually, it was
retained because ensemble performance depends on complementary
predictions, not only individual model scores.

------------------------------------------------------------------------

# 5. Hyperparameter Optimization with Optuna

Optuna was used to systematically search the model hyperparameter space.

The XGBoost optimization demonstrated:

-   200+ optimization trials;
-   pruning of unpromising trials;
-   persistent study storage;
-   recovery/resumption of long-running experiments;
-   analysis of parameter importance;
-   analysis of best-trial parameter distributions.

One XGBoost study completed with:

``` text
Best Score : 0.194846
```

The optimization process took approximately:

``` text
45,951 seconds
```

This experience reinforced several practical ML engineering lessons:

-   long-running experiments should use persistent storage;
-   pruning can substantially reduce wasted computation;
-   experiment state should not depend on a single notebook session;
-   model selection should be based on the competition metric rather
    than intuition alone.

------------------------------------------------------------------------

# 6. Cross-Validation and OOF Predictions

A reusable evaluation engine was developed around cross-validation.

For each fold the pipeline:

1.  splits the data;
2.  clones the estimator;
3.  trains the model;
4.  generates validation probabilities;
5.  calculates Log Loss;
6.  calculates ROC-AUC;
7.  calculates the competition score;
8.  stores the trained fold model;
9.  records feature importance.

The evaluation pipeline also produces **out-of-fold (OOF) predictions**.

OOF predictions were particularly important for ensemble development
because they provide predictions for every training observation without
using that observation to train its corresponding fold model.

------------------------------------------------------------------------

# 7. ML Artifact Management

The evaluation workflow was designed to preserve reproducible experiment
artifacts.

The project stores artifacts such as:

``` text
outputs/
├── models/
├── predictions/
├── metrics/
├── importance/
└── optuna/
```

Examples include:

-   trained fold models;
-   OOF predictions;
-   test predictions;
-   cross-validation metrics;
-   experiment summaries;
-   feature importance;
-   Optuna study databases.

This separation makes it possible to perform downstream experiments
without unnecessarily retraining models.

------------------------------------------------------------------------

# 8. Ensemble Learning

After obtaining strong individual models, the next step was to
investigate whether their predictions contained complementary
information.

The LightGBM and XGBoost OOF predictions had a correlation of
approximately:

``` text
0.99129
```

This is very high, meaning the two models make highly similar
predictions.

Nevertheless, the ensemble produced a measurable improvement.

The ensemble framework supports:

-   simple averaging;
-   weighted averaging;
-   prediction correlation analysis;
-   diversity analysis;
-   strategy evaluation;
-   grid search over ensemble weights.

The best weights were obtained from the OOF predictions using a grid
search.

The search minimized the competition score rather than maximizing it.

------------------------------------------------------------------------

# 9. Public Leaderboard Validation

The local ensemble was converted into a submission using test-set
predictions generated from the saved fold models.

The LightGBM + XGBoost ensemble achieved:

``` text
Public Score: 0.700050246
```

compared with:

``` text
LightGBM-only: 0.700118576
```

This confirmed that the ensemble provided a real public leaderboard
improvement.

------------------------------------------------------------------------

# 10. What I Learned

This project provided practical experience beyond simply training
classification models.

### Machine Learning

-   Binary classification
-   Class imbalance
-   Probability prediction
-   Log Loss optimization
-   ROC-AUC evaluation
-   Feature engineering
-   Feature selection
-   Gradient boosting
-   Model comparison
-   Ensemble learning

### Model Optimization

-   Optuna
-   Hyperparameter search spaces
-   Trial pruning
-   Parameter importance analysis
-   Persistent Optuna studies
-   Resuming long-running optimization jobs

### ML Engineering

-   Modular Python project structure
-   Reusable evaluation functions
-   Cross-validation pipelines
-   OOF prediction generation
-   Model artifact persistence
-   Reproducible experiment outputs
-   Separation of training, evaluation, and ensemble logic

### Competition Engineering

-   Public leaderboard interpretation
-   Local CV vs. public leaderboard analysis
-   Submission management
-   Ensemble validation
-   Experiment tracking
-   Iterative model development under computational constraints

------------------------------------------------------------------------

# 11. Project Architecture

A simplified version of the project structure is:

``` text
financial_hardship_prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── data exploration
│   ├── feature engineering
│   ├── feature selection
│   ├── LightGBM experiments
│   ├── XGBoost experiments
│   └── model ensemble
│
├── src/
│   ├── config.py
│   ├── wrangle.py
│   ├── feature_engineering.py
│   ├── validation.py
│   ├── cv.py
│   ├── model_evaluation.py
│   ├── ensemble.py
│   └── ...
│
├── outputs/
│   ├── models/
│   ├── predictions/
│   ├── metrics/
│   ├── importance/
│   └── optuna/
│
└── submissions/
```

------------------------------------------------------------------------

# 12. Reproducibility

The workflow was designed so that final models and predictions can be
reproduced without rerunning expensive hyperparameter optimization.

The general pipeline is:

``` text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Top 137 Features
   ↓
Optuna Hyperparameter Optimization
   ↓
Final Model Parameters
   ↓
5-Fold Cross-Validation
   ↓
Saved Fold Models + OOF Predictions
   ↓
Ensemble Optimization
   ↓
Test Predictions
   ↓
Submission
```

------------------------------------------------------------------------

# 13. Future Experiments

The next planned experiments are:

### CatBoost

Train and tune CatBoost using the same 137-feature representation and
evaluation pipeline.

### Three-model ensemble

Optimize:

``` text
LightGBM + XGBoost + CatBoost
```

and determine whether CatBoost provides enough prediction diversity to
improve the leaderboard.

### Pseudo-labeling

Only investigate pseudo-labeling if the simpler ensemble reaches a
plateau and the leaderboard results justify the additional complexity.

### Feature selection

Further feature-selection experiments will only be pursued if they
demonstrate measurable improvement over the current Top-137
representation.

### Stacking

Stacking will be considered only after simpler weighted ensembles have
been exhausted.

------------------------------------------------------------------------

# 14. Why this project matters professionally

This project demonstrates more than the ability to call a
machine-learning library.

It demonstrates an understanding of the complete modeling lifecycle:

``` text
Problem Definition
      ↓
Data Understanding
      ↓
Feature Engineering
      ↓
Feature Selection
      ↓
Model Development
      ↓
Hyperparameter Optimization
      ↓
Cross-Validation
      ↓
Experiment Tracking
      ↓
Artifact Management
      ↓
Ensemble Learning
      ↓
Submission
      ↓
Leaderboard Validation
```

The strongest engineering aspect of the project is the transition from
isolated experiments to a reusable ML experimentation framework.

------------------------------------------------------------------------

# 15. Recommended presentation

For a portfolio or GitHub repository, this project should be presented
primarily as a **machine-learning competition / applied ML engineering
project**.

An inference UI is optional.

A Streamlit application could be added later to demonstrate how the
final model can be used to generate a financial-hardship probability for
a single customer. However, it should not replace the competition
workflow or imply that the competition model is automatically a
production financial decision system.

For recruiter-facing purposes, the current competition framing is
actually valuable because it clearly demonstrates:

-   metric-driven model development;
-   experimentation;
-   optimization;
-   cross-validation;
-   ensemble learning;
-   reproducibility;
-   computational trade-offs;
-   leaderboard validation.

If an inference UI is eventually added, it should be positioned as a
**technical demonstration / model inference interface**, not as a real
financial decision-making product.

------------------------------------------------------------------------

# 16. Author

**Muhammad Ayuba Bello**

Computer Science Graduate \| Data Science & Machine Learning \| Software
Engineering

This project forms part of an ongoing portfolio focused on practical
machine learning, data science, and ML engineering.
