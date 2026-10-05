# This module contains functions that return configured model objects.
"""
Model factory
"""
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# LightGBM_Default (I'll create 2 seperate LightGBM: Default & Balanced to test with )
def get_lightgbm_default(random_state=42):
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=0.10,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )

# LightGBM_Balanced: This is identical except that it automatically compensates for the minority class.
def get_lightgbm_balanced(random_state=42):
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=0.10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
# NOTE: Alternatively, after experimentation you can replace class_weight="balanced" with a tuned scale_pos_weight, but "balanced" is a strong starting point.

# CatBoost_Default
def get_catboost_default(random_state=42):
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        bootstrap_type="Bayesian",
        random_seed=random_state,
        verbose=False,
    )

# CatBoost_Balanced
"""
CatBoost doesn't use class_weight="balanced" directly. Instead, it expects explicit class weights.

Since your dataset is:

Class 0 → 85%
Class 1 → 15%

the ratio is approximately: 85 / 15 ≈ 5.67
"""
def get_catboost_balanced(class_weights, random_state=42):
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        bootstrap_type="Bayesian",
        class_weights=class_weights,
        random_seed=random_state,
        verbose=False
    )

# XGBoost
def get_xgboost(random_state=42):
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
    )

#HistGradientBoosting
def get_histgb(random_state=42):
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=500,
        max_leaf_nodes=31,
        random_state=random_state,
    )

#ExtraTreesClassifier
def get_extra_trees(random_state=42):
    return ExtraTreesClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )

#Random Forest
def get_random_forest(random_state=42):
    return RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )

#LogisticRegression
def get_logistic(random_state=42):
    return LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=random_state,
    )