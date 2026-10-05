"""
Centralized model configurations.

Every model in the project should be instantiated
from this file to ensure consistency between:

- Optuna tuning
- Final training
- Cross-validation
- Ensembling
"""

from config import RANDOM_STATE


# ==========================================================
# LightGBM
LIGHTGBM_DEFAULTS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# XGBoost
XGBOOST_DEFAULTS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# CatBoost
CATBOOST_DEFAULTS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "random_seed": RANDOM_STATE,
    "verbose": False,
}