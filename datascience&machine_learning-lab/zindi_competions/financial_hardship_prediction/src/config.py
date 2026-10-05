"""
config.py

Project configuration file.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = PROJECT_ROOT / "models"

SUBMISSION_DIR = PROJECT_ROOT / "submissions"

# Files
TRAIN_FILE = RAW_DATA_DIR / "Train.csv"
TEST_FILE = RAW_DATA_DIR / "Test.csv"

SAMPLE_FILE = RAW_DATA_DIR / "SampleSubmission.csv"

DICTIONARY_FILE = RAW_DATA_DIR / "data_dictionary.csv"

TARGET = "liquidity_stress_next_30d"

ID = "ID"

RANDOM_STATE = 42

####################################################
# LightGBM Search Space (Wide Exploration)
LIGHTGBM_SEARCH_SPACE_V1 = {
    "learning_rate": (0.01, 0.08),
    "n_estimators": (500, 2000),
    "num_leaves": (20, 150),
    "max_depth": (4, 12),
    "min_child_samples": (10, 120),
    "min_child_weight": (1e-3, 10.0),
    "subsample": (0.60, 1.00),
    "subsample_freq": (1, 7),
    "colsample_bytree": (0.60, 1.00),
    "max_bin": (128, 512),
    "reg_alpha": (0.0, 15.0),
    "reg_lambda": (0.0, 15.0),
    "min_split_gain": (0.0, 1.0),
    "min_sum_hessian_in_leaf": (1e-3, 10.0),
    "scale_pos_weight": (1.0, 8.0),
}


####################################################
# LightGBM Search Space (Refined Search)
LIGHTGBM_SEARCH_SPACE_V2 = {
    "learning_rate": (0.015, 0.035),
    "n_estimators": (500, 1200),
    "num_leaves": (30, 70),
    "max_depth": (8, 10),
    "min_child_samples": (10, 60),
    "min_child_weight": (0.1, 8.0),
    "subsample": (0.70, 0.85),
    "subsample_freq": (1, 2),
    "colsample_bytree": (0.75, 0.92),
    "max_bin": (300, 512),
    "reg_alpha": (3.0, 7.0),
    "reg_lambda": (0.0, 15.0),
    "min_split_gain": (0.0, 0.20),
    "min_sum_hessian_in_leaf": (1e-3, 4.0),
    # Fixed after Optuna analysis
    "scale_pos_weight": 1.0,
}

####################################################
# XGBoost Search Space (Wide Exploration)
XGBOOST_SEARCH_SPACE_V1 = {
    # Learning
    "learning_rate": (0.01, 0.10),
    "n_estimators": (500, 2000),
    # Tree
    "max_depth": (3, 10),
    "min_child_weight": (1e-3, 10.0),
    "gamma": (0.0, 5.0),
    # Sampling
    "subsample": (0.60, 1.00),
    "colsample_bytree": (0.60, 1.00),
    # Regularization
    "reg_alpha": (0.0, 15.0),
    "reg_lambda": (0.0, 15.0),
    # Class imbalance
    "scale_pos_weight": (1.0, 8.0),
}

# XGBoost Search Space (Refined Search)
XGBOOST_SEARCH_SPACE_V2 = {
    "learning_rate": (0.010, 0.020),
    "n_estimators": (1000, 2200),
    "max_depth": (6, 9),
    "min_child_weight": (1.0, 8.0),
    "gamma": (0.5, 2.0),
    "subsample": (0.72, 0.85),
    "colsample_bytree": (0.85, 0.95),
    "reg_alpha": (4.0, 12.0),
    "reg_lambda": (2.0, 8.0),
    "scale_pos_weight": (1.0, 1.5),
}

