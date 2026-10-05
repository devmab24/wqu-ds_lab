"""
evaluate.py
Model Evaluation Engine
-----------------------
Features
--------
✓ Cross Validation
✓ Automatic predict_proba / decision_function
✓ Automatic CatBoost support
✓ Automatic Feature Importance
✓ Automatic Model Saving
✓ Out-of-Fold Predictions
✓ Competition Metric
✓ Experiment Tracking
✓ Training Time Logging

Designed for:
- LightGBM
- CatBoost
- XGBoost
- RandomForest
- ExtraTrees
- HistGradientBoosting
- Logistic Regression
"""

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import time
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)

from copy import deepcopy
from sklearn.base import clone

# Result Object
@dataclass
class EvaluationResult:
    model_name: str
    fold_scores: pd.DataFrame
    oof_predictions: np.ndarray
    trained_models: list
    feature_importance: pd.DataFrame | None
    overall_logloss: float
    overall_auc: float
    overall_score: float
    training_time: float
    timestamp: str

# Probability Helper
def _predict_probability(model, X):
    """
    Returns prediction probabilities regardless
    of estimator implementation.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.clip(scores, -500, 500)
        return 1 / (1 + np.exp(-scores))
    raise ValueError(
        "Estimator must implement predict_proba() "
        "or decision_function()."
    )

# Model Fitting Helper
def _fit_model(
    estimator,
    X_train,
    y_train,
    categorical_features=None,
):
    """
    Automatically detects CatBoost models.
    """
    model_name = estimator.__class__.__name__
    if model_name.startswith("CatBoost"):
        estimator.fit(
            X_train,
            y_train,
            cat_features=categorical_features,
            verbose=False,
        )
    else:
        estimator.fit(
            X_train,
            y_train,
        )

    return estimator

# Feature Importance Helper
def _get_feature_importance(
    estimator,
    feature_names,
):
    """
    Returns feature importance DataFrame
    if supported by the estimator.
    """
    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": estimator.feature_importances_
        })

    return None

# Output Directory Helper
def _create_output_directories():
    """
    Create all output directories used by the evaluation pipeline.
    """

    output_dir = Path("../outputs")

    model_dir = output_dir / "models"
    prediction_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    importance_dir = output_dir / "importance"

    for directory in [
        output_dir,
        model_dir,
        prediction_dir,
        metrics_dir,
        importance_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return {
        "output": output_dir,
        "models": model_dir,
        "predictions": prediction_dir,
        "metrics": metrics_dir,
        "importance": importance_dir,
    }


#Evaluate model (core engine)
def evaluate_model(model, X, y, cv, categorical_features=None):
    """
    Evaluate a model using cross validation.
    
    Parameters
    ----------
    model : sklearn estimator
    X : pd.DataFrame
    y : pd.Series
    cv : sklearn CV splitter
    categorical_features : list | None
    Returns
    -------
    EvaluationResult
    """

    # Initialization
    start_time = time.time()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    dirs = _create_output_directories()
    
    # Create artifact directories
    output_dir = dirs["output"]
    model_dir = dirs["models"]
    prediction_dir = dirs["predictions"]
    oof_dir = prediction_dir / "oof"
    test_dir = prediction_dir / "test"
    oof_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    metrics_dir = dirs["metrics"]
    importance_dir = dirs["importance"]

    model_name = model.__class__.__name__

    print("=" * 70)
    print(f"Model : {model_name}")
    print("=" * 70)

    # Containers
    oof_predictions = np.zeros(len(y))
    trained_models = []
    fold_scores = []
    feature_importance = []

    # Cross Validation
    for fold, (train_idx, valid_idx) in enumerate( cv.split(X, y), start=1):
        fold_start = time.time()

        print(f"\nFold {fold}")
        print("-" * 60)

        # Split
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        # Clone estimator
        estimator = _clone_estimator(model)

        # Train
        estimator = _fit_model(
            estimator,
            X_train,
            y_train,
            categorical_features,
        )

        # Save model in memory
        trained_models.append(estimator)
        
        # Create model-specific folder
        model_output_dir = model_dir / model_name
        model_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        
        joblib.dump(
            estimator,
            model_output_dir / f"fold_{fold}.pkl",
        )

        # Predict
        preds = _predict_probability(
            estimator,
            X_valid,
        )

        oof_predictions[valid_idx] = preds

        # Metrics
        logloss = log_loss(
            y_valid,
            preds,
        )

        auc = roc_auc_score(
            y_valid,
            preds,
        )

        competition_score = (
            0.60 * logloss + 0.40 * (1 - auc)
        )

        fold_time = time.time() - fold_start

        # Fold Results
        fold_scores.append(
            {
                "Fold": fold,
                "LogLoss": logloss,
                "ROC_AUC": auc,
                "CompetitionScore": competition_score,
                "TimeSeconds": fold_time,
            }
        )

        # Console Output
        print(f"LogLoss           : {logloss:.6f}")
        print(f"ROC-AUC           : {auc:.6f}")
        print(f"Competition Score : {competition_score:.6f}")
        print(f"Time              : {fold_time:.2f} sec")

        # Feature Importance
        importance = _get_feature_importance(
            estimator,
            X.columns,
        )

        if importance is not None:
            importance["Fold"] = fold
            feature_importance.append(importance)

    # Overall Metrics
    overall_logloss = log_loss( y, oof_predictions,)
    overall_auc = roc_auc_score( y, oof_predictions,)
    overall_score = ( 0.60 * overall_logloss + 0.40 * (1 - overall_auc))
    training_time = time.time() - start_time

    # Fold Score DataFrame
    fold_scores = pd.DataFrame(
        fold_scores
    )

    # Summary
    print()
    print("=" * 70)
    print("Overall Results")
    print("=" * 70)
    print(f"LogLoss           : {overall_logloss:.6f}")
    print(f"ROC-AUC           : {overall_auc:.6f}")
    print(f"Competition Score : {overall_score:.6f}")
    print(f"Training Time     : {training_time:.2f} sec")
    print("=" * 70)

    # Aggregate Feature Importance
    if len(feature_importance) > 0:
        feature_importance = (
            pd.concat(feature_importance, ignore_index=True)
            .groupby("Feature", as_index=False)["Importance"]
            .mean()
            .sort_values(
                "Importance",
                ascending=False,
            )
        )

        # feature_importance.to_csv(output_dir / f"{model_name}_importance_{timestamp}.csv", index=False, )
        feature_importance.to_csv(
            importance_dir /
            f"{model_name}_importance_{timestamp}.csv",
            index=False,
        )
    else:
        feature_importance = None

    # Save Fold Metrics
    fold_scores.to_csv(
        metrics_dir /
        f"{model_name}_cv_scores_{timestamp}.csv",
        index=False,
    )
    # fold_scores.to_csv(
    #     output_dir
    #     /
    #     f"{model_name}_cv_scores_{timestamp}.csv",
    #     index=False,
    # )

    # Save OOF Predictions
    oof_df = pd.DataFrame({
        "Prediction": oof_predictions
    })

    oof_df.to_csv(
        oof_dir /
        f"{model_name}_{timestamp}.csv",
        index=False,
    )
    # oof_df.to_csv(
    #     output_dir
    #     /
    #     f"{model_name}_oof_{timestamp}.csv",
    #     index=False,
    # )

    # Save Experiment Summary
    summary = pd.DataFrame({
        "Model":[model_name],
        "LogLoss":[overall_logloss],
        "ROC_AUC":[overall_auc],
        "CompetitionScore":[overall_score],
        "TrainingTime":[training_time],
        "Timestamp":[timestamp]
    })

    summary.to_csv(
        metrics_dir /
        f"{model_name}_summary_{timestamp}.csv",
        index=False,
    )
    # summary.to_csv(
    #     output_dir
    #     /
    #     f"{model_name}_summary_{timestamp}.csv",
    #     index=False,
    # )

    # Return
    result = EvaluationResult(
        model_name=model_name,
        fold_scores=fold_scores,
        oof_predictions=oof_predictions,
        trained_models=trained_models,
        feature_importance=feature_importance,
        overall_logloss=overall_logloss,
        overall_auc=overall_auc,
        overall_score=overall_score,
        training_time=training_time,
        timestamp=timestamp,
    )
    
    return result

    
    # return EvaluationResult(
    #     model_name=model_name,
    #     fold_scores=fold_scores,
    #     oof_predictions=oof_predictions,
    #     trained_models=trained_models,
    #     feature_importance=feature_importance,
    #     overall_logloss=overall_logloss,
    #     overall_auc=overall_auc,
    #     overall_score=overall_score,
    #     training_time=training_time,
    #     timestamp=timestamp,
    # )

# This allows the pipeline to work even when we later add another estimator that doesn't implement cloning correctly
def _clone_estimator(model):
    """
    Robust estimator cloning.
    Uses sklearn.clone() whenever possible.
    Falls back to deepcopy() if cloning fails.
    """

    try:
        return clone(model)
    except Exception:
        return deepcopy(model)
        