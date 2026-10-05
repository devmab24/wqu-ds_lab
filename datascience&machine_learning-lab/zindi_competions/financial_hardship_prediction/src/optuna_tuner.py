import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.metrics import (log_loss,roc_auc_score,)

from experiment import OptunaResult

from config import LIGHTGBM_SEARCH_SPACE_V1, LIGHTGBM_SEARCH_SPACE_V2,  XGBOOST_SEARCH_SPACE_V1, XGBOOST_SEARCH_SPACE_V2
from model_configs import (
    LIGHTGBM_DEFAULTS,
    XGBOOST_DEFAULTS,
    CATBOOST_DEFAULTS,
)


"""
optuna_tuner.py

Goal: Search for the best hyperparameters.

Hyperparameter optimization utilities using Optuna.
(Currently supports LightGBM.)
"""

DEFAULT_HISTGB_SEARCH_SPACE = {}

# Model Search Space Registry
MODEL_SEARCH_SPACES = {
    "lightgbm": LIGHTGBM_SEARCH_SPACE_V2,
    "xgboost": XGBOOST_SEARCH_SPACE_V2,
    "histgb": DEFAULT_HISTGB_SEARCH_SPACE,
}

class OptunaTuner:
    """
    Generic Optuna hyperparameter tuner.
    Currently supports:
        - LightGBM
    Future:
        - XGBoost
        - HistGradientBoosting
    """

    def __init__(
        self,
        cv,
        n_trials=100,
        random_state=42,
        direction="minimize",
        metric="competition",
        experiment_tracker=None,
        study_name="lightgbm",
        model_name="LightGBM",
        model_type="lightgbm",
        verbose=True,
        pruner="median",
        n_startup_trials=10,
        n_warmup_steps=0,
        search_space=None,
    ):
        """
            Parameters
            ----------
            cv : sklearn Cross Validation object
    
            n_trials : int
                Number of Optuna trials.
            random_state : int
            study_name : str
            direction : str
                "minimize" or "maximize"
            metric : str
                Currently only supports:
                    "competition"
            verbose : bool
                Print Optuna progress.
        """

        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.study_name = study_name
        self.direction = direction
        self.metric = metric
        self.experiment_tracker = experiment_tracker
        self.model_name = model_name
        self.verbose = verbose
        
        # Output directories
        self.output_dir = ( Path("../outputs") / "optuna" / self.study_name.lower())
        self.output_dir.mkdir( parents=True, exist_ok=True,)
        
        # Files        
        self.study_path = ( self.output_dir / f"{self.study_name.lower()}.db" )
        self.best_params_path = ( self.output_dir / "best_params.json")
        self.trials_path = (self.output_dir / "trials.csv" )

        # Filled after optimize()
        self.study = None
        self.best_model_ = None
        self.best_params_ = None
        self.best_score_ = None

        #Prunners
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        
        # Model
        self.model_type = model_type
        
        # Default Search Space
        self.search_space = MODEL_SEARCH_SPACES[self.model_type].copy()        
        # User Overrides
        if search_space is not None:
            self.search_space.update(search_space)
             

    def optimize(self,X,y, ):
        """
        Run Optuna hyperparameter optimization.
        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Target values.
    
        Returns
        -------
        OptunaResult
            Summary of the optimization study.
        """
    
        # Start timer    
        self.optimization_start = time.time()
    
        # Store data for objective()    
        self.X = X
        self.y = y

        # Create study 
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=f"sqlite:///{self.study_path}",
            load_if_exists=True,
            pruner=self._build_pruner(),
        )
    
        # Display summary    
        if self.verbose:
            print("=" * 60)
            print("Starting Optuna Optimization")
            print("=" * 60)
            print(f"Study Name : {self.study_name}")
            print(f"Trials     : {self.n_trials}")
            print()
    
        # Run optimization    
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            callbacks=[self._progress_callback],
            show_progress_bar=False,
        )
    
        # Compute optimization time    
        self.optimization_time = (
            time.time() - self.optimization_start
        )
    
        # Store best results    
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value
    
        # Save best parameters    
        with open( self.best_params_path, "w", ) as f:
            json.dump(
                self.best_params_,
                f,
                indent=4,
            )
    
        # Save all trials    
        trials = self.study.trials_dataframe()
        trials.to_csv(
            self.trials_path,
            index=False,
        )
    
        # Train best model on full dataset    
        self.best_model_ = self._build_model(
            self.best_params_,
        )
    
        self.best_model_.fit( X, y,)
    
        # Store feature names    
        self.feature_names_ = list(X.columns)
    
        # Automatic experiment logging(Enable after ExperimentTracker.log_optuna() is complete)
        if self.experiment_tracker is not None:
            self.experiment_tracker.log_optuna(
                results=OptunaResult(
                    study_name=self.study_name,
                    model_name=self.model_name,
                    best_score=self.best_score_,
                    best_params=self.best_params_,
                    n_trials=len(self.study.trials),
                    optimization_time=self.optimization_time,
                )
            )
    
        # Summary    
        if self.verbose:
            print()
            print("=" * 60)
            print("Optimization Finished")
            print("=" * 60)
            print(f"Best Score : {self.best_score_:.6f}")
            print()
            print("Best Parameters")
            
            for key, value in self.best_params_.items():
                print(f"{key:<30}: {value}")

            print()
            print(f"Optimization Time : {self.optimization_time:.1f} sec")
            print("=" * 60)
    
        # Return OptunaResult    
        return OptunaResult(
            study_name=self.study_name,
            model_name=self.model_name,
            best_score=self.best_score_,
            best_params=self.best_params_,
            n_trials=len(self.study.trials),
            optimization_time=self.optimization_time,
        )
     
    def _objective(self, trial):
        """
        Objective function optimized by Optuna.
    
        Parameters
        ----------
        trial : optuna.Trial
        Returns
        -------
        float
            Mean competition score across all CV folds.
        """
    
        # Suggest hyperparameters
        params = self._suggest_params(trial)
        # params = self._suggest_lightgbm_params(trial)

        # Store CV scores
        fold_scores = []
        # Cross Validation
        for fold, (train_idx, valid_idx) in enumerate(
            self.cv.split(self.X, self.y),
            start=1,
        ):
            # Split data
            X_train = self.X.iloc[train_idx]
            X_valid = self.X.iloc[valid_idx]
    
            y_train = self.y.iloc[train_idx]
            y_valid = self.y.iloc[valid_idx]
            # Build model
            model = self._build_model(params)
    
            # Evaluate fold
            score = self._evaluate_fold(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )
            # Store fold score
            fold_scores.append(score)
            # Running CV score
            current_score = np.mean(fold_scores)
            # Report intermediate score to Optuna
            trial.report(
                current_score,
                step=fold,
            )
    
            # Prune unpromising trials
            if trial.should_prune():
                if self.verbose:
                    print(
                        f"Trial {trial.number} "
                        f"pruned after fold {fold} "
                        f"(score={current_score:.6f})"
                    )
    
                raise optuna.TrialPruned()
    
        # Return mean CV score
        return np.mean(fold_scores)

    def _suggest_params(self, trial):
        """
        Suggest hyperparameters for the selected model.
        """
    
        if self.model_type == "lightgbm":
            return self._suggest_lightgbm_params(trial)
    
        if self.model_type == "xgboost":
            return self._suggest_xgboost_params(trial)
    
        if self.model_type == "histgb":
            return self._suggest_histgb_params(trial)
    
        raise ValueError(
            f"Unsupported model_type: {self.model_type}"
        )

    def _suggest_lightgbm_params(self, trial):
        """
        Suggest a LightGBM hyperparameter configuration.
    
        Parameters
        ----------
        trial : optuna.Trial
    
        Returns
        -------
        dict
        """
    
        ####################################################
        # Helper
        def suggest(name, log=False, integer=False):
    
            value = self.search_space[name]
    
            # Fixed parameter
            if not isinstance(value, tuple):
                return value
    
            low, high = value
    
            if integer:
                return trial.suggest_int(
                    name,
                    low,
                    high,
                )
    
            return trial.suggest_float(
                name,
                low,
                high,
                log=log,
            )
    
        ####################################################
        # Build parameter dictionary
    
        params = {
    
            # Fixed
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": self.random_state,
            "verbosity": -1,
    
            # Learning
            "learning_rate": suggest(
                "learning_rate",
                log=True,
            ),
    
            "n_estimators": suggest(
                "n_estimators",
                integer=True,
            ),
    
            # Tree
            "num_leaves": suggest(
                "num_leaves",
                integer=True,
            ),
    
            "max_depth": suggest(
                "max_depth",
                integer=True,
            ),
    
            "min_child_samples": suggest(
                "min_child_samples",
                integer=True,
            ),
    
            "min_child_weight": suggest(
                "min_child_weight",
                log=True,
            ),
    
            # Sampling
            "subsample": suggest(
                "subsample",
            ),
    
            "subsample_freq": suggest(
                "subsample_freq",
                integer=True,
            ),
    
            "colsample_bytree": suggest(
                "colsample_bytree",
            ),
    
            # Histogram
            "max_bin": suggest(
                "max_bin",
                integer=True,
            ),
    
            # Regularization
            "reg_alpha": suggest(
                "reg_alpha",
            ),
    
            "reg_lambda": suggest(
                "reg_lambda",
            ),
    
            "min_split_gain": suggest(
                "min_split_gain",
            ),
    
            "min_sum_hessian_in_leaf": suggest(
                "min_sum_hessian_in_leaf",
                log=True,
            ),
    
            # Class imbalance
            "scale_pos_weight": suggest(
                "scale_pos_weight",
            ),
        }
    
        return params
        
    def _suggest_xgboost_params(self, trial):
        """
        Suggest an XGBoost hyperparameter configuration.
    
        Parameters
        ----------
        trial : optuna.Trial
    
        Returns
        -------
        dict
        """
    
        # Helper
        def suggest(name, log=False, integer=False):
            value = self.search_space[name]
            # Fixed parameter
            if not isinstance(value, tuple):
                return value
    
            low, high = value
    
            if integer:
                return trial.suggest_int(
                    name,
                    low,
                    high,
                )
    
            return trial.suggest_float(
                name,
                low,
                high,
                log=log,
            )
    
        # Build parameter dictionary
        params = {
            # Fixed parameters
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": self.random_state,
    
            # Faster training
            "tree_method": "hist",
            "verbosity": 0,
            "n_jobs": -1,
    
            # Learning
            "learning_rate": suggest( "learning_rate", log=True,),
            "n_estimators": suggest( "n_estimators", integer=True,),
    
            # Tree    
            "max_depth": suggest( "max_depth", integer=True, ),
            "min_child_weight": suggest( "min_child_weight", log=True,),
    
            "gamma": suggest(
                "gamma",
            ),
    
            # Sampling
            "subsample": suggest(
                "subsample",
            ),
    
            "colsample_bytree": suggest(
                "colsample_bytree",
            ),
    
            # Regularization
            "reg_alpha": suggest(
                "reg_alpha",
            ),
    
            "reg_lambda": suggest(
                "reg_lambda",
            ),
    
            # Imbalance
            "scale_pos_weight": suggest(
                "scale_pos_weight",
            ),
        }
    
        return params
        
    #Not yet implemented
    def _suggest_histgb_params(self, trial):
        raise NotImplementedError

    def _build_model(self, params):
        """
        Build a model using the supplied hyperparameters.
        Parameters
        ----------
        params : dict
            Hyperparameters suggested by Optuna.
        Returns
        -------
        sklearn estimator
        """
        # LightGBM    
        if self.model_type.lower() == "lightgbm":
            return LGBMClassifier(
                **LIGHTGBM_DEFAULTS,
                **params,
            )

        # XGBoost
        elif self.model_type.lower() == "xgboost":
            return XGBClassifier(
                **XGBOOST_DEFAULTS,
                **params,
            )

        # CatBoost
        elif self.model_type.lower() == "catboost":
            return CatBoostClassifier(
                **CATBOOST_DEFAULTS,
                **params,
            )

        # HistGradientBoosting    
        elif self.model_type.lower() == "histgradientboosting":
            # later
            pass

        # Unknown model   
        raise ValueError(
            f"Unsupported study_name: {self.model_type}"
        )

    def _evaluate_fold(
        self,
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
    ):
        """
        Train one fold and return the competition score.
    
        Parameters
        ----------
        model : sklearn estimator
        X_train : pd.DataFrame
        y_train : pd.Series
        X_valid : pd.DataFrame
        y_valid : pd.Series
    
        Returns
        -------
        float
            Competition score for one fold.
        """
    
        # Train
        model.fit(
            X_train,
            y_train,
        )
    
        # Predict probabilities    
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_valid)[:, 1]
    
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_valid)
            scores = np.clip(
                scores,
                -500,
                500,
            )
            preds = 1 / (1 + np.exp(-scores))
        else:
            raise ValueError(
                "Estimator must support predict_proba() "
                "or decision_function()."
            )
    
        # Metrics    
        logloss = log_loss(
            y_valid,
            preds,
        )
    
        auc = roc_auc_score(
            y_valid,
            preds,
        )
    
        # Competition score    
        return self._compute_score(
            logloss,
            auc,
        )

 
    def _build_pruner(self):
        """
        Build the Optuna pruner.
    
        Returns
        -------
        optuna.pruners.BasePruner
        """
        # No pruning
        if self.pruner is None:
            return optuna.pruners.NopPruner()
    
        # Median Pruner
        if self.pruner == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=1,
            )
    
        # Successive Halving
        if self.pruner == "successive_halving":
            return optuna.pruners.SuccessiveHalvingPruner()
    
        # Hyperband
        if self.pruner == "hyperband":
            return optuna.pruners.HyperbandPruner()
    
        # Unknown pruner
        raise ValueError(
            f"Unknown pruner: {self.pruner}"
        )
        
    #Tracking progress for each trial
    def _progress_callback(self, study, trial,):
        """
        Callback executed after every completed Optuna trial.
        """
        if not self.verbose:
            return
    
        elapsed = time.time() - self.optimization_start
    
        print(
            f"[Trial {trial.number + 1:>3}/{self.n_trials}] "
            f"Current={trial.value:.6f} | "
            f"Best={study.best_value:.6f} | "
            f"Elapsed={elapsed:.1f}s"
        )

    def _get_range(self, parameter):
        """
        Return the search range for a parameter.
        Parameters
        ----------
        parameter : str
        Returns
        -------
        tuple
        """
        return self.search_space[parameter]


    def _compute_score(self, logloss,auc,):
        """
        Compute the optimization objective.
    
        Parameters
        ----------
        logloss : float
        auc : float
        Returns
        -------
        float
        """
        # Competition Metric    
        if self.metric == "competition":
            return ( 0.60 * logloss + 0.40 * (1 - auc))
    
        # Pure LogLoss    
        elif self.metric == "logloss":
            return logloss
    
        # Maximize ROC-AUC    
        elif self.metric == "auc":
            #To keep one optimization direction across all metrics, we convert AUC into a minimization objective: Loss = 1 − AUC
            return 1 - auc 
    
        # Unknown metric    
        raise ValueError(
            f"Unknown metric: {self.metric}"
        )
