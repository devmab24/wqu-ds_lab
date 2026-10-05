"""
ensemble.py

Reusable ensemble module.
Supports:
    - Weighted Average
    - Geometric Mean
    - Median Ensemble
    - Rank Average
    - Grid Search
    - Optuna Weight Optimization
    - Correlation Analysis
    - Submission Generation
"""

from __future__ import annotations
import optuna

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import rankdata

from sklearn.metrics import ( log_loss, roc_auc_score)



class EnsembleModel:
    """
    Generic model ensemble.
    Supports any number of models.

    Example:
    -------
    ensemble.add_model(
        "LightGBM",
        prediction,
    )

    ensemble.add_model(
        "XGBoost",
        prediction,
    )

    ensemble.weighted_average(...)
    """

    # Constructor
    def __init__(self):
        self.models = {}
        # self.weights = {} #Never used
        self.best_weights = None
        self.best_score = np.inf
        self.best_strategy = None
        self.history = []
        self.last_evaluation = None
        self._strategies = {
            "average": self.average,
            "weighted": self.weighted_average,
            "geometric": self.geometric_mean,
            "median": self.median,
            "rank": self.rank_average,
            "harmonic": self.harmonic_mean,
            "maximum": self.maximum,
            "minimum": self.minimum,
        }

    #Representation
    def __repr__(self):
        return ( f"EnsembleModel("f"n_models={len(self.models)})")

    # Validation Helpers
    def _check_models(self):
        if len(self.models) == 0:
            raise ValueError("No models have been added.")

    # Prediction Length
    def _validate_lengths(self):
        lengths = [ len(v) for v in self.models.values() ]
        
        if len(set(lengths)) != 1:
            raise ValueError(
                "Prediction lengths do not match."
            )

    # Weight Validation
    def _validate_weights(self, weights):
        """
        Validate and normalize ensemble weights.
    
        Supports:
            - list
            - tuple
            - numpy array
            - dict {model_name: weight}
        """
    
        # Dictionary -> ordered array
        if isinstance(weights, dict):
    
            missing = set(self.models.keys()) - set(weights.keys())
            extra = set(weights.keys()) - set(self.models.keys())
    
            if missing:
                raise ValueError(
                    f"Missing weights for models: {sorted(missing)}"
                )
    
            if extra:
                raise ValueError(
                    f"Unknown models in weights: {sorted(extra)}"
                )
    
            weights = [
                weights[name]
                for name in self.models.keys()
            ]
    
        weights = np.asarray(weights, dtype=float)
    
        if len(weights) != len(self.models):
            raise ValueError(
                "Number of weights must equal number of models."
            )
    
        if np.any(weights < 0):
            raise ValueError(
                "Weights must be non-negative."
            )
    
        if np.isclose(weights.sum(), 0):
            raise ValueError(
                "Weights sum to zero."
            )
    
        # Normalize
        weights /= weights.sum()
    
        return weights
    # def _validate_weights(self,weights,):
    #     if len(weights) != len(self.models):
    #         raise ValueError("Number of weights " "must equal number of models.")

    #     weights = np.asarray(weights)
    #     if np.any(weights < 0):
    #         raise ValueError( "Weights must be non-negative." )

    #     if weights.sum() == 0:
    #         raise ValueError( "Weights sum to zero.")

    #     return weights / weights.sum()

    # Add Model
    def add_model(self,name,prediction,):
        """
        Add prediction.
        Parameters
        ----------

        name
            Model name

        prediction
            Probability prediction
        """

        prediction = np.asarray( prediction )
        self.models[name] = prediction
        self._validate_lengths()

    # Remove Model
    def remove_model(self,name,):
        if name not in self.models:
            raise ValueError(
                f"{name} "
                "not found."
            )

        del self.models[name]

    # List Models
    def list_models(self):
        return list(self.models.keys())

    # Number of Models
    @property
    def n_models(self):
        return len(self.models)

    # Clear
    def clear(self):
        self.models = {}
        self.best_weights = None
        self.best_score = np.inf
        self.best_strategy = None

    #Helper
    def _load_prediction_file(
        self,
        model_name,
        file_path,
    ):
        """
        Load and validate a prediction file.
        """
    
        df = pd.read_csv(file_path)
    
        if "Prediction" not in df.columns:
            raise ValueError(
                f"{model_name}: Missing 'Prediction' column."
            )
    
        prediction = df["Prediction"].to_numpy()
    
        if np.isnan(prediction).any():
            raise ValueError(
                f"{model_name}: Prediction file contains NaN values."
            )
    
        if np.isinf(prediction).any():
            raise ValueError(
                f"{model_name}: Prediction file contains infinite values."
            )
    
        if ((prediction < 0) | (prediction > 1)).any():
            raise ValueError(
                f"{model_name}: Predictions must be between 0 and 1."
            )
    
        return prediction
        
    # Load Predictions
    def load_predictions(self, prediction_files):
        """
        Load multiple prediction files.
        """
        self.clear()
    
        for model_name, file_path in prediction_files.items():
            file_path = Path(file_path)
    
            if not file_path.exists():
                raise FileNotFoundError(file_path)
    
            prediction = self._load_prediction_file(
                model_name,
                file_path,
            )
    
            self.add_model(
                model_name,
                prediction,
            )

    # Summary
    def summary(self):
        print()
        print("="*60)
        print("ENSEMBLE SUMMARY")
        print("="*60)
        print()
        print(f"Models : {self.n_models}")
        print()
        for model in self.list_models():
            print(f" • {model}")
        
        print()
        if self.best_strategy is not None:
            print(
                f"Best Strategy : "
                f"{self.best_strategy}"
            )

        if self.best_score != np.inf:
            print(
                f"Best Score : "
                f"{self.best_score:.6f}"
            )

        if self.best_weights is not None:
            print()
            print("Weights")
            print("-"*20)
            for k,v in self.best_weights.items():
                print(f"{k:<20}{v:.4f}")

        print()
        print("="*60)
        

    # Prediction Matrix Helper
    def _prediction_matrix(self):
        """
        Returns
        -------
        np.ndarray

        Shape:
            (n_samples, n_models)
        """

        self._check_models()
        self._validate_lengths()
        return np.column_stack(
            list(self.models.values())
        )

    # Weighted Average
    def weighted_average(self, weights):
        """
        Weighted average ensemble.
        """
    
        weights = self._validate_weights(weights)
    
        X = self._prediction_matrix()
    
        return np.average(
            X,
            axis=1,
            weights=weights,
        )
    # def weighted_average(self, weights):
    #     """
    #     Weighted average ensemble.

    #     Parameters
    #     ----------
    #     weights : list-like
    #     Returns
    #     -------
    #     ndarray
    #     """
    #     if isinstance(weights, dict):
    #         weights = [
    #             weights[name]
    #             for name in self.models.keys()
    #         ]
    
    #     weights = self._validate_weights(weights)
    #     X = self._prediction_matrix()
    
    #     return np.average(X,axis=1,weights=weights,)
        
    # def weighted_average(self,weights,):
    #     """
    #     Weighted average ensemble.

    #     Parameters
    #     ----------
    #     weights : list-like
    #     Returns
    #     -------
    #     ndarray
    #     """

    #     weights = self._validate_weights(weights)
    #     X = self._prediction_matrix()
    #     prediction = X @ weights
    #     return np.clip(
    #         prediction,
    #         1e-15,
    #         1 - 1e-15,
    #     )

    # Simple Average
    def average(self):
        """
        Equal-weight average.
        """

        X = self._prediction_matrix()
        prediction = X.mean(axis=1)
        return np.clip(
            prediction,
            1e-15,
            1 - 1e-15,
        )

    # Geometric Mean
    def geometric_mean(self):
        """
        Geometric Mean Ensemble.
        """

        X = self._prediction_matrix()
        X = np.clip(
            X,
            1e-15,
            1 - 1e-15,
        )

        prediction = np.exp(
            np.mean(
                np.log(X),
                axis=1,
            )
        )

        return np.clip(
            prediction,
            1e-15,
            1 - 1e-15,
        )

    # Median Ensemble
    def median(self):
        """
        Median Ensemble.
        """

        X = self._prediction_matrix()
        prediction = np.median(
            X,
            axis=1,
        )

        return np.clip(
            prediction,
            1e-15,
            1 - 1e-15,
        )

    # Maximum Ensemble
    def maximum(self):
        """
        Maximum prediction ensemble.
        """

        X = self._prediction_matrix()
        prediction = X.max(axis=1)
        return np.clip(
            prediction,
            1e-15,
            1 - 1e-15,
        )

    # Minimum Ensemble
    def minimum(self):
        """
        Minimum prediction ensemble.
        """

        X = self._prediction_matrix()
        prediction = X.min(axis=1)

        return np.clip(
            prediction,
            1e-15,
            1 - 1e-15,
        )

    # Rank Average
    def rank_average(self):
        """
        Rank Average Ensemble.

        Each model is converted into ranks
        before averaging.
        """

        X = self._prediction_matrix()
        ranked = np.column_stack(
            [
                rankdata(col)
                for col in X.T
            ]
        )

        prediction = ranked.mean(axis=1)
        prediction = (
            prediction
            - prediction.min()
        ) / (
            prediction.max()
            - prediction.min()
        )

        return np.clip(
            prediction,
            1e-15,
            1 - 1e-15,
        )

    # Harmonic Mean
    def harmonic_mean(self):
        """
        Harmonic Mean Ensemble.
        """

        X = self._prediction_matrix()
        X = np.clip(
            X,
            1e-15,
            1,
        )
        prediction = X.shape[1] / np.sum(
            1 / X,
            axis=1,
        )

        return np.clip(
            prediction,
            1e-15,
            1 - 1e-15,
        )

    # Strategy Dispatcher
    def predict( self, strategy="average", weights=None,):
        """
        Generate ensemble prediction.
        """
    
        strategy = strategy.lower()
    
        if strategy not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
    
        if strategy == "weighted":
            if weights is None:
                raise ValueError("Weights are required for weighted strategy." )
    
            return self.weighted_average(weights)
    
        return self._strategies[strategy]()
        

    # Evaluation & Model Comparison
    
    # Competition Metric
    def competition_score(
        self,
        y_true,
        prediction,
    ):
        """
        Calculate competition metric.
        """

        prediction = np.clip(
            prediction,
            1e-15,
            1 - 1e-15,
        )

        ll = log_loss(
            y_true,
            prediction,
        )

        auc = roc_auc_score(
            y_true,
            prediction,
        )

        score = (

            0.60 * ll

            +

            0.40 * (1 - auc)

        )

        return {
            "LogLoss": ll,
            "ROC_AUC": auc,
            "CompetitionScore": score,
        }

    # Evaluate One Strategy
    def evaluate(
        self,
        y_true,
        strategy="average",
        weights=None,
    ):
        """
        Evaluate a single ensemble strategy.
        """
    
        prediction = self.predict(
            strategy=strategy,
            weights=weights,
        )
    
        metrics = self.competition_score(
            y_true,
            prediction,
        )
    
        metrics["Strategy"] = strategy
    
        # Store normalized weights for weighted ensembles
        if strategy.lower() == "weighted":
            normalized = self._validate_weights(weights)
            metrics["Weights"] = {
                model: float(weight)
                for model, weight in zip(
                    self.models.keys(),
                    normalized,
                )
            }
        else:
            metrics["Weights"] = None
        
        self.last_evaluation = metrics
        self.history.append(metrics.copy())
        self.last_evaluation = metrics
                
        return metrics

    def history_dataframe(self):
        """
        Return all ensemble evaluations.
        """
    
        if len(self.history) == 0:
            return pd.DataFrame()
    
        return pd.DataFrame(self.history)

    #So that we can then do `ensemble.history_dataframe()` to see history

    def best_result(self):
        """
        Return the best ensemble evaluated so far.
        """
    
        history = self.history_dataframe()
        if history.empty:
            return None
    
        return history.sort_values(
            "CompetitionScore"
        ).iloc[0]
        
    # def evaluate(
    #     self,
    #     y_true,
    #     strategy="average",
    #     weights=None,
    # ):
    #     """
    #     Evaluate a single ensemble strategy.
    #     """

    #     prediction = self.predict(

    #         strategy=strategy,

    #         weights=weights,

    #     )

    #     metrics = self.competition_score(

    #         y_true,

    #         prediction,

    #     )

    #     metrics["Strategy"] = strategy

    #     return metrics

    # Evaluate All Strategies
    def evaluate_all(
        self,
        y_true,
    ):
        """
        Evaluate every supported strategy.
        """

        strategies = [
            "average",
            "geometric",
            "median",
            "rank",
            "harmonic",
            "maximum",
            "minimum",

        ]

        results = []

        for strategy in strategies:

            metrics = self.evaluate(

                y_true,

                strategy=strategy,

            )

            results.append(metrics)

        df = pd.DataFrame(results)

        df = df.sort_values(

            "CompetitionScore"

        ).reset_index(drop=True)

        return df

    # Evaluate Weighted Average
    def evaluate_weighted(
        self,
        y_true,
        weights,
    ):
        """
        Evaluate weighted average.
        """

        metrics = self.evaluate(

            y_true,

            strategy="weighted",

            weights=weights,

        )

        return metrics

    # Compare Individual Models
    def compare_models(
        self,
        y_true,
    ):
        """
        Compare every individual model.
        """

        rows = []

        for name, prediction in self.models.items():

            metrics = self.competition_score(

                y_true,

                prediction,

            )

            metrics["Model"] = name

            rows.append(metrics)

        df = pd.DataFrame(rows)

        df = df.sort_values(

            "CompetitionScore"

        ).reset_index(drop=True)

        return df

    # Compare Models + Ensembles
    def leaderboard(
        self,
        y_true,
    ):
        """
        Compare models and ensemble methods.
        """

        model_scores = self.compare_models(

            y_true

        )

        model_scores = model_scores.rename(
            columns={
                "Model": "Name"
            }

        )

        model_scores["Type"] = "Model"

        ensemble_scores = self.evaluate_all(
            y_true
        )

        ensemble_scores = ensemble_scores.rename(
            columns={
                "Strategy": "Name"
            }

        )

        ensemble_scores["Type"] = "Ensemble"

        leaderboard = pd.concat(

            [
                model_scores,
                ensemble_scores,
            ],

            ignore_index=True,
        )

        leaderboard = leaderboard.sort_values(
            "CompetitionScore"
        ).reset_index(drop=True)

        return leaderboard

    # Correlation Matrix
    def correlation_matrix(
        self,
    ):
        """
        Correlation among model predictions.
        """

        X = self._prediction_matrix()

        df = pd.DataFrame(

            X,

            columns=self.list_models(),

        )

        return df.corr()

    # Agreement Matrix
    def agreement_matrix(
        self,
        threshold=0.5,
    ):
        """
        Percentage agreement
        between models.
        """

        binary = {}

        for name, pred in self.models.items():
            binary[name] = (
                pred >= threshold
            ).astype(int)

        binary = pd.DataFrame(binary)
        agreement = pd.DataFrame(
            index=binary.columns,
            columns=binary.columns,
            dtype=float,

        )

        for c1 in binary.columns:
            for c2 in binary.columns:
                agreement.loc[c1, c2] = (
                    binary[c1]
                    ==
                    binary[c2]
                ).mean()

        return agreement

    # Prediction Summary
    def prediction_summary(
        self,
    ):
        """
        Summary statistics for each model.
        """

        rows = []

        for name, pred in self.models.items():

            rows.append({
                "Model": name,
                "Mean": pred.mean(),
                "Std": pred.std(),
                "Min": pred.min(),
                "25%": np.percentile(pred,25),
                "Median": np.median(pred),
                "75%": np.percentile(pred,75),
                "Max": pred.max(),
            })

        return pd.DataFrame(rows)

    # Model Diversity
    def diversity(
        self,
    ):
        """
        Average pairwise correlation.
        """

        corr = self.correlation_matrix()
        mask = np.triu(
            np.ones(corr.shape),
            k=1,
        ).astype(bool)

        values = corr.where(mask).stack()

        return {
            "AverageCorrelation": values.mean(),
            "MinimumCorrelation": values.min(),
            "MaximumCorrelation": values.max(),
        }


    # Generic Weight Optimization

    # Generate Weight Combinations (2 Models)
    def _generate_weight_grid(
        self,
        step=0.05,
    ):
        """
        Generate weight combinations for two models.

        Returns
        -------
        list[list]
        """

        if self.n_models != 2:

            raise NotImplementedError(
                "Grid search currently supports exactly 2 models."
            )

        weights = []

        for w in np.arange(0, 1 + step, step):

            weights.append(
                [w, 1 - w]
            )

        return weights

    # Grid Search
    def search_weights(
        self,
        y_true,
        step=0.01,
    ):
        """
        Grid search for the best ensemble weights.
        """

        results = []

        model_names = self.list_models()

        for weights in self._generate_weight_grid(step):

            metrics = self.evaluate(
                y_true,
                strategy="weighted",
                weights=weights,
            )

            #tO make it work for multiple models
            row = {
                **dict(zip(model_names, weights)),
                **metrics,
            }
            # row = {
            #     model_names[0]: weights[0],
            #     model_names[1]: weights[1],
            #     **metrics,
            # }

            results.append(row)

        results = pd.DataFrame(results)
        results = results.sort_values(
            "CompetitionScore"
        ).reset_index(drop=True)

        best = results.iloc[0]
        self.best_score = best["CompetitionScore"]
        self.best_strategy = "weighted"
        self.best_weights = {
            model_names[0]: best[model_names[0]],
            model_names[1]: best[model_names[1]],
        }

        self.best_result = best.to_dict()
        
        return results

    # Best Prediction
    def best_prediction(self):
        """
        Predict using the best discovered weights.
        """

        if self.best_weights is None:
            raise ValueError(
                "Run search_weights() or optimize_weights() first."
            )

        weights = list(
            self.best_weights.values()
        )

        return self.predict(
            strategy="weighted",
            weights=weights,
        )

    # Best Metrics
    def best_metrics(
        self,
        y_true,
    ):
        """
        Evaluate the best ensemble.
        """

        prediction = self.best_prediction()
        return self.competition_score(
            y_true,
            prediction,
        )

    # Top Results
    def top_results(
        self,
        results,
        n=10,
    ):
        """
        Return top weight combinations.
        """

        return results.head(n)

    # Best Weights DataFrame
    def best_weights_df(self):
        if self.best_weights is None:
            raise ValueError(
                "No optimized weights."
            )

        return pd.DataFrame({
            "Model": self.best_weights.keys(),
            "Weight": self.best_weights.values(),
        })

    # Optuna Optimization

    # Objective Function
    def _objective(
        self,
        trial,
        y_true,
    ):
        """
        Optuna objective.
        """

        weights = []

        for i in range(self.n_models):
            weights.append(
                trial.suggest_float(
                    f"w{i}",
                    0,
                    1,
                )
            )

        prediction = self.weighted_average(
            weights
        )

        metrics = self.competition_score(
            y_true,
            prediction,
        )

        return metrics["CompetitionScore"]

    # Optimize Weights
    def optimize_weights(
        self,
        y_true,
        n_trials=100,
        study_name=None,
    ):
        """
        Optimize ensemble weights using Optuna.
        """

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
        )

        study.optimize(
            lambda trial:
            self._objective(
                trial,
                y_true,
            ),

            n_trials=n_trials,
        )

        best = study.best_params

        weights = []

        for i in range(self.n_models):
            weights.append(
                best[f"w{i}"]
            )

        weights = self._validate_weights(
            weights
        )

        self.best_weights = {
            model: weight
            for model, weight in zip(
                self.list_models(),
                weights,
            )
        }

        self.best_score = study.best_value
        self.best_strategy = "weighted"

        return study

    # Optimization Summary
    def optimization_summary(self):
        print()
        print("="*60)
        print("BEST ENSEMBLE")
        print("="*60)
        print()
        print(
            f"Strategy : "
            f"{self.best_strategy}"
        )

        print()
        print(
            f"Score : "
            f"{self.best_score:.6f}"
        )

        print()
        for model, weight in self.best_weights.items():
            print(
                f"{model:<20}"
                f"{weight:.4f}"
            )

        print()
        print("="*60)

    # Save Best Weights
    def save_best_weights(
        self,
        path,
    ):
        """
        Save optimized weights.
        """

        if self.best_weights is None:
            raise ValueError(
                "No optimized weights."
            )

        with open(path, "w") as f:
            json.dump(
                self.best_weights,
                f,
                indent=4,
            )

    # Load Best Weights
    def load_best_weights(
        self,
        path,
    ):

        with open(path) as f:
            self.best_weights = json.load(f)

        return self.best_weights


    # 1. Automatic Submission Generation
    def create_submission(
        self,
        sample_submission,
        prediction,
        target="Target",
    ):
        """
        Create a competition submission DataFrame.
        """
    
        submission = sample_submission.copy()
        submission[target] = prediction
    
        return submission

    # Save Submission
    def save_submission(
        self,
        submission,
        path,
    ):
        """
        Save submission CSV.
        """
    
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    
        submission.to_csv(
            path,
            index=False,
        )
    
        print(f"Saved to {path}")

    # Export Leaderboard
    def save_leaderboard(
        self,
        leaderboard,
        path,
    ):
        leaderboard.to_csv(
            path,
            index=False,
        )

    # Save Search Results
    def save_search_results(
        self,
        results,
        path,
    ):
        results.to_csv(
            path,
            index=False,
        )

    # Save Correlation Matrix
    def save_correlation(
        self,
        path,
    ):
        self.correlation_matrix().to_csv(path)