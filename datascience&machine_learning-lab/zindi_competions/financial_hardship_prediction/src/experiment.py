from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from dataclasses import dataclass


"""
experiment/tracker.py
Experiment tracking utilities.
"""
@dataclass
class OptunaResult:
    """
    Stores the outcome of an Optuna optimization study.
    """
    study_name: str
    best_score: float
    best_params: dict
    n_trials: int
    optimization_time: float
    model_name: str
    
class ExperimentTracker:
    def __init__( self, experiment_file="../outputs/experiments/experiment_logs.csv",):
        self.experiment_file = Path(experiment_file)
        self.experiment_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _append(self, experiment: dict,):
        """
        Append one experiment to the experiment log.
        """
        # Convert to dataframe
        df = pd.DataFrame([experiment])
    
        # Append or create file    
        if self.experiment_file.exists():
            existing = pd.read_csv(
                self.experiment_file,
            )
            df = pd.concat(
                [
                    existing,
                    df,
                ],
                ignore_index=True,
            )
    
        # Save    
        df.to_csv( self.experiment_file, index=False,)
    
        # Display last experiment    
        print()
        print("Experiment saved.")
        print(df.tail(1))
        
    def log_evaluation(self, results, feature_selection=None, notes=None,**kwargs,):
        """
        Log a model evaluation experiment.
        Parameters
        ----------
        results : EvaluationResult
        feature_selection : dict, optional
        notes : str, optional
        """
        # Default feature selection    
        if feature_selection is None:
            feature_selection = {}
    
        # Automatic notes
        if notes is None:
            if feature_selection.get("selection_strategy"):
                notes = (
                    f"{results.model_name} | "
                    f"{feature_selection['selection_strategy']}"
                )
                if feature_selection.get("top_n") is not None:
                    notes += (
                        f" | Top "
                        f"{feature_selection['top_n']} Features"
                    )
            else:
                notes = results.model_name
    
        # Model parameters    
        model_parameters = None
    
        if len(results.trained_models) > 0:
            first_model = results.trained_models[0]
            # Compatibility with both storage styles
            if isinstance(first_model, dict):
                estimator = first_model["model"]
            else:
                estimator = first_model

            model_parameters = json.dumps(
                estimator.get_params(),
                default=str,
            )
    
        # Experiment dictionary    
        experiment = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ExperimentType": "evaluation",
            "Model": results.model_name,
            "SelectionStrategy": feature_selection.get("selection_strategy"),
            "TopN": feature_selection.get("top_n"),
            "Percentile": feature_selection.get("percentile"),
            "UseCVImportance": feature_selection.get("use_cv_importance"),
            "FeaturesBefore": feature_selection.get("features_before"),
            "FeaturesAfter": feature_selection.get("features_after"),
            "LogLoss": results.overall_logloss,
            "ROC_AUC": results.overall_auc,
            "CompetitionScore": results.overall_score,
            "TrainingTime": getattr(results, "training_time", None),
    
            # Optuna placeholders    
            "OptunaTrials": None,
            "OptunaBestValue": None,
            "OptunaStudy": None,
            "OptunaDuration": None,
            "BestParameters": None,
            ####################################################
            "ModelParameters": model_parameters,
            "Notes": notes,
        }
    
        # Extra user-defined metadata
        experiment.update(kwargs)
        # Save    
        self._append(experiment)

    def log_optuna(self, results, feature_selection=None, notes=None, **kwargs,):
        """
        Log an Optuna optimization experiment.
    
        Parameters
        ----------
        results : OptunaResult
        feature_selection : dict
        notes : str
        """
    
        # Feature selection    
        if feature_selection is None:
            feature_selection = {}
    
        # Notes    
        if notes is None:
            notes = (
                f"{results.model_name} "
                f"| Optuna ({results.n_trials} trials)"
            )
    
            if feature_selection.get("top_n") is not None:
                notes += (
                    f" | Top "
                    f"{feature_selection['top_n']} Features"
                )
    
        # Experiment dictionary    
        experiment = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ExperimentType": "optuna",
            "Model": results.model_name,
            # Feature Selection    
            "SelectionStrategy": feature_selection.get("selection_strategy"),
            "TopN": feature_selection.get("top_n"),
            "Percentile": feature_selection.get("percentile"),
            "UseCVImportance": feature_selection.get("use_cv_importance"),
            "FeaturesBefore": feature_selection.get("features_before"),
            "FeaturesAfter": feature_selection.get("features_after"),
            # Evaluation placeholders    
            "LogLoss": None,
            "ROC_AUC": None,
            "CompetitionScore": None,
            "TrainingTime": None,
            # Optuna    
            "OptunaTrials": results.n_trials,
            "OptunaBestValue": results.best_score,
            "OptunaStudy": results.study_name,
            "OptunaDuration": results.optimization_time,
            "BestParameters":
                json.dumps(
                    results.best_params,
                    default=str,
                ),    
            "ModelParameters": None,
            "Notes": notes,
        }
    
        # Extra metadata    
        experiment.update(kwargs)
        # Save    
        self._append(experiment)


    def load(self):
        """
        Load the experiment log.
        Returns
        -------
        pd.DataFrame
        """
        if not self.experiment_file.exists():
            print("No experiment log found.")
            return pd.DataFrame()
        return pd.read_csv(
            self.experiment_file,
        )

    def leaderboard(
        self,
        experiment_type=None,
        model=None,
        sort_by="CompetitionScore",
        ascending=True,
        top=10,
        columns=None,
    ):
        """
        Display the best experiments.
        Parameters
        ----------
        experiment_type : str or None
        model : str or None
        sort_by : str
        ascending : bool
        top : int
        Returns
        -------
        pd.DataFrame
        """
        # Load    
        df = self.load()
        if df.empty:
            return df
    
        # Filter experiment type    
        if experiment_type is not None:
            df = df[
                df["ExperimentType"] == experiment_type
            ]
    
        # Filter model    
        if model is not None:
            df = df[
                df["Model"] == model
            ]
    
        # Sort    
        df = df.sort_values(
            by=sort_by,
            ascending=ascending,
        )

        # Select columns        
        if columns is not None:
            df = df[columns]

        # Return top rows    
        return df.head(top)

    def best(
        self,
        experiment_type=None,
        model=None,
        sort_by="CompetitionScore",
        ascending=True,
    ):
        """
        Return the single best experiment.
        Parameters
        ----------
        experiment_type : str or None
        model : str or None
        sort_by : str
        ascending : bool
        Returns
        -------
        pd.Series
        """
        df = self.leaderboard(
            experiment_type=experiment_type,
            model=model,
            sort_by=sort_by,
            ascending=ascending,
            top=1,
        )
    
        if df.empty:
            return None
    
        return df.iloc[0]
        
    # # Main logger
    # def log(
    #     self,
    #     results,
    #     model_name=None,
    #     feature_selection=None,
    #     notes="",
    #     **kwargs,
    # ):

    #     """
    #     Append one experiment to experiment_log.csv

    #     Parameters
    #     ----------
    #     results : EvaluationResult
    #     model_name : optional
    #         Overrides results.model_name
    #     feature_selection : dict
    #         Metadata returned from FeatureSelector

    #     notes : str
    #     """

    #     if model_name is None:
    #         model_name = results.model_name

    #     feature_selection = feature_selection or {}

    #     # Automatically generate notes if omitted
    #     if not notes:
    #         strategy = feature_selection.get("selection_strategy", "None")
    #         top_n = feature_selection.get("top_n")
    #         percentile = feature_selection.get("percentile")
    #         if strategy == "top_n":
    #             notes = f"{model_name} | Top {top_n} Features"
    #         elif strategy == "percentile":
    #             notes = f"{model_name} | Top {percentile:.0%} Features"
    #         else:
    #             notes = model_name

    #     experiment = {
    #         # Experiment
    #         "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #         "Model": model_name,

    #         # Feature Selection
    #         "SelectionStrategy": feature_selection.get("selection_strategy"),
    #         "TopN": feature_selection.get("top_n"),
    #         "Percentile": feature_selection.get("percentile"),
    #         "UseCVImportance": feature_selection.get("use_cv_importance"),
    #         "FeaturesBefore": feature_selection.get("features_before"),
    #         "FeaturesAfter": feature_selection.get("features_after"),

    #         # Metrics
    #         "LogLoss": results.overall_logloss,
    #         "ROC_AUC": results.overall_auc,
    #         "CompetitionScore": results.overall_score,
    #         "TrainingTime": getattr(results, "training_time", None),

    #         # Extra
    #         "Notes": notes,
    #     }

    #     ####################################################

    #     experiment.update(kwargs)

    #     ####################################################
    #     experiment["ModelParameters"] = json.dumps(
    #         results.trained_models[0].get_params(),
    #         default=str,
    #     )
        
    #     ####################################################
    #     df = pd.DataFrame([experiment])
    #     if self.file.exists():
    #         old = pd.read_csv(self.file)

    #         df = pd.concat(
    #             [old, df],
    #             ignore_index=True,
    #         )

    #     df.to_csv(
    #         self.file,
    #         index=False,
    #     )

    #     print("\nExperiment saved.")
    #     print(df.tail(1))
