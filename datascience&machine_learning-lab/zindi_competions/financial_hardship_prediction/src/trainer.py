"""
trainer.py

High-level experiment runner.
"""
# Imports
from pathlib import Path
import joblib

from evaluate import evaluate_model
from experiment import ExperimentTracker


# Trainer Class
class ModelTrainer:

    """
    Runs an entire ML experiment.
        ↓
    evaluate
        ↓
    log experiment
        ↓
    save model
        ↓
    return EvaluationResult
    """

    def __init__(self):
        self.tracker = ExperimentTracker()

    # This becomes the entire competition pipeline
    def run( self, model, X, y, cv, feature_count, notes="", categorical_features=None ):
        # Evaluate
        results = evaluate_model(
            model=model,
            X=X,
            y=y,
            cv=cv,
            categorical_features=categorical_features
        )

        # Log experiment
        experiment_id = self.tracker.log_experiment(
            results=results,
            parameters=model.get_params(),
            feature_count=feature_count,
            notes=notes,
        )

        # Automatically save every fold model
        for fold in results.trained_models:
            filename = (self.tracker.model_dir / f"{experiment_id}_Fold{fold['fold']}.pkl" )
        
            joblib.dump(
                fold["model"],
                filename,
            )

        #Nice experiment summary
        print()
        print("="*70)
        print("Experiment Finished")
        print("="*70)
        print(f"Experiment : {experiment_id}")
        print(f"Model      : {results.model_name}")
        print(f"Score      : {results.overall_score:.5f}")
        print(f"AUC        : {results.overall_auc:.5f}")
        print(f"LogLoss    : {results.overall_logloss:.5f}")
        print(f"Time       : {results.training_time:.1f} sec")
        print("="*70)

        # Return
        return results