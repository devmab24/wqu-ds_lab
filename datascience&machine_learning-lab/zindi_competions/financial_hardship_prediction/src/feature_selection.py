"""
feature_selection.py

Competition Feature Selection Pipeline
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold

from lightgbm import LGBMClassifier
from sklearn.base import clone


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Stage 1 Feature Selection

    1. Remove constant features
    2. Remove duplicate features
    3. Remove highly correlated features
    """

    def __init__(
        self,
        variance_threshold=0.0,
        correlation_threshold=0.95,
        remove_constant=True,
        remove_duplicates=True,
        remove_correlated=True,
        use_lightgbm=False,
        selection_strategy="threshold",
        importance_threshold=0,
        top_n=None,
        percentile=None,
        use_cv_importance=False,
        cv=None,
        
        output_dir="../outputs/feature_selection",
        verbose=True,
    ):

        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

        self.remove_constant = remove_constant
        self.remove_duplicates = remove_duplicates
        self.remove_correlated = remove_correlated
        
        self.verbose = verbose

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.constant_features = []
        self.duplicate_features = []
        self.correlated_features = []

        self.selected_columns = None

        self.percentile = percentile
        self.low_importance_features = []
        self.feature_importance = None
        
        self.use_lightgbm = use_lightgbm
        self.selection_strategy = selection_strategy
        self.importance_threshold = importance_threshold
        self.top_n = top_n
        self.use_cv_importance = use_cv_importance
        self.cv = cv

    ###########################################################
    # Fit
    ###########################################################
    def fit(self, X, y=None):
        # Number of features before any selection
        self.original_feature_count = X.shape[1]
        
        X = X.copy()
        if self.verbose:
            print("=" * 60)
            print("Running Feature Selection")
            print("=" * 60)

        # Stage 1
        if self.remove_constant:
            X = self._remove_constant(X)
        
        # Stage 2
        if self.remove_duplicates:
            X = self._remove_duplicates(X)
        
        # Stage 3
        if self.remove_correlated:
            X = self._remove_correlated(X)

        #Stage 4
        if self.use_lightgbm:
            X = self._lightgbm_selection(X, y,)
        
        self.selected_columns = X.columns.tolist()
        
        self._save_summary()
        
        if self.verbose:
            print("\nFeature Selection Finished")
            print(f"Remaining Features : {len(self.selected_columns)}")
            print("=" * 60)

        self.selection_summary = {
            "selection_strategy": self.selection_strategy,
            "top_n": self.top_n,
            "percentile": self.percentile,
            "use_cv_importance": self.use_cv_importance,
            "features_before": self.original_feature_count,
            "features_after": len(self.selected_columns),
        }
        
        return self

    ###########################################################
    # Transform
    ###########################################################
    def transform(self, X):
        return X[self.selected_columns].copy()

    ###########################################################
    # Fit Transform
    ###########################################################
    def fit_transform(self, X, y=None):
        self.fit(X, y)

        return self.transform(X)

    ###########################################################
    # Constant Features
    ###########################################################
    def _remove_constant(self, X):
        numeric_cols = X.select_dtypes(include=np.number).columns
        selector = VarianceThreshold(
            threshold=self.variance_threshold
        )
        
        selector.fit(X[numeric_cols])
        kept = numeric_cols[selector.get_support()]
        self.constant_features = list( set(numeric_cols) - set(kept) )

        if self.verbose:
            print( f"Constant Features Removed : {len(self.constant_features)}" )

        X = X.drop(columns=self.constant_features)

        return X

    ###########################################################
    # Duplicate Features
    ###########################################################
    def _remove_duplicates(self, X):
        duplicates = []
        cols = X.columns

        for i in range(len(cols)):
            col1 = cols[i]
            for j in range(i + 1, len(cols)):
                col2 = cols[j]
                if X[col1].equals(X[col2]):
                    duplicates.append(col2)
                    
        self.duplicate_features = duplicates
        if self.verbose:
            print( f"Duplicate Features Removed : {len(duplicates)}" )

        return X.drop(columns=duplicates)

    ###########################################################
    # Correlated Features
    ###########################################################
    def _remove_correlated(self, X):
        numeric = X.select_dtypes(include=np.number)
        corr = numeric.corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1 ).astype(bool))

        self.correlated_features = [ column for column in upper.columns if any( upper[column] > self.correlation_threshold)]
        if self.verbose:
            print( f"Correlated Features Removed : " f"{len(self.correlated_features)}" )

        return X.drop(columns=self.correlated_features)

    ###########################################################
    # Stage 4
    ###########################################################
    def _lightgbm_selection( self, X, y):
        """
        Remove features with very low LightGBM importance.
        """
        model = LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )

        # Compute Feature Importance
        if self.use_cv_importance:
            importance = self._cross_validated_importance( X, y,)
        else:
            model.fit(X, y)
            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_,
            })
        
            importance = (
                importance
                .sort_values( "Importance", ascending=False, )
                .reset_index(drop=True)
            )

        importance = importance.sort_values( "Importance", ascending=False, ).reset_index(drop=True)
        self.feature_importance = importance.copy()

        # Threshold Strategy
        if self.selection_strategy == "threshold":
            remove = importance[ importance["Importance"] <= self.importance_threshold ]
        
        # Top N Strategy       
        elif self.selection_strategy == "top_n":
            remove = importance.iloc[self.top_n:]
        
        # Percentile Strategy
        elif self.selection_strategy == "percentile":
            n_remove = int(len(importance) * self.percentile)
            remove = importance.tail(n_remove)
        
        # Median Strategy       
        elif self.selection_strategy == "median":
            median = importance["Importance"].median()
            remove = importance[ importance["Importance"] < median ]
            
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

        #Get low_importance features to drop
        self.low_importance_features = remove["Feature"].tolist()
        if self.verbose:
            print( f"Low Importance Features Removed : " f"{len(self.low_importance_features)}" )
        
        # Drop low importance features
        X = X.drop( columns=self.low_importance_features )
        
        # Updated importance features
        self.selected_columns = X.columns.tolist()
        
        #Return updated features
        return X
        
    ###########################################################
    # Cross validation importance
    ###########################################################
    def _cross_validated_importance(self, X, y):
        """
        Compute average feature importance across folds.
        """
        if self.cv is None:
            raise ValueError(
                "cv must be provided when use_cv_importance=True."
            )
        importance_list = []
    
        for fold, (train_idx, valid_idx) in enumerate( self.cv.split(X, y), start=1,):
            print(f"Feature Selection Fold {fold}")
            model = LGBMClassifier(
                objective="binary",
                random_state=42,
                verbosity=-1,
            )
    
            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )
    
            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_,
            })
    
            importance_list.append(importance)
    
        importance = pd.concat(importance_list)
        importance = (
            importance
            .groupby("Feature")["Importance"]
            .mean()
            .reset_index()
            .sort_values( "Importance", ascending=False, )
            .reset_index(drop=True)
        )
    
        return importance

    ###########################################################
    # Save Reports
    ###########################################################
    def _save_summary(self):
        pd.DataFrame({ "Constant": self.constant_features }).to_csv( 
            self.output_dir / "removed_constant.csv", index=False)

        pd.DataFrame({ "Duplicate": self.duplicate_features}).to_csv( 
            self.output_dir / "removed_duplicates.csv", index=False,)

        pd.DataFrame({ "Correlated": self.correlated_features }).to_csv( 
            self.output_dir / "removed_correlated.csv", index=False,)
        
        pd.DataFrame({  "LowImportance": self.low_importance_features }).to_csv(
            self.output_dir / "removed_low_importance.csv", index=False,)
        
        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                self.output_dir / "lightgbm_importance.csv", index=False, )

        pd.DataFrame({ "Selected_Features": self.selected_columns}).to_csv(
            self.output_dir / "selected_features.csv", index=False,)

        summary = pd.DataFrame({
            "Metric": [
                "Constant",
                "Duplicate",
                "Correlated",
                "Low Importance",
                "Remaining"
            ],
            "Count": [
                len(self.constant_features),
                len(self.duplicate_features),
                len(self.correlated_features),
                len(self.low_importance_features),
                len(self.selected_columns),
            ]
        })

        summary.to_csv(self.output_dir / "feature_selection_summary.csv", index=False,)