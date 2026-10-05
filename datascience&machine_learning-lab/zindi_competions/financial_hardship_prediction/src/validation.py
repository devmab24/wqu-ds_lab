"""
Dataset Validation Module
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold


class DatasetValidator(BaseEstimator, TransformerMixin):
    """
    This validator class will:
    - Remove constant and near-constant features.
    - Remove highly correlated features.
    - Identify potential leakage or duplicate information.
    - Rank features using model-based importance (e.g., LightGBM or CatBoost).
    - Produce a leaner, higher-quality training matrix.
    """
    def __init__(
        self,
        id_column="ID",
        corr_threshold=0.98,
        variance_threshold=0.0,
    ):
    
        self.id_column = id_column
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        #############################################
        # Learned during fit()
        self.identifier_columns_ = []
        self.duplicate_columns_ = []
        self.constant_columns_ = []
        self.low_variance_columns_ = []
        self.high_corr_columns_ = []
        self.feature_names_ = None
        
    # def __init__(
    #     self,
    #     # target,
    #     id_column="ID",
    #     corr_threshold=0.98,
    #     variance_threshold=0.0,
    # ):
    #     #Properties
    #     # self.target = target
    #     self.id_column = id_column
    #     self.corr_threshold = corr_threshold
    #     self.variance_threshold = variance_threshold
    #     self.columns_to_drop = []
    #     self.removed_identifier = []
    #     self.removed_constant = []
    #     self.removed_duplicate = []
    #     self.removed_correlation = []

    def fit(self, X, y=None):
        """
        Learn which features should be removed.
        """
    
        X = X.copy()
        ####################################################
        # Identifier
        if self.id_column in X.columns:
            self.identifier_columns_ = [self.id_column]
            X = X.drop(columns=self.identifier_columns_)
    
        ####################################################
        # Duplicate columns
        duplicated = X.T.duplicated()
        self.duplicate_columns_ = X.columns[
            duplicated
        ].tolist()
    
        X = X.drop(columns=self.duplicate_columns_)

        ####################################################
        # Constant features
        self.constant_columns_ = [
            c
            for c in X.columns
            if X[c].nunique() <= 1
        ]
    
        X = X.drop(columns=self.constant_columns_)
        
        ####################################################
        # Low variance
        numeric_cols = X.select_dtypes(
            include=np.number
        ).columns.tolist()
    
        selector = VarianceThreshold(
            threshold=self.variance_threshold
        )
    
        selector.fit(X[numeric_cols])
    
        keep = selector.get_support()
    
        keep_numeric = list(
            np.array(numeric_cols)[keep]
        )
    
        self.low_variance_columns_ = [
            c
            for c in numeric_cols
            if c not in keep_numeric
        ]
    
        X = X.drop(columns=self.low_variance_columns_)
    
        ####################################################
        # High correlation
        numeric = X.select_dtypes(
            include=np.number
        )
    
        corr = numeric.corr().abs()
        upper = corr.where(
            np.triu(
                np.ones(corr.shape),
                k=1,
            ).astype(bool)
        )
    
        self.high_corr_columns_ = [
            c
            for c in upper.columns
            if any(
                upper[c] > self.corr_threshold
            )
        ]
    
        X = X.drop(columns=self.high_corr_columns_)
    
        ####################################################
        self.feature_names_ = X.columns.tolist()

        print("=" * 60)
        print("Dataset Validation Summary")
        print("=" * 60)
        
        print(f"Identifier Removed         : {len(self.identifier_columns_)}")
        print(f"Duplicate Features Removed: {len(self.duplicate_columns_)}")
        print(f"Constant Features Removed : {len(self.constant_columns_)}")
        print(f"Low Variance Removed      : {len(self.low_variance_columns_)}")
        print(f"High Correlation Removed  : {len(self.high_corr_columns_)}")
        
        print(f"Remaining Features        : {len(self.feature_names_)}")
        
        print("=" * 60)
    
        return self
        
    # def fit(self, X, y=None):
    #     return self

    def transform(self, X):
        X = X.copy()
    
        ####################################################
        # Apply learned removals
        drop_cols = (
            self.identifier_columns_
            + self.duplicate_columns_
            + self.constant_columns_
            + self.low_variance_columns_
            + self.high_corr_columns_
        )
    
        existing = [
            c
            for c in drop_cols
            if c in X.columns
        ]
    
        X = X.drop(
            columns=existing
        )
    
        ####################################################
        # Ensure feature alignment
        missing = [
            c
            for c in self.feature_names_
            if c not in X.columns
        ]
    
        if missing:
            raise ValueError(
                f"Missing columns during transform: {missing}"
            )
    
        X = X[self.feature_names_]
    
        ####################################################
        self._check_missing(X)
        self._check_inf(X)
    
        return X
    
    # def transform(self, X):

    #     X = X.copy()
    #     X = self._remove_identifier(X)
    #     X = self._remove_duplicate_features(X)
    #     X = self._remove_constant_features(X)
    #     X = self._remove_low_variance(X)
    #     X = self._remove_high_correlation(X)
    #     X = self._check_missing(X)
    #     X = self._check_inf(X)

    #     return X

    # #Method that checks and drops Identifier
    # def _remove_identifier(self, df):
    #     if self.id_column in df.columns:
    #         df = df.drop(columns=self.id_column)
    
    #     return df

    # #Duplicate Columns removal method
    # def _remove_duplicate_features(self, df):
    #     duplicated = df.T.duplicated()
    #     cols = df.columns[duplicated]
    #     df = df.drop(columns=cols)
    
    #     return df

    # #Constant Features
    # def _remove_constant_features(self, df):
    #     """
    #     Remove constant features.
    #     """
    #     constant = [c for c in df.columns if df[c].nunique() <= 1 ]
    #     if constant:
    #         print(f"Constant Features Removed: {len(constant)}")
    
    #     return df.drop(columns=constant)

    # # Low Variance
    # def _remove_low_variance(self, df):
    #     """
    #     Remove low-variance numeric features.
    #     """
    #     numeric_cols = df.select_dtypes( include=np.number).columns.tolist()
    #     selector = VarianceThreshold(
    #         threshold=self.variance_threshold
    #     )
    
    #     selector.fit(df[numeric_cols])
    #     keep = selector.get_support()
    #     keep_cols = list(
    #         np.array(numeric_cols)[keep]
    #     )
    
    #     categorical_cols = df.select_dtypes(
    #         exclude=np.number
    #     ).columns.tolist()
    
    #     final_cols = keep_cols + categorical_cols
    
    #     return df[final_cols]
        
        # numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
        # # Never remove the target
        # feature_cols = [c for c in numeric_cols if c != self.target]
        # selector = VarianceThreshold(
        #     threshold=self.variance_threshold
        # )
    
        # selector.fit(df[feature_cols])
        # keep = selector.get_support()
        # keep_cols = list(np.array(feature_cols)[keep])
    
        # # Keep all categorical columns
        # categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        # final_cols = keep_cols + categorical_cols #+ [self.target]
        # if self.target in df.columns:
        #     final_cols.append(self.target)
    
        # return df[final_cols]


    # # Correlation Filter
    # def _remove_high_correlation(self, df):
    #     """
    #     Remove highly correlated numeric features.
    #     """    
    #     numeric = df.select_dtypes(
    #         include=np.number
    #     )
    
    #     corr = numeric.corr().abs()
    
    #     upper = corr.where(
    #         np.triu(
    #             np.ones(corr.shape),
    #             k=1,
    #         ).astype(bool)
    #     )
    
    #     drop = [ col for col in upper.columns if any( upper[col] > self.corr_threshold )]
    #     print(f"Highly Correlated Features Removed: {len(drop)}")
    
    #     return df.drop(columns=drop)
        
    # def _remove_high_correlation(self, df):
    #     numeric = df.select_dtypes(include=np.number)
    #     feature_cols = [ c for c in numeric.columns if c != self.target ]
    #     corr = numeric[feature_cols].corr().abs()
    #     upper = corr.where(
    #         np.triu(np.ones(corr.shape), k=1).astype(bool)
    #     )
        
    #     drop = [ col for col in upper.columns if any(upper[col] > self.corr_threshold) ]
    #     print(f"Highly Correlated Features Removed: {len(drop)}")
    
    #     return df.drop(columns=drop)

    # Missing Values
    def _check_missing(self, df):
        missing = df.isna().sum().sum()
        print(f"Missing Values : {missing}")
    
        return df

    # Infinity Check
    def _check_inf(self, df):
        numeric = df.select_dtypes(include=np.number)
        inf = np.isinf(numeric).sum().sum()
        print(f"Infinite Values : {inf}")
    
        return df

    