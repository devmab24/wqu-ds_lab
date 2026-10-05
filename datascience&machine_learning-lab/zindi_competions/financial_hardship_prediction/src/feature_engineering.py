#The code below works perfectly fine, but I want to test a new optimized version
"""
Competition Feature Engineering Pipeline
"""

import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.monthly_groups = {}

    def fit(self, X, y=None):
        self.monthly_groups = self._identify_monthly_groups(X)

        return self

    def transform(self, X):
        X = X.copy()
        
        # Stage 1
        X = self._statistical_features(X)
        
        # Stage 2
        X = self._trend_features(X)
        X = self._momentum_features(X)
        X = self._volatility_features(X)

        # Stage 3
        X = self._financial_health_features(X)
        X = self._financial_summary_features(X)

        #Not yet defined
        # X = self._ratio_features(X)
        # X = self._interaction_features(X)
        X = X.copy()
    
        return X

    #########################################
    # Helper Methods
    #########################################
    def _identify_monthly_groups(self, df):
        pattern = re.compile(r"m([1-6])_(.+)")
    
        groups = {}
    
        for col in df.columns:
            match = pattern.match(col)
            if match:
                month = int(match.group(1))
                feature = match.group(2)
                groups.setdefault(feature, []).append(col)
    
        for feature in groups:
            groups[feature] = sorted(
                groups[feature],
                key=lambda x: int(re.match(r"m([1-6])_", x).group(1))
            )
    
        return groups

    def _month_matrix(self, df, cols):
        """
        Helper function:
        Returns the six monthly columns as a NumPy array.
        """
    
        return df[cols].to_numpy(dtype=float)

    #########################################
    # Stage 1
    #########################################
    def _statistical_features(self, df):
        new_features = {}
        for feature, cols in self.monthly_groups.items():
            values = df[cols]
            new_features[f"{feature}_mean"] = values.mean(axis=1)
            new_features[f"{feature}_std"] = values.std(axis=1)
            new_features[f"{feature}_min"] = values.min(axis=1)
            new_features[f"{feature}_max"] = values.max(axis=1)
            new_features[f"{feature}_median"] = values.median(axis=1)
            new_features[f"{feature}_range"] = (
                values.max(axis=1) - values.min(axis=1)
            )
        new_features = pd.DataFrame(
            new_features,
            index=df.index
        )
    
        return pd.concat(
            [df, new_features],
            axis=1
        )

    #########################################
    # Stage 2
    #########################################
    def _trend_features(self, df):
        """
        Create linear trend (slope) features for each monthly feature group.
        """
        months = np.arange(1, 7)
        new_features = {}
    
        for feature, cols in self.monthly_groups.items():
            values = self._month_matrix(df, cols)
    
            # Fit a straight line across the 6 months
            slopes = np.polyfit(months, values.T, deg=1)[0]
            new_features[f"{feature}_trend"] = slopes
    
        # Create all new columns at once
        trend_df = pd.DataFrame(
            new_features,
            index=df.index
        )
    
        # Concatenate once (avoids DataFrame fragmentation)
        df = pd.concat(
            [df, trend_df],
            axis=1
        )
    
        return df

    def _momentum_features(self, df):
        """
        Create momentum-based features from the six monthly observations.
        """
        new_features = {}
        for feature, cols in self.monthly_groups.items():
            values = self._month_matrix(df, cols)
            first_half = values[:, :3].mean(axis=1)
            second_half = values[:, 3:].mean(axis=1)
    
            new_features[f"{feature}_momentum"] = (
                second_half - first_half
            )
    
            new_features[f"{feature}_last_first_ratio"] = ( values[:, -1] + 1 ) / ( values[:, 0] + 1 )
            new_features[f"{feature}_pct_change"] = ( values[:, -1] - values[:, 0]) / ( values[:, 0] + 1 )
    
        momentum_df = pd.DataFrame(
            new_features,
            index=df.index
        )
    
        return pd.concat(
            [df, momentum_df],
            axis=1
        )

    def _volatility_features(self, df):
        """
        Create volatility and month-to-month change features.
        """
        new_features = {}
        for feature, cols in self.monthly_groups.items():
            values = self._month_matrix(df, cols)
            mean = values.mean(axis=1)
            std = values.std(axis=1)
            new_features[f"{feature}_cv"] = ( std / (mean + 1) )
            diffs = np.diff(values, axis=1)
            new_features[f"{feature}_avg_change"] = ( diffs.mean(axis=1))
            new_features[f"{feature}_max_change"] = ( diffs.max(axis=1) )
            new_features[f"{feature}_min_change"] = ( diffs.min(axis=1) )
            new_features[f"{feature}_increase_count"] = ((diffs > 0).sum(axis=1) )
            new_features[f"{feature}_decrease_count"] = ( (diffs < 0).sum(axis=1))
    
        volatility_df = pd.DataFrame(
            new_features,
            index=df.index
        )
    
        return pd.concat(
            [df, volatility_df],
            axis=1
        )

    #########################################
    # Stage 3
    #########################################
    def _financial_health_features(self, df):
        """
        Create monthly financial health indicators.
        """    
        eps = 1
        new_features = {}
    
        for month in range(1, 7):
            received = f"m{month}_received_total_value"
            withdraw = f"m{month}_withdraw_total_value"
            deposit = f"m{month}_deposit_total_value"
            send = f"m{month}_mm_send_total_value"
            merchant = f"m{month}_merchantpay_total_value"
            balance = f"m{month}_daily_avg_bal"
    
            # Net Cash Flow
            if received in df.columns and withdraw in df.columns:
                new_features[f"m{month}_net_cashflow"] = ( df[received] - df[withdraw] )
    
            # Income Spent Ratio
            if received in df.columns and send in df.columns:
                new_features[f"m{month}_income_spent_ratio"] = ( df[send] / (df[received] + eps))
    
            # Withdrawal / Deposit Ratio
            if withdraw in df.columns and deposit in df.columns:
                new_features[f"m{month}_withdraw_deposit_ratio"] = ( df[withdraw] / (df[deposit] + eps) )
    
            # Merchant Dependency
            if merchant in df.columns and received in df.columns:
                new_features[f"m{month}_merchant_ratio"] = ( df[merchant] / (df[received] + eps) )
    
            # Liquidity Buffer
            if balance in df.columns and received in df.columns:
                new_features[f"m{month}_liquidity_buffer"] = ( df[balance] / (df[received] + eps) )
    
        health_df = pd.DataFrame(
            new_features,
            index=df.index,
        )
    
        return pd.concat(
            [df, health_df],
            axis=1,
        )

    def _financial_summary_features(self, df):
        """
        Aggregate the monthly financial health features.
        """
        health_features = [
            "net_cashflow",
            "income_spent_ratio",
            "withdraw_deposit_ratio",
            "merchant_ratio",
            "liquidity_buffer",
        ]
    
        new_features = {}
        for feature in health_features:
            cols = [ c for c in df.columns if c.startswith("m") and c.endswith(feature) ]
    
            if len(cols) != 6:
                continue
    
            values = self._month_matrix(df, cols)
            new_features[f"{feature}_mean"] = values.mean(axis=1)
            new_features[f"{feature}_std"] = values.std(axis=1)
            new_features[f"{feature}_trend"] = ( values[:, -1] - values[:, 0] )
            new_features[f"{feature}_max"] = values.max(axis=1)
            new_features[f"{feature}_min"] = values.min(axis=1)
    
        summary_df = pd.DataFrame(
            new_features,
            index=df.index,
        )
    
        return pd.concat(
            [df, summary_df],
            axis=1,
        )