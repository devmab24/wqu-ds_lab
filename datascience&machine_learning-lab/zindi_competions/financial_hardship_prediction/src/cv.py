# This module will manage every experiment

"""
Cross Validation Utilities
"""
from sklearn.model_selection import StratifiedKFold


def get_cv():

    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
