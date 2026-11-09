"""
Exploratory Data Analysis (EDA) Module for Wine Physicochemical Datasets
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates descriptive statistics, missing value audit, and duplicate counts."""
    stats = {
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "summary_table": df.describe().round(3),
        "skewness": df.skew().round(3).to_dict()
    }
    return stats


def plot_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise Pearson correlation matrix for all physicochemical attributes."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(3)
    return corr
