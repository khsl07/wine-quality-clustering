"""
Dimensionality Reduction Module using Principal Component Analysis (PCA)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any


def standardize_data(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Standardizes physicochemical features to zero mean and unit variance."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler


def compute_variance_explained(scaled_data: np.ndarray) -> Dict[str, Any]:
    """
    Computes full spectrum PCA to analyze individual and cumulative variance explained.
    Demonstrates how variance accumulates across components up to 85%+ retention.
    """
    pca_full = PCA()
    pca_full.fit(scaled_data)
    
    indiv_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(indiv_var)
    
    # Find component index reaching >= 85% variance
    n_85 = int(np.argmax(cum_var >= 0.85) + 1)
    
    return {
        "individual_variance": indiv_var.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "n_components_for_85pct": n_85,
        "pca_model": pca_full
    }


def apply_pca(scaled_data: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Applies PCA reduction to n_components (typically 2 for 2D projection).
    Returns transformed embedding and the fitted PCA model.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(scaled_data)
    return X_pca, pca


def get_feature_loadings(pca_model: PCA, feature_names: list) -> pd.DataFrame:
    """Calculates feature loadings (eigenvector weights) for principal components."""
    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=[f"PC{i+1}" for i in range(pca_model.components_.shape[0])],
        index=feature_names
    )
    return loadings
