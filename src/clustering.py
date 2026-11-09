"""
Unsupervised Clustering Module: K-Means, Elbow Method & Silhouette Analysis
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Tuple, Any


def evaluate_elbow_and_silhouette(
    data: np.ndarray,
    k_range: range = range(2, 11)
) -> Dict[str, Any]:
    """
    Evaluates optimal cluster count using:
      1. Elbow Method (Inertia / Within-Cluster Sum of Squares)
      2. Silhouette Analysis (Mean Silhouette Coefficient)
      3. Davies-Bouldin & Calinski-Harabasz validation scores
    """
    inertias = []
    silhouette_scores = []
    db_scores = []
    ch_scores = []
    k_values = list(k_range)
    
    # Inertia for k=1
    km1 = KMeans(n_clusters=1, random_state=42, n_init=10).fit(data)
    all_inertias = [km1.inertia_]
    
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(data)
        
        inertias.append(km.inertia_)
        all_inertias.append(km.inertia_)
        
        sil = silhouette_score(data, labels)
        db = davies_bouldin_score(data, labels)
        ch = calinski_harabasz_score(data, labels)
        
        silhouette_scores.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)
        
    best_idx = int(np.argmax(silhouette_scores))
    best_k = k_values[best_idx]
    best_silhouette = silhouette_scores[best_idx]
    
    return {
        "k_values": k_values,
        "inertias": inertias,
        "all_inertias_k1_to_max": all_inertias,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": db_scores,
        "calinski_harabasz_scores": ch_scores,
        "optimal_k_by_silhouette": best_k,
        "max_silhouette_score": best_silhouette
    }


def run_kmeans(
    data: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray, float]:
    """
    Fits K-Means clustering algorithm to segment wine samples into n_clusters.
    Returns the fitted model, cluster labels, and silhouette score.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(data)
    sil_score = float(silhouette_score(data, labels))
    return km, labels, sil_score


def run_alternative_clustering(
    data: np.ndarray,
    n_clusters: int = 3
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmarks alternative unsupervised clustering algorithms:
      - Agglomerative Hierarchical Clustering (Ward linkage)
      - Gaussian Mixture Model (Expectation-Maximization)
      - DBSCAN (Density-Based Spatial Clustering)
    """
    results = {}
    
    # 1. Agglomerative Hierarchical
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    agg_labels = agg.fit_predict(data)
    results["Agglomerative"] = {
        "labels": agg_labels,
        "silhouette": float(silhouette_score(data, agg_labels)),
        "davies_bouldin": float(davies_bouldin_score(data, agg_labels))
    }
    
    # 2. Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(data)
    results["GaussianMixture"] = {
        "labels": gmm_labels,
        "silhouette": float(silhouette_score(data, gmm_labels)),
        "davies_bouldin": float(davies_bouldin_score(data, gmm_labels))
    }
    
    # 3. DBSCAN
    db = DBSCAN(eps=2.3, min_samples=5)
    db_labels = db.fit_predict(data)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    db_sil = float(silhouette_score(data, db_labels)) if n_db_clusters > 1 else 0.0
    results["DBSCAN"] = {
        "labels": db_labels,
        "n_clusters_found": n_db_clusters,
        "silhouette": db_sil
    }
    
    return results


def compute_cluster_profiles(
    raw_df: pd.DataFrame,
    labels: np.ndarray
) -> pd.DataFrame:
    """Computes mean and std of physicochemical properties across segmented wine clusters."""
    df_clustered = raw_df.copy()
    df_clustered["Cluster"] = labels
    profiles = df_clustered.groupby("Cluster").agg(["mean", "std"])
    return profiles
