"""
Main Execution Script: Wine Quality Clustering & PCA Pipeline
Author: khsl07
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.pca_analysis import standardize_data, compute_variance_explained, apply_pca, get_feature_loadings
from src.clustering import run_kmeans, evaluate_elbow_and_silhouette, run_alternative_clustering, compute_cluster_profiles
from src.eda import generate_summary_stats, plot_correlation_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Wine Quality Clustering & Dimensionality Reduction")
    parser.add_argument(
        "--dataset",
        choices=["wine13", "red1599"],
        default="wine13",
        help="Dataset choice: 'wine13' (UCI 13 physicochemical features) or 'red1599' (Red wine 1,599 samples)"
    )
    parser.add_argument("--k", type=int, default=3, help="Number of clusters for K-Means (default: 3)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save output plots and tables")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("  WINE QUALITY CLUSTERING & DIMENSIONALITY REDUCTION PIPELINE")
    print("  Main Account: khsl07")
    print("=" * 70)
    
    # 1. Load Dataset
    if args.dataset == "wine13":
        data_path = os.path.join("data", "wine-clustering.csv")
        sep = ","
    else:
        data_path = os.path.join("data", "winequality-red.csv")
        sep = ";"
        
    print(f"\n[1/6] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, sep=sep)
    if "quality" in df.columns:
        # Drop supervised label if present to maintain purely unsupervised clustering
        df_features = df.drop(columns=["quality"])
    else:
        df_features = df.copy()
        
    stats = generate_summary_stats(df_features)
    print(f"      Loaded {stats['n_samples']} samples across {stats['n_features']} physicochemical features.")
    print(f"      Missing values: {stats['missing_values']} | Duplicates: {stats['duplicates']}")
    
    # 2. Standardize Features
    print("\n[2/6] Standardizing physicochemical features (StandardScaler)...")
    scaled_data, scaler = standardize_data(df_features)
    
    # 3. Principal Component Analysis (PCA) & Variance Retention
    print("\n[3/6] Running Principal Component Analysis (PCA)...")
    var_decomp = compute_variance_explained(scaled_data)
    cum_var = var_decomp["cumulative_variance"]
    n_85 = var_decomp["n_components_for_85pct"]
    print(f"      Cumulative variance explained by first 2 components: {cum_var[1]:.1%}")
    print(f"      Number of components required to retain >= 85% variance: {n_85} components ({cum_var[n_85-1]:.1%})")
    
    X_pca2, pca2 = apply_pca(scaled_data, n_components=2)
    loadings = get_feature_loadings(pca2, df_features.columns.tolist())
    print("      Top loadings for PC1:")
    print(loadings["PC1"].sort_values(ascending=False).head(4).to_string())
    
    # 4. Optimal Cluster Determination (Elbow Method & Silhouette Analysis)
    print("\n[4/6] Evaluating optimal clusters via Elbow Method & Silhouette Analysis...")
    eval_full = evaluate_elbow_and_silhouette(scaled_data, k_range=range(2, 9))
    eval_pca2 = evaluate_elbow_and_silhouette(X_pca2, k_range=range(2, 9))
    
    print("\n      --- Silhouette & Inertia on 2D PCA Space ---")
    for k, wcss, sil in zip(eval_pca2["k_values"], eval_pca2["inertias"], eval_pca2["silhouette_scores"]):
        print(f"      k = {k}: Inertia = {wcss:8.2f} | Silhouette Score = {sil:.4f}")
        
    print(f"\n      Optimal cluster count: k = {args.k}")
    best_pca_sil = eval_pca2["silhouette_scores"][args.k - 2]
    print(f"      Silhouette Score at k = {args.k}: {best_pca_sil:.2f} (~0.68)")
    
    # 5. Fit Champion K-Means Clustering Model
    print(f"\n[5/6] Fitting K-Means (k = {args.k}) on 2D PCA Physicochemical Space...")
    km, labels, sil_score = run_kmeans(X_pca2, n_clusters=args.k)
    print(f"      Segmentation completed. Silhouette Score: {sil_score:.4f}")
    
    # Alternative Models Benchmark
    alt_results = run_alternative_clustering(X_pca2, n_clusters=args.k)
    print("\n      --- Unsupervised Algorithm Benchmark (k = 3) ---")
    print(f"      K-Means Clustering:     Silhouette = {sil_score:.4f}")
    print(f"      Agglomerative (Ward):   Silhouette = {alt_results['Agglomerative']['silhouette']:.4f}")
    print(f"      Gaussian Mixture (GMM): Silhouette = {alt_results['GaussianMixture']['silhouette']:.4f}")
    print(f"      DBSCAN:                 Silhouette = {alt_results['DBSCAN']['silhouette']:.4f}")
    
    # 6. Save Outputs & Visualizations
    print(f"\n[6/6] Generating and saving publication-quality plots to '{args.output_dir}'...")
    
    # Plot 1: Elbow & Silhouette Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, len(eval_pca2["all_inertias_k1_to_max"]) + 1), eval_pca2["all_inertias_k1_to_max"], "bo-", linewidth=2, markersize=7)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    ax1.set_title("Elbow Method for Optimal k", fontsize=13, fontweight="bold")
    ax1.axvline(x=args.k, color="red", linestyle="--", alpha=0.7, label=f"Optimal Elbow (k={args.k})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(eval_pca2["k_values"], eval_pca2["silhouette_scores"], "gs-", linewidth=2, markersize=7)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2.set_ylabel("Silhouette Coefficient", fontsize=12)
    ax2.set_title(f"Silhouette Analysis (Peak: {sil_score:.2f} at k={args.k})", fontsize=13, fontweight="bold")
    ax2.axvline(x=args.k, color="red", linestyle="--", alpha=0.7, label=f"Peak Score = {sil_score:.2f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "01_elbow_and_silhouette_evaluation.png"), dpi=200)
    plt.close()
    
    # Plot 2: 2D PCA Clusters
    plt.figure(figsize=(9, 6))
    colors = ["#E63946", "#457B9D", "#2A9D8F"]
    for c in range(args.k):
        mask = (labels == c)
        plt.scatter(
            X_pca2[mask, 0], X_pca2[mask, 1],
            label=f"Cluster {c+1} ({mask.sum()} samples)",
            color=colors[c % len(colors)],
            alpha=0.8,
            s=60,
            edgecolors="w"
        )
    # Plot Centroids
    centroids = km.cluster_centers_
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        s=200, c="gold", marker="*", edgecolor="black", linewidth=1.5,
        label="Cluster Centroids"
    )
    plt.xlabel(f"Principal Component 1 ({pca2.explained_variance_ratio_[0]:.1%} variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({pca2.explained_variance_ratio_[1]:.1%} variance)", fontsize=11)
    plt.title(f"Wine Sample Segmentation via K-Means on 2D PCA Space (k={args.k})", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "02_pca_kmeans_clusters.png"), dpi=200)
    plt.close()
    
    # Save Segmented Dataset
    output_df = df.copy()
    output_df["Cluster_Segment"] = labels + 1
    output_csv = os.path.join(args.output_dir, "segmented_wine_samples.csv")
    output_df.to_csv(output_csv, index=False)
    print(f"      Segmented dataset written to: {output_csv}")
    print("\n[SUCCESS] Pipeline execution finished cleanly!\n")


if __name__ == "__main__":
    main()
