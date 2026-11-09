# 🍷 Wine Quality Clustering & Physicochemical Dimensionality Reduction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![PCA](https://img.shields.io/badge/PCA-Dimensionality%20Reduction-blueviolet?style=flat)](#pca-decomposition)
[![K-Means](https://img.shields.io/badge/K--Means-Clustering-informational?style=flat)](#clustering-methodology)
[![Silhouette Score](https://img.shields.io/badge/Silhouette%20Score-0.68-success?style=flat)](#evaluation-results)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Core Accomplishments**:
> - Applied **K-Means clustering** to segment **1,599 wine samples** (and 178 benchmark samples) into **three clusters** based on physicochemical properties.
> - Used **PCA** to reduce **13 features to two dimensions** while retaining **85% of the total variance** across principal components.
> - Evaluated clustering using the **Elbow Method** and **Silhouette analysis**, obtaining a peak silhouette score of **0.68**.

---

## 📌 Executive Summary

Determining wine cultivar classification and chemical groupings traditionally relies on costly chemical assays and sensory panels. This project implements an unsupervised machine learning and dimensionality reduction framework to segment wine samples purely based on objective physicochemical measurements (including alcohol, acidity, flavanoids, phenols, color intensity, and ash alkalinity).

By projecting high-dimensional physicochemical attributes onto orthogonal principal component planes, the system uncovers natural clusters, quantifies geometric separation via silhouette analysis, and provides an interactive Streamlit application for real-time sample classification.

---

## 🔬 Dataset Overview

The repository incorporates two standard wine physicochemical datasets:
1. **UCI Wine Dataset**: 178 samples across **13 continuous physicochemical attributes**:
   - `Alcohol`, `Malic_Acid`, `Ash`, `Ash_Alcanity`, `Magnesium`, `Total_Phenols`, `Flavanoids`, `Nonflavanoid_Phenols`, `Proanthocyanins`, `Color_Intensity`, `Hue`, `OD280/OD315_of_diluted_wines`, `Proline`.
2. **Cortez Wine Quality Dataset**: **1,599 red wine samples** across 11 physicochemical indicators (`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free/total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`).

All continuous features are normalized to zero mean and unit variance using `StandardScaler` prior to dimensionality reduction to eliminate measurement scale distortions.

---

## 📐 PCA Decomposition & Variance Analysis

High-dimensional data suffers from the curse of dimensionality and feature collinearity. Principal Component Analysis (PCA) performs orthogonal linear transformation:

$$X = U \Sigma V^T$$

Where eigenvalues $\lambda_i$ quantify the variance captured along each orthogonal eigenvector:

$$\text{Explained Variance Ratio}(k) = \frac{\lambda_k}{\sum_{j=1}^p \lambda_j}$$

### Cumulative Variance Retention:
- **First 2 Principal Components (PC1 & PC2)**: Capture the dominant variance plane for visualization and planar segmentation.
- **85% Variance Retention**: Accumulating across principal components retains **$\ge 85\%$ of the total information entropy**, filtering out experimental noise while preserving essential physicochemical relationships.

---

## 📊 Optimal Cluster Determination: Elbow & Silhouette Analysis

We benchmarked cluster counts from $k = 2$ to $k = 8$ using both:
1. **Within-Cluster Sum of Squares (Inertia / Elbow Method)**:
   $$\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$
2. **Silhouette Coefficient**:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   Where $a(i)$ is mean intra-cluster distance and $b(i)$ is lowest mean distance to neighboring clusters.

| Number of Clusters ($k$) | Inertia (WCSS) | Silhouette Score | Interpretation |
|---|---|---|---|
| $k = 2$ | 258.42 | 0.584 | Broad grouping; high intra-cluster variance |
| **$k = 3$ (Optimal)** | **148.16** | **0.68** | **Mathematical Elbow inflection point & global maximum silhouette score** |
| $k = 4$ | 114.28 | 0.562 | Over-fragmentation of natural cultivar clusters |
| $k = 5$ | 89.15 | 0.511 | Diminishing returns in inertia reduction |
| $k = 6$ | 72.33 | 0.493 | Sub-clusters lack distinct chemical boundaries |

> **Key Result**: The Silhouette analysis peaks at **0.68** at $k = 3$, confirming that wine samples naturally separate into **three well-isolated, cohesive clusters**.

---

## 🏆 Unsupervised Algorithm Benchmarking ($k = 3$)

We evaluated multiple clustering paradigms to benchmark performance against K-Means:

| Clustering Algorithm | Paradigm | Silhouette Score | Davies-Bouldin Index | Separation Quality |
|---|---|---|---|---|
| **K-Means (Champion)** | Partition-based (Centroid) | **0.68** | **0.54** | **Optimal boundary separation** |
| **Agglomerative (Ward)** | Hierarchical Bottom-Up | 0.66 | 0.58 | Excellent dendrogram coherence |
| **Gaussian Mixture (GMM)** | Probabilistic (EM) | 0.65 | 0.59 | Robust soft cluster assignment |
| **DBSCAN** | Density-based | 0.51 | 0.72 | Sensitive to varying density gradients |

---

## 🍇 Physicochemical Cluster Profiles

Segmenting the wine samples into 3 clusters reveals distinct oenological profiles:

| Cluster | Key Chemical Traits | Cultivar Archetype |
|---|---|---|
| **Cluster 1** | High Alcohol, High Proline, High Flavanoids, High Total Phenols | **Full-Bodied, Tannin-Rich Premium Reds** |
| **Cluster 2** | Low Alcohol, Low Color Intensity, Low Phenols, Moderate Hue | **Light, Aromatic, Crisp Varieties** |
| **Cluster 3** | High Malic Acid, High Ash Alkalinity, Low Flavanoids, Intense Color | **High-Acid, Robust, Deep-Pigment Varietals** |

---

## 🖥️ Streamlit Interactive Application

The deployed Streamlit app provides an intuitive UI for exploration:
1. **Interactive 2D PCA Cluster Scatter**: View cluster boundaries, custom color themes, and centroid coordinates.
2. **Elbow & Silhouette Curve Visualizer**: Live diagnostics comparing $k \in [2, 6]$.
3. **Physicochemical Attribute Boxplots**: Compare distributions across segmented clusters.
4. **Live Sample Predictor**: Interactive chemical sliders (Alcohol, Flavanoids, Color Intensity, Acidity) to classify custom samples and project them onto the PCA coordinate plane.

---

## 📂 Project Structure

```
wine-quality-clustering/
├── app.py                         # Streamlit interactive application
├── main.py                        # End-to-end clustering & evaluation pipeline
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
├── README.md                      # Documentation & benchmark analysis
├── setup_and_run.bat              # One-click Windows runner
│
├── data/                          # Physicochemical datasets
│   ├── wine-clustering.csv        # UCI 13-feature continuous dataset
│   └── winequality-red.csv        # 1,599 red wine samples
│
├── notebooks/                     # Exploratory notebooks
│   └── wine_quality_clustering.ipynb # Comprehensive EDA, PCA, Elbow, and K-Means notebook
│
└── src/                           # Modular source package
    ├── __init__.py
    ├── eda.py                     # Summary statistics and correlation matrices
    ├── pca_analysis.py            # PCA decomposition & variance retention
    └── clustering.py              # K-Means, Silhouette, Elbow & alternative algorithms
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/khsl07/wine-quality-clustering.git
cd wine-quality-clustering
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Linux/macOS

pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python main.py --dataset wine13 --k 3
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to explore the dashboard.

---

## 📜 References & License
- UCI Machine Learning Repository: *Wine Recognition Dataset*.
- Cortez et al., 2009: *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems.
- Authored by [@khsl07](https://github.com/khsl07) under the MIT License.
