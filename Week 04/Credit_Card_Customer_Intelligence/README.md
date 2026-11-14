
# Credit Card Customer Intelligence

Unsupervised learning project using Kaggle's [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

## Dataset setup
Download the CSV from Kaggle and place it at `data/CC GENERAL.csv`. The dataset is not included in this ZIP.

## Install and run (PowerShell)
From this project's root folder:

```powershell
python -m pip install -r requirements.txt
python src/main.py
```

## Execution order: what happens
1. **main.py** orchestrates the complete pipeline.
2. **preprocessing.py** reads the CSV, separates customer IDs, converts columns to numeric, replaces invalid/infinite values, imputes missing values with medians, and engineers three behavior ratios.
3. **eda.py** writes descriptive statistics, missing-value counts, correlations, a heatmap, and feature distribution plots.
4. **scale_data()** standardizes all features so variables with large numeric ranges do not dominate distance calculations.
5. **clustering.py** tests K-Means with k=2..8, selects the best tested silhouette score, then compares K-Means, Agglomerative Clustering, Gaussian Mixture, and DBSCAN. It saves customer assignments, model metrics, cluster profiles, and PCA plots.
6. **anomaly_detection.py** applies Isolation Forest and saves unusual-record flags, anomaly scores, and a PCA visualization.

## Outputs
CSV reports are saved in `outputs/`; plots are saved in `outputs/visualizations/`.

## Interpretation cautions
- Anomaly does not mean confirmed fraud; this dataset has no fraud labels.
- Cluster numbers are arbitrary. Interpret them using cluster profiles.
- Silhouette is an internal metric, not proof of business usefulness.
- DBSCAN's `eps` is a starting value and may need tuning.
- PCA is a 2D projection of higher-dimensional data.
- This project does not predict default, fraud, profitability, or customer lifetime value.
