
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUT, PLOTS

def run_eda(raw, features):
    OUTPUT.mkdir(exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    raw.describe().T.to_csv(OUTPUT / "descriptive_statistics.csv")
    raw.isna().sum().rename("missing_count").to_csv(OUTPUT / "missing_values_before_cleaning.csv")
    corr = features.corr()
    corr.to_csv(OUTPUT / "correlation_matrix.csv")
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(PLOTS / "correlation_heatmap.png", dpi=150)
    plt.close()
    cols = [c for c in ["BALANCE", "PURCHASES", "CREDIT_LIMIT", "PAYMENTS", "CASH_ADVANCE"] if c in features]
    if cols:
        features[cols].hist(figsize=(12, 8), bins=30)
        plt.tight_layout()
        plt.savefig(PLOTS / "feature_distributions.png", dpi=150)
        plt.close()
    print("EDA reports and plots saved.")
