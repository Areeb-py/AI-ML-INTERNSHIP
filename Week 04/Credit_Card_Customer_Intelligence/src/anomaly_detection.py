
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from config import OUTPUT, PLOTS, RANDOM_STATE, CONTAMINATION

def detect_anomalies(X, ids):
    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE, n_estimators=200)
    pred = model.fit_predict(X)
    scores = model.decision_function(X)
    result = pd.DataFrame({"CUSTOMER_ID": ids, "ANOMALY": pred == -1, "ANOMALY_SCORE": scores})
    result.to_csv(OUTPUT / "anomaly_results.csv", index=False)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    plt.figure(figsize=(9, 6))
    plt.scatter(coords[pred == 1, 0], coords[pred == 1, 1], s=8, alpha=.4, label="Typical")
    plt.scatter(coords[pred == -1, 0], coords[pred == -1, 1], s=18, label="Potential anomaly")
    plt.legend(); plt.title("Isolation Forest - PCA Projection")
    plt.tight_layout(); plt.savefig(PLOTS / "anomalies_pca.png", dpi=150); plt.close()
    print("Potential anomalies:", int((pred == -1).sum()))
