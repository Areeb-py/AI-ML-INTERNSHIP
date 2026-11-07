
from preprocessing import load_data, clean_data, scale_data
from eda import run_eda
from clustering import run_clustering
from anomaly_detection import detect_anomalies

def main():
    print("\n[1/6] Loading data")
    raw = load_data()
    print("\n[2/6] Cleaning and feature engineering")
    ids, features = clean_data(raw)
    print("\n[3/6] Exploratory data analysis")
    run_eda(raw, features)
    print("\n[4/6] Scaling features")
    X, scaler = scale_data(features)
    print("\n[5/6] Clustering customers")
    run_clustering(X, ids, features.columns)
    print("\n[6/6] Detecting anomalies")
    detect_anomalies(X, ids)
    print("\nFinished. See the outputs folder.")

if __name__ == "__main__":
    main()
