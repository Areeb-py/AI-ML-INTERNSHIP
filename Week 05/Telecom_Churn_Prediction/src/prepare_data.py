"""Clean, split, and preprocess data. Fit transformations on training data only."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run `python src/download_data.py` or place the Kaggle CSV there."
        )

    df = pd.read_csv(DATA_PATH)
    print("Original dataset shape:", df.shape)
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn"])

    y = df.pop("Churn").astype("float32").to_numpy()
    X = df
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train = preprocessor.fit_transform(X_train).astype("float32")
    X_test = preprocessor.transform(X_test).astype("float32")

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    np.savez_compressed(
        MODEL_DIR / "processed_data.npz",
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )
    summary = {
        "rows": int(len(df)),
        "features_after_encoding": int(X_train.shape[1]),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "positive_class": "Churn = Yes (1)",
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
    }
    (OUTPUT_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
