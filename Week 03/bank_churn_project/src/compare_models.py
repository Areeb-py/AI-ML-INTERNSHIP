"""Compare fixed baseline configurations for Random Forest, XGBoost, LightGBM."""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from common import load_data, split_data, build_preprocessor


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=-1, n_jobs=-1
        ),
    }

    rows = []
    for name, classifier in models.items():
        # A fresh preprocessor is created for each model to avoid reusing fitted state.
        model = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", classifier),
        ])
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (churn)": precision_score(y_test, pred, zero_division=0),
            "Recall (churn)": recall_score(y_test, pred, zero_division=0),
            "F1 (churn)": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
        })

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    print("\n--- Model comparison (sorted by ROC-AUC) ---")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
