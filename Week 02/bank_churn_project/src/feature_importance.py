"""Show model feature importance and save a chart/table."""

import pandas as pd
import matplotlib.pyplot as plt

from common import load_data, split_data, build_preprocessor, MODEL_DIR, RESULTS_DIR, ensure_output_dirs
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def main():
    ensure_output_dirs()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # This standalone script uses the same fixed XGBoost settings as model comparison.
    model = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)

    names = model.named_steps["preprocessor"].get_feature_names_out()
    scores = model.named_steps["classifier"].feature_importances_
    importance = pd.DataFrame({"Feature": names, "Importance": scores})
    importance = importance.sort_values("Importance", ascending=False)

    print("\nTop 15 features (XGBoost importance):")
    print(importance.head(15).to_string(index=False))

    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    top = importance.head(10).sort_values("Importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top["Feature"], top["Importance"])
    plt.xlabel("Importance score")
    plt.ylabel("Feature")
    plt.title("Top 10 XGBoost Features for Customer Churn")
    plt.tight_layout()
    chart_path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(chart_path, dpi=150)
    print(f"\nSaved feature table to: {RESULTS_DIR / 'feature_importance.csv'}")
    print(f"Saved chart to: {chart_path}")
    plt.show()


if __name__ == "__main__":
    main()
