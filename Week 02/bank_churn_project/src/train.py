"""Train and tune candidate models, evaluate the winner, and save it."""

import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from common import (
    load_data, split_data, build_preprocessor,
    MODEL_DIR, RESULTS_DIR, ensure_output_dirs
)


def main():
    ensure_output_dirs()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor(X_train)

    # All candidates share the same preprocessing and train/test split.
    # GridSearchCV selects parameters using training-only cross-validation.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    param_grid = [
        {
            "classifier": [RandomForestClassifier(
                random_state=42, class_weight="balanced"
            )],
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 8, None],
            "classifier__min_samples_split": [2, 10],
        },
        {
            "classifier": [XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            )],
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        },
        {
            "classifier": [LGBMClassifier(
                random_state=42, verbosity=-1, n_jobs=1
            )],
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, -1],
            "classifier__learning_rate": [0.05, 0.1],
        },
    ]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise",
    )

    print(f"Dataset rows: {len(X)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("Tuning Random Forest, XGBoost, and LightGBM...")
    search.fit(X_train, y_train)

    # The holdout test set is used only after cross-validation selects a model.
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "selected_model": type(best_model.named_steps["classifier"]).__name__,
        "best_cv_roc_auc": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision_churn": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall_churn": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1_churn": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "best_params": {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                        for k, v in search.best_params_.items()},
    }

    print("\n--- Best cross-validation result ---")
    print("Selected model:", metrics["selected_model"])
    print("Best CV ROC-AUC:", round(metrics["best_cv_roc_auc"], 4))
    print("Best parameters:", search.best_params_)

    print("\n--- Holdout test results ---")
    for key in [
        "test_accuracy", "test_precision_churn", "test_recall_churn",
        "test_f1_churn", "test_roc_auc"
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    print("\nClassification report (0=stayed, 1=left):")
    report = classification_report(y_test, y_pred, target_names=["Stayed", "Left"], zero_division=0)
    print(report)
    print("Confusion matrix (rows=actual, columns=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save the entire fitted pipeline so predictions include preprocessing.
    model_path = MODEL_DIR / "best_churn_model.joblib"
    joblib.dump(best_model, model_path)

    metrics_path = RESULTS_DIR / "evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=["actual_stayed", "actual_left"],
                 columns=["predicted_stayed", "predicted_left"]).to_csv(
        RESULTS_DIR / "confusion_matrix.csv"
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    # Save the CV results so the tested candidates can be inspected later.
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(RESULTS_DIR / "grid_search_results.csv", index=False)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved evaluation files to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
