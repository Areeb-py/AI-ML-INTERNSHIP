"""Demonstrate model-based feature selection without leaking test data."""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from common import load_data, split_data, build_preprocessor, RESULTS_DIR, ensure_output_dirs


def main():
    ensure_output_dirs()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Feature selection is fitted on training data only, inside the pipeline.
    # SelectFromModel keeps features whose importance is at least the median.
    selector = SelectFromModel(
        estimator=ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        threshold="median",
    )

    model = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("feature_selection", selector),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42,
            class_weight="balanced", n_jobs=-1
        )),
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    # Reconstruct selected names after preprocessing and selection.
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    selected_mask = model.named_steps["feature_selection"].get_support()
    selected_names = feature_names[selected_mask]

    print("Selected features:", len(selected_names), "of", len(feature_names))
    print("\nSelected feature names:")
    for name in selected_names:
        print("-", name)

    print("\nCV ROC-AUC scores:", [round(float(s), 4) for s in cv_scores])
    print("Mean CV ROC-AUC:", round(float(cv_scores.mean()), 4))
    print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Test ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

    pd.DataFrame({"Selected feature": selected_names}).to_csv(
        RESULTS_DIR / "selected_features.csv", index=False
    )
    print(f"\nSaved selected features to: {RESULTS_DIR / 'selected_features.csv'}")


if __name__ == "__main__":
    main()
