"""Shared utilities for the bank customer churn project."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "Churn_Modelling.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
TARGET = "Exited"

# These columns are identifiers or names, so they are not useful model inputs.
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
CATEGORICAL_FEATURES = ["Geography", "Gender"]


def load_data():
    """Load the expected CSV and validate its required columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Place Churn_Modelling.csv inside the project's data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop only identifier columns that actually exist in the CSV.
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # The target must be present and contain both binary classes.
    if df[TARGET].isna().any():
        raise ValueError(f"Target column '{TARGET}' contains missing values.")
    observed_targets = set(df[TARGET].unique())
    if not observed_targets.issubset({0, 1}) or len(observed_targets) != 2:
        raise ValueError(
            f"Expected binary target values 0 and 1 in '{TARGET}', "
            f"but found {sorted(observed_targets)}."
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Validate the categorical columns used by this project.
    missing_cats = [c for c in CATEGORICAL_FEATURES if c not in X.columns]
    if missing_cats:
        raise ValueError(
            f"Expected categorical columns are missing: {missing_cats}. "
            "Update CATEGORICAL_FEATURES in src/common.py if your dataset differs."
        )

    return X, y


def split_data(X, y):
    """Create a stratified holdout test set; keep it out of model selection."""
    return train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )


def build_preprocessor(X):
    """Build preprocessing that is fitted only on each training fold."""
    categorical = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    numerical = [c for c in X.columns if c not in categorical]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical),
    ])


def ensure_output_dirs():
    """Create output folders when needed."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
