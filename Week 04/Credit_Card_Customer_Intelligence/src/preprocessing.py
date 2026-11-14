
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from config import DATA, OUTPUT

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Put 'CC GENERAL.csv' inside: {DATA.parent}")
    df = pd.read_csv(DATA)
    print("Loaded:", df.shape)
    return df

def clean_data(df):
    data = df.copy()
    ids = data.pop("CUST_ID") if "CUST_ID" in data.columns else pd.Series(data.index.astype(str), name="CUSTOMER_ID")
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    if {"BALANCE", "CREDIT_LIMIT"} <= set(data.columns):
        data["CREDIT_UTILIZATION"] = data["BALANCE"] / data["CREDIT_LIMIT"].replace(0, np.nan)
    if {"CASH_ADVANCE", "PURCHASES"} <= set(data.columns):
        data["CASH_ADVANCE_TO_PURCHASES"] = data["CASH_ADVANCE"] / (data["PURCHASES"] + 1)
    if {"PAYMENTS", "MINIMUM_PAYMENTS"} <= set(data.columns):
        data["PAYMENT_TO_MINIMUM_RATIO"] = data["PAYMENTS"] / (data["MINIMUM_PAYMENTS"] + 1)
    data = data.dropna(axis=1, how="all")
    data = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(data), columns=data.columns)
    OUTPUT.mkdir(exist_ok=True)
    saved = data.copy()
    saved.insert(0, "CUSTOMER_ID", ids.reset_index(drop=True))
    saved.to_csv(OUTPUT / "cleaned_customers.csv", index=False)
    print("Cleaned feature matrix:", data.shape)
    return ids.reset_index(drop=True), data

def scale_data(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler
