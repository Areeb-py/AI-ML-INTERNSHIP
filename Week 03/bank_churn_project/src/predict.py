"""Load the saved model and predict churn for one new customer."""

import joblib
import pandas as pd

from common import MODEL_DIR


def main():
    model_path = MODEL_DIR / "best_churn_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}.\n"
            "Run `python src/train.py` first."
        )

    model = joblib.load(model_path)

    # Example customer. Replace values with a real customer's values.
    # Required input columns: CreditScore, Geography, Gender, Age, Tenure,
    # Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
    customer = pd.DataFrame([{
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 49,
        "Tenure": 1,
        "Balance": 6000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101110.0,
    }])

    # Pipeline applies the same preprocessing learned during training.
    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0, 1])

    print("Predicted class:", prediction, "(1=likely to leave, 0=likely to stay)")
    print(f"Estimated churn probability: {probability:.1%}")
    print(
        "Interpretation:",
        "The model predicts churn." if prediction == 1
        else "The model predicts the customer will stay."
    )
    print("\nNote: this is a model estimate, not a certainty.")


if __name__ == "__main__":
    main()
