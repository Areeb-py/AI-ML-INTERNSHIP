# Bank Customer Churn Prediction

A learning project demonstrating Week 3 supervised-learning concepts using a bank customer churn dataset.

## What it demonstrates

- **Feature engineering:** one-hot encoding for categorical columns and standardization for numerical columns.
- **Missing-value handling:** median imputation for numerical data and most-frequent imputation for categorical data.
- **Pipelines:** preprocessing is fitted inside each training fold, reducing data leakage.
- **Tree-based models:** Random Forest, XGBoost, and LightGBM.
- **Model comparison:** compare accuracy, churn precision/recall/F1, and ROC-AUC.
- **Feature importance:** inspect XGBoost feature importance.
- **Feature selection:** use `SelectFromModel` with an ExtraTrees estimator inside a pipeline.
- **Hyperparameter tuning:** `GridSearchCV` with stratified cross-validation.
- **Model persistence and prediction:** save the fitted preprocessing/model pipeline and use it for new records.

## Project structure

```text
bank_churn_project/
├── data/
│   └── Churn_Modelling.csv       # Add this dataset yourself
├── models/
│   └── best_churn_model.joblib   # Created by train.py
├── results/                      # Created outputs
├── src/
│   ├── common.py
│   ├── train.py
│   ├── compare_models.py
│   ├── feature_importance.py
│   ├── feature_selection.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## 1. Setup (Windows PowerShell)

Open PowerShell in the project root folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the project's Python directly without activating:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Add the dataset

Place `Churn_Modelling.csv` in `data/Churn_Modelling.csv`.

The expected target column is `Exited`, with values 0 and 1. The code expects `Geography` and `Gender` columns.

## 3. Recommended run order

Run each command from the project root:

```powershell
python src/compare_models.py
python src/feature_importance.py
python src/feature_selection.py
python src/train.py
python src/predict.py
```

`train.py` is the main training workflow. It searches across parameter grids for Random Forest, XGBoost, and LightGBM, selects the model with the best mean cross-validation ROC-AUC, evaluates it once on the held-out test set, and saves the fitted pipeline.

The initial comparison and feature scripts use fixed model configurations to make each concept easier to study. They are optional and do not need to be run before `train.py`.

## Outputs

After running the scripts, the `results/` folder can contain:

- `evaluation.json` — selected model, best parameters, and test metrics.
- `classification_report.txt` — precision, recall, F1-score and support.
- `confusion_matrix.csv` — actual versus predicted class counts.
- `grid_search_results.csv` — cross-validation results for every parameter combination.
- `feature_importance.csv` and `feature_importance.png` — XGBoost feature importance.
- `selected_features.csv` — feature names retained by the feature-selection example.

The trained pipeline is saved to `models/best_churn_model.joblib`.

## Important notes

- Use the held-out test set only for final evaluation, not for selecting parameters.
- Accuracy alone can be misleading because fewer customers leave than stay. Review recall and precision for class `1` (`Left`) as well.
- Feature importance describes how the fitted model uses features; it does not prove causation.
- `predict.py` contains an example customer. Change the example values to predict for a different customer.
- This is an educational model, not a production banking decision system. Review fairness, calibration, privacy, and operational requirements before real-world use.
