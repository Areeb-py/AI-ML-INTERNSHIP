# Telecom Customer Churn Prediction — Week 05

An end-to-end binary-classification project using the IBM Telco Customer Churn dataset. It trains a multilayer perceptron (MLP) twice:
1. **NumPy from scratch** — manual forward propagation, backpropagation, and Adam.
2. **PyTorch** — `nn.Module`, autograd, `BCEWithLogitsLoss`, and Adam.

## Dataset

Download the CSV from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Or run the downloader to fetch the IBM public CSV mirror:

```powershell
python src/download_data.py
```

Expected CSV path:
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Setup (PowerShell)

Run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run order

From the project root:

```powershell
python src/download_data.py
python src/prepare_data.py
python src/train_numpy.py
python src/train_pytorch.py
python src/evaluate.py
```

If you downloaded the Kaggle CSV manually, you can skip the first command.

## Outputs

- `models/preprocessor.joblib` — fitted preprocessing pipeline.
- `models/processed_data.npz` — train/test arrays used by both networks.
- `models/numpy_mlp.npz` — NumPy network weights.
- `models/pytorch_mlp.pth` — PyTorch model weights.
- `outputs/data_summary.json` — dataset and split summary.
- `outputs/metrics.json` — accuracy, precision, recall, F1, and ROC-AUC.
- `outputs/predictions_comparison.csv` — actual labels and both models' predictions.
- `outputs/confusion_matrix_numpy.png` and `outputs/confusion_matrix_pytorch.png`.

## Methodology

- Target: `Churn` (`Yes` = 1, `No` = 0).
- `customerID` is removed because it is an identifier.
- `TotalCharges` is converted to numeric; blank values are imputed.
- Data is split into training and test sets using stratification.
- Imputation, one-hot encoding, and standardization are fitted on training data only to avoid test-data leakage.
- Both implementations use the same architecture: input → 32 ReLU units → 16 ReLU units → 1 output logit.
- The NumPy version manually implements gradients and Adam. PyTorch calculates gradients through autograd.

## Evaluation

Churn is the positive class. Since the classes are imbalanced, do not rely on accuracy alone. Review recall (how many churners are caught), precision (how many flagged customers actually churn), F1, ROC-AUC, and the confusion matrices.

This is an educational project, not a production retention system. Validate on current, representative data and review fairness, privacy, and business impact before any real-world deployment.

## Suggested extensions

1. Change the classification threshold and examine the precision/recall trade-off.
2. Compare Adam with SGD.
3. Experiment with hidden-layer sizes.
4. Train a Logistic Regression baseline.
5. Explain which metric should matter most if the company wants to identify as many potential churners as possible.
