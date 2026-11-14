"""Evaluate both saved models on the same held-out test set."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from train_pytorch import ChurnMLP
from train_numpy import relu, sigmoid

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "models" / "processed_data.npz"
NUMPY_FILE = ROOT / "models" / "numpy_mlp.npz"
TORCH_FILE = ROOT / "models" / "pytorch_mlp.pth"
OUTPUT_DIR = ROOT / "outputs"

def numpy_predict(X, p):
    a1 = relu(X @ p["W1"] + p["b1"])
    a2 = relu(a1 @ p["W2"] + p["b2"])
    return sigmoid(a2 @ p["W3"] + p["b3"]).ravel()

def metrics(y, probabilities):
    predicted = (probabilities >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1_score": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }
    return result, predicted

def save_cm(y, predicted, filename, title):
    cm = confusion_matrix(y, predicted)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Stay", "Churn"])
    ax.set_yticks([0, 1], ["Stay", "Churn"])
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)

def main():
    required = [DATA_FILE, NUMPY_FILE, TORCH_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files; run preparation and both training scripts first:\n" + "\n".join(missing))

    OUTPUT_DIR.mkdir(exist_ok=True)
    data = np.load(DATA_FILE)
    X_test = data["X_test"]
    y_test = data["y_test"].astype(int)

    stored = np.load(NUMPY_FILE)
    numpy_params = {key: stored[key] for key in stored.files}
    numpy_probs = numpy_predict(X_test, numpy_params)
    numpy_metrics, numpy_pred = metrics(y_test, numpy_probs)

    checkpoint = torch.load(TORCH_FILE, map_location="cpu", weights_only=True)
    model = ChurnMLP(checkpoint["input_features"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        torch_probs = torch.sigmoid(logits).numpy().ravel()
    torch_metrics, torch_pred = metrics(y_test, torch_probs)

    results = {"numpy_mlp": numpy_metrics, "pytorch_mlp": torch_metrics}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({
        "actual_churn": y_test,
        "numpy_probability": numpy_probs,
        "numpy_prediction": numpy_pred,
        "pytorch_probability": torch_probs,
        "pytorch_prediction": torch_pred,
    }).to_csv(OUTPUT_DIR / "predictions_comparison.csv", index=False)

    save_cm(y_test, numpy_pred, "confusion_matrix_numpy.png", "NumPy MLP")
    save_cm(y_test, torch_pred, "confusion_matrix_pytorch.png", "PyTorch MLP")

    print("\nTEST RESULTS (positive class = churn)")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("\nNumPy classification report:\n", classification_report(y_test, numpy_pred, zero_division=0))
    print("\nPyTorch classification report:\n", classification_report(y_test, torch_pred, zero_division=0))
    print(f"\nReports saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
