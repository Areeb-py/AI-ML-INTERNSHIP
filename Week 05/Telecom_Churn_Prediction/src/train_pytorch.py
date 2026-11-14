"""Same MLP architecture trained with PyTorch autograd and Adam."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "models" / "processed_data.npz"
MODEL_FILE = ROOT / "models" / "pytorch_mlp.pth"

class ChurnMLP(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Raw logit; no sigmoid in the model.
        )

    def forward(self, x):
        return self.network(x)

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Run `python src/prepare_data.py` first.")
    torch.manual_seed(42)
    data = np.load(DATA_FILE)
    X = torch.tensor(data["X_train"], dtype=torch.float32)
    y = torch.tensor(data["y_train"].reshape(-1, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

    model = ChurnMLP(X.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | loss: {total_loss / len(X):.4f}")

    torch.save(
        {"model_state_dict": model.state_dict(), "input_features": X.shape[1]},
        MODEL_FILE,
    )
    print(f"Saved PyTorch model: {MODEL_FILE}")

if __name__ == "__main__":
    main()
