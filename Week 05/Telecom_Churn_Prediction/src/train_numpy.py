"""MLP from scratch in NumPy: forward propagation, backpropagation, and Adam."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "models" / "processed_data.npz"
MODEL_FILE = ROOT / "models" / "numpy_mlp.npz"

def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(0.0, z)

def bce_logits(logits, y):
    return np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits))))

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Run `python src/prepare_data.py` first.")
    data = np.load(DATA_FILE)
    X = data["X_train"]
    y = data["y_train"].reshape(-1, 1)

    rng = np.random.default_rng(42)
    n_samples, n_features = X.shape
    h1, h2 = 32, 16
    params = {
        "W1": rng.normal(0, np.sqrt(2 / n_features), (n_features, h1)),
        "b1": np.zeros((1, h1)),
        "W2": rng.normal(0, np.sqrt(2 / h1), (h1, h2)),
        "b2": np.zeros((1, h2)),
        "W3": rng.normal(0, np.sqrt(1 / h2), (h2, 1)),
        "b3": np.zeros((1, 1)),
    }
    first_moment = {k: np.zeros_like(v) for k, v in params.items()}
    second_moment = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr, epochs, batch_size = 0.001, 350, 128
    step = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_samples)
        total_loss = 0.0
        for start in range(0, n_samples, batch_size):
            ids = order[start:start + batch_size]
            xb, yb = X[ids], y[ids]
            batch_n = len(xb)

            # Forward pass.
            z1 = xb @ params["W1"] + params["b1"]
            a1 = relu(z1)
            z2 = a1 @ params["W2"] + params["b2"]
            a2 = relu(z2)
            logits = a2 @ params["W3"] + params["b3"]
            probs = sigmoid(logits)
            total_loss += bce_logits(logits, yb) * batch_n

            # Backward pass: sigmoid + binary cross entropy simplifies to (p - y).
            dz3 = (probs - yb) / batch_n
            dW3 = a2.T @ dz3
            db3 = dz3.sum(axis=0, keepdims=True)
            da2 = dz3 @ params["W3"].T
            dz2 = da2 * (z2 > 0)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0, keepdims=True)
            da1 = dz2 @ params["W2"].T
            dz1 = da1 * (z1 > 0)
            dW1 = xb.T @ dz1
            db1 = dz1.sum(axis=0, keepdims=True)
            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

            # Adam update.
            step += 1
            for key, grad in grads.items():
                first_moment[key] = beta1 * first_moment[key] + (1 - beta1) * grad
                second_moment[key] = beta2 * second_moment[key] + (1 - beta2) * (grad ** 2)
                m_hat = first_moment[key] / (1 - beta1 ** step)
                v_hat = second_moment[key] / (1 - beta2 ** step)
                params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | loss: {total_loss / n_samples:.4f}")

    np.savez_compressed(MODEL_FILE, **params)
    print(f"Saved NumPy model: {MODEL_FILE}")

if __name__ == "__main__":
    main()
