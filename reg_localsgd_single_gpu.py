import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


# -----------------------------
# 1) Load Boston Housing
# -----------------------------
def load_boston_openml():
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="boston", version=1, as_frame=False)
    X = data.data.astype(np.float32)                   # (506, 13)
    y = data.target.astype(np.float32).reshape(-1, 1)  # (506, 1)
    return X, y


# -----------------------------
# 2) Standardization
# -----------------------------
def standardize_X(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std


def standardize_y(y: np.ndarray):
    mean = y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, keepdims=True) + 1e-8
    y_std = (y - mean) / std
    return y_std, mean, std


# -----------------------------
# 3) Dataset
# -----------------------------
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# 4) Model
# -----------------------------
class FC4(nn.Module):
    # 13 -> 32 -> 16 -> 1
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, k: int, seed: int):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    return [idx[i::k] for i in range(k)]


@torch.no_grad()
def fedavg(models):
    n = len(models)
    sd0 = models[0].state_dict()
    avg_sd = {}
    for k in sd0:
        avg_sd[k] = sum(m.state_dict()[k] for m in models) / n
    for m in models:
        m.load_state_dict(avg_sd)


@torch.no_grad()
def evaluate_mse(model, loader, device):
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        total += criterion(pred, yb).item()
        n += xb.size(0)
    return total / n


def train_one_epoch(model, opt, loader, device):
    model.train()
    criterion = nn.MSELoss()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / n


# -----------------------------
# Local SGD + FedAvg + CSV
# -----------------------------
def train_local_sgd_fedavg(
    make_model_fn,
    dataset,
    device,
    workers=8,
    momentum=0.0,
    base_lr=0.1,
    batch_size=16,
    epochs=1000,
    comm_every=5,
    seed=1234,
    csv_dir="results",
):
    print(f"\n=== workers={workers}, momentum={momentum}, base_lr={base_lr} ===")

    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/boston_W{workers}_mom{momentum}.csv"

    parts = split_indices(len(dataset), workers, seed)

    loaders = [
        DataLoader(Subset(dataset, p), batch_size=batch_size, shuffle=True)
        for p in parts
    ]
    eval_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    models, opts = [], []
    for _ in range(workers):
        m = make_model_fn().to(device)
        opt = torch.optim.SGD(
            m.parameters(),
            lr=base_lr,
            momentum=momentum,
            weight_decay=1e-4
        )
        models.append(m)
        opts.append(opt)

    fedavg(models)

    rec = []

    for epoch in range(1, epochs + 1):
        lr = base_lr / epoch
        for opt in opts:
            for pg in opt.param_groups:
                pg["lr"] = lr

        local_losses = [
            train_one_epoch(models[w], opts[w], loaders[w], device)
            for w in range(workers)
        ]
        local_loss_avg = sum(local_losses) / workers

        if epoch % comm_every == 0:
            fedavg(models)
            global_mse = evaluate_mse(models[0], eval_loader, device)

            rec.append([epoch, lr, local_loss_avg, global_mse])

            print(
                f"Epoch {epoch:4d} | lr={lr:.5f} | "
                f"local_loss_avg={local_loss_avg:.6f} | "
                f"global_mse={global_mse:.6f}"
            )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "lr",
            "local_loss_avg",
            "global_mse_after_fedavg"
        ])
        writer.writerows(rec)

    print(f"Saved CSV to {csv_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(1234)

    X, y = load_boston_openml()
    X = standardize_X(X)
    y, y_mean, y_std = standardize_y(y)

    ds = NumpyDataset(X, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for mom in [0.0, 0.5, 0.9]:
        train_local_sgd_fedavg(
            make_model_fn=FC4,
            dataset=ds,
            device=device,
            workers=1,
            momentum=mom,
            base_lr=0.1,
            epochs=2000,
            comm_every=10,
        )


if __name__ == "__main__":
    main()
