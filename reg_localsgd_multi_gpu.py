import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
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
def evaluate_mse(model, loader, device):
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(xb)
        total += criterion(pred, yb).item()
        n += xb.size(0)
    return total / n


def train_one_epoch(model, opt, loader, device):
    model.train()
    criterion = nn.MSELoss()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        loss = criterion(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / n


@torch.no_grad()
def fedavg_sync_model_(model: nn.Module, world_size: int):
    """
    FedAvg across ranks:
      param = (sum over ranks param) / world_size
    """
    for p in model.parameters():
        dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
        p.data.div_(world_size)


# -----------------------------
# Distributed worker
# -----------------------------
def worker_main(
    rank: int,
    world_size: int,
    momentum: float,
    base_lr: float,
    batch_size: int,
    epochs: int,
    comm_every: int,
    seed: int,
    csv_dir: str,
):
    # ---- init dist ----
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # ---- device bind ----
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # ---- seeds ----
    # 注意：split_indices 使用同一个 seed 保证划分一致；
    # 这里用 seed+rank 避免每个 rank 的 DataLoader shuffle 序列完全相同（不影响划分）
    set_seed(seed + rank)

    # ---- load data (each rank loads; openml cache usually makes it fine) ----
    X, y = load_boston_openml()
    X = standardize_X(X)
    y, _, _ = standardize_y(y)
    ds = NumpyDataset(X, y)

    # ---- partition ----
    parts = split_indices(len(ds), world_size, seed)
    local_idx = parts[rank]

    local_loader = DataLoader(
        Subset(ds, local_idx),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    # ---- model/opt ----
    model = FC4().to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=1e-4,
    )

    # ---- initial sync (match your original: fedavg(models) right after init) ----
    fedavg_sync_model_(model, world_size)

    # ---- records (rank0 only) ----
    rec = []
    if rank == 0:
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = f"{csv_dir}/boston_W{world_size}_mom{momentum}.csv"
        print(f"\n=== multi-gpu: workers={world_size}, momentum={momentum}, base_lr={base_lr} ===")
        print("CSV:", csv_path)

    for epoch in range(1, epochs + 1):
        lr = base_lr / epoch
        for pg in opt.param_groups:
            pg["lr"] = lr

        # local train
        local_loss = train_one_epoch(model, opt, local_loader, device)

        # average local_loss across workers (for logging consistency)
        t = torch.tensor([local_loss], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        local_loss_avg = float((t / world_size).item())

        if epoch % comm_every == 0:
            # FedAvg
            fedavg_sync_model_(model, world_size)

            # global eval only on rank0 (same as your original: evaluate on full dataset after fedavg)
            if rank == 0:
                global_mse = evaluate_mse(model, eval_loader, device)
                rec.append([epoch, lr, local_loss_avg, global_mse])

                print(
                    f"Epoch {epoch:4d} | lr={lr:.5f} | "
                    f"local_loss_avg={local_loss_avg:.6f} | "
                    f"global_mse={global_mse:.6f}"
                )

    # save csv only rank0
    if rank == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "local_loss_avg", "global_mse_after_fedavg"])
            writer.writerows(rec)
        print(f"Saved CSV to {csv_path}")

    dist.destroy_process_group()


# -----------------------------
# Launcher
# -----------------------------
def run_multigpu(
    world_size: int,
    momentums=(0.0, 0.5, 0.9),
    base_lr=0.1,
    batch_size=16,
    epochs=2000,
    comm_every=10,
    seed=1234,
    csv_dir="results",
):
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 多卡环境才能运行该版本。")
    if world_size > torch.cuda.device_count():
        raise RuntimeError(f"world_size={world_size} 但机器只有 {torch.cuda.device_count()} 张 GPU。")

    for mom in momentums:
        mp.spawn(
            worker_main,
            args=(world_size, mom, base_lr, batch_size, epochs, comm_every, seed, csv_dir),
            nprocs=world_size,
            join=True,
        )


def main():
    # 这里设置你要用几张卡（一个 worker 一张卡）
    world_size = 1

    run_multigpu(
        world_size=world_size,
        momentums=(0.0, 0.5, 0.9),
        base_lr=0.1,
        batch_size=16,
        epochs=2000,
        comm_every=10,
        seed=1234,
        csv_dir="results",
    )


if __name__ == "__main__":
    main()
