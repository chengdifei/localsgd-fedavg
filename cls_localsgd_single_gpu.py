import os
import csv
import time
import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# -----------------------------
#  ResNet20 for CIFAR
# -----------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(torch.flatten(x, 1))


def resnet20(num_classes):
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes)


# -----------------------------
#  Utils
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def average_models(models):
    n = len(models)
    ps = [dict(m.named_parameters()) for m in models]
    for k in ps[0]:
        avg = sum(p[k].data for p in ps) / n
        for p in ps:
            p[k].data.copy_(avg)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on a loader: return (avg_loss, accuracy)."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += ce(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, opt, loader, device):
    """
    Train one epoch on 'loader'.
    Return average training loss on THIS worker's local data for this epoch.
    """
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_sum, total = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        total += x.size(0)

    return loss_sum / total


# -----------------------------
#  Config
# -----------------------------
@dataclass
class CFG:
    dataset: str = "cifar100"
    data_root: str = "./data"

    #workers_list: tuple = (1, 3, 10)
    workers_list: tuple = (8,)
    #momentums: tuple = (0.0, 0.5, 0.9)
    momentums: tuple = (0.9,)

    epochs: int = 1000
    comm_every: int = 5

    batch_size: int = 128
    lr: float = 0.2              # base lr = 0.1
    weight_decay: float = 1e-4
    seed: int = 1234
    num_workers_dataloader: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    eval_batch_size: int = 512
    csv_dir: str = "results"

# -----------------------------
#  Dataset
# -----------------------------
def get_dataset(cfg):
    tf = T.Compose([T.ToTensor()])
    if cfg.dataset == "cifar100":
        ds = torchvision.datasets.CIFAR100(cfg.data_root, True, download=True, transform=tf)
        return ds, 100
    else:
        ds = torchvision.datasets.CIFAR10(cfg.data_root, True, download=True, transform=tf)
        return ds, 10


def split_indices(n, k, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


# -----------------------------
#  Main
# -----------------------------
def main():
    cfg = CFG()
    print(cfg)

    set_seed(cfg.seed)
    trainset, num_classes = get_dataset(cfg)
    device = torch.device(cfg.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for workers in cfg.workers_list:
        for mom in cfg.momentums:
            print(f"\n=== workers={workers}, momentum={mom} ===")

            # per-worker local dataloaders (training: shuffle=True)
            parts = split_indices(len(trainset), workers, cfg.seed)
            local_loaders = [
                DataLoader(
                    Subset(trainset, p),
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=cfg.num_workers_dataloader,
                )
                for p in parts
            ]

            # per-worker local eval dataloaders (eval: shuffle=False)
            local_eval_loaders = [
                DataLoader(
                    Subset(trainset, p),
                    batch_size=cfg.eval_batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers_dataloader,
                )
                for p in parts
            ]

            # global eval loader (full train set)
            eval_loader = DataLoader(trainset, batch_size=cfg.eval_batch_size, shuffle=False)

            # build models/opts
            models, opts = [], []
            for _ in range(workers):
                m = resnet20(num_classes).to(device)
                opt = optim.SGD(
                    m.parameters(),
                    lr=cfg.lr,  # will be overwritten each epoch: cfg.lr / epoch
                    momentum=mom,
                    weight_decay=cfg.weight_decay,
                )
                models.append(m)
                opts.append(opt)

            # initial sync
            average_models(models)

            # records (only at comm points, i.e., epoch % comm_every == 0)
            rec_epoch: List[int] = []
            rec_lr: List[float] = []
            rec_local_loss_avg: List[float] = []

            rec_local_train_loss_end_avg: List[float] = []
            rec_local_train_acc_end_avg: List[float] = []

            rec_train_loss_after_avg: List[float] = []
            rec_train_acc_after_avg: List[float] = []

            for epoch in range(1, cfg.epochs + 1):
                # --- NEW: lr = 0.1 / epoch (inverse decay) ---
                lr_epoch = cfg.lr / epoch
                for opt in opts:
                    for pg in opt.param_groups:
                        pg["lr"] = lr_epoch

                # --- local training + collect each worker local train loss (during training) ---
                local_losses = []
                for w in range(workers):
                    l = train_one_epoch(models[w], opts[w], local_loaders[w], device)
                    local_losses.append(l)
                local_loss_avg_workers = sum(local_losses) / len(local_losses)

                # --- communication/eval record (only at comm points) ---
                if epoch % cfg.comm_every == 0:
                    # pre-avg local eval (each worker on its local data)
                    local_end_losses, local_end_accs = [], []
                    for w in range(workers):
                        lw, aw = evaluate(models[w], local_eval_loaders[w], device)
                        local_end_losses.append(lw)
                        local_end_accs.append(aw)
                    local_train_loss_end_avg = sum(local_end_losses) / workers
                    local_train_acc_end_avg = sum(local_end_accs) / workers

                    # FedAvg / model averaging
                    average_models(models)

                    # global model (after avg) evaluated on full trainset
                    train_loss0, train_acc0 = evaluate(models[0], eval_loader, device)

                    lr_now = opts[0].param_groups[0]["lr"]

                    rec_epoch.append(epoch)
                    rec_lr.append(lr_now)
                    rec_local_loss_avg.append(local_loss_avg_workers)

                    rec_local_train_loss_end_avg.append(local_train_loss_end_avg)
                    rec_local_train_acc_end_avg.append(local_train_acc_end_avg)

                    rec_train_loss_after_avg.append(train_loss0)
                    rec_train_acc_after_avg.append(train_acc0)

                    print(
                        f"Epoch {epoch} | lr {lr_now:.6f} | "
                        f"local_loss(avg_workers) {local_loss_avg_workers:.4f} | "
                        f"local_train_loss_end(avg) {local_train_loss_end_avg:.4f} | "
                        f"local_train_acc_end(avg) {local_train_acc_end_avg*100:.2f}% | "
                        f"train_loss(after_avg) {train_loss0:.4f} | "
                        f"train_acc(after_avg) {train_acc0*100:.2f}%"
                    )

            # --- save CSV for this (workers, momentum) ---
            os.makedirs(cfg.csv_dir, exist_ok=True)
            csv_path = f"{cfg.csv_dir}/{cfg.dataset}_W{workers}_mom{mom}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "lr",
                    "local_loss_avg_workers",
                    "local_train_loss_end_avg",
                    "local_train_acc_end_avg",
                    "train_loss_after_avg",
                    "train_acc_after_avg",
                ])
                for e, lr, ll, lte, lae, tl, ta in zip(
                    rec_epoch,
                    rec_lr,
                    rec_local_loss_avg,
                    rec_local_train_loss_end_avg,
                    rec_local_train_acc_end_avg,
                    rec_train_loss_after_avg,
                    rec_train_acc_after_avg,
                ):
                    writer.writerow([e, lr, ll, lte, lae, tl, ta])

            print(f"Saved CSV to {csv_path}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"\n[TOTAL] elapsed time: {time.perf_counter() - t0:.2f} sec")


if __name__ == "__main__":
    main()
