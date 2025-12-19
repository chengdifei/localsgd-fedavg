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
import torch.distributed as dist
import torch.multiprocessing as mp
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
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, k: int, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on a loader: return (avg_loss, accuracy)."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += ce(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, opt, loader, device):
    """Train one epoch on local loader, return avg loss."""
    model.train()
    ce = nn.CrossEntropyLoss()
    loss_sum, total = 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        total += x.size(0)

    return loss_sum / total


@torch.no_grad()
def fedavg_sync_model_(model: nn.Module, world_size: int):
    """
    FedAvg: all_reduce(sum) then divide by world_size, in-place on parameters.
    注意：只同步 parameters（不动 BN buffers），与你原先 average_models 对齐。
    """
    for p in model.parameters():
        dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
        p.data.div_(world_size)


# -----------------------------
#  Config
# -----------------------------
@dataclass
class CFG:
    dataset: str = "cifar100"
    data_root: str = "./data"

    # 多卡时：workers_list 里建议只放一个值 == GPU 数量
    workers_list: tuple = (1,)
    momentums: tuple = (0.9,)

    epochs: int = 1000
    comm_every: int = 5

    batch_size: int = 128
    lr: float = 0.2
    weight_decay: float = 1e-4
    seed: int = 1234

    num_workers_dataloader: int = 4
    eval_batch_size: int = 512
    csv_dir: str = "results"


def get_dataset(cfg: CFG):
    tf = T.Compose([T.ToTensor()])
    if cfg.dataset == "cifar100":
        ds = torchvision.datasets.CIFAR100(cfg.data_root, train=True, download=True, transform=tf)
        return ds, 100
    else:
        ds = torchvision.datasets.CIFAR10(cfg.data_root, train=True, download=True, transform=tf)
        return ds, 10


# -----------------------------
#  Distributed worker
# -----------------------------
def ddp_worker(rank: int, world_size: int, cfg: CFG, mom: float):
    # 1) init process group
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 2) device
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 3) seeds（每个 rank 稍微不同，避免完全一致的 dataloader shuffle 序列）
    set_seed(cfg.seed + rank)

    # 4) dataset & partition
    trainset, num_classes = get_dataset(cfg)
    parts = split_indices(len(trainset), world_size, cfg.seed)  # 固定划分（与你原版一致）
    local_idx = parts[rank]

    local_loader = DataLoader(
        Subset(trainset, local_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers_dataloader,
        pin_memory=True,
    )
    local_eval_loader = DataLoader(
        Subset(trainset, local_idx),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers_dataloader,
        pin_memory=True,
    )
    # 全局 eval：每个 rank 都能构建，但只在 rank0 打印/写 CSV
    eval_loader = DataLoader(
        trainset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers_dataloader,
        pin_memory=True,
    )

    # 5) model & opt
    model = resnet20(num_classes).to(device)
    opt = optim.SGD(
        model.parameters(),
        lr=cfg.lr,          # 每 epoch 会改成 cfg.lr / epoch
        momentum=mom,
        weight_decay=cfg.weight_decay,
    )

    # 6) initial sync（确保初始参数一致）
    fedavg_sync_model_(model, world_size)

    # 7) 记录（仅 rank0 保存/打印）
    rec_epoch: List[int] = []
    rec_lr: List[float] = []
    rec_local_loss: List[float] = []
    rec_local_train_loss_end_avg: List[float] = []
    rec_local_train_acc_end_avg: List[float] = []
    rec_train_loss_after_avg: List[float] = []
    rec_train_acc_after_avg: List[float] = []

    # 计时（只 rank0 打印总耗时）
    if rank == 0:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        # lr = cfg.lr / epoch
        lr_epoch = cfg.lr / epoch
        for pg in opt.param_groups:
            pg["lr"] = lr_epoch

        # local train 1 epoch
        local_loss = train_one_epoch(model, opt, local_loader, device)

        # 到通信点：先各自 eval（本地），再 FedAvg，同步后全局 eval
        if epoch % cfg.comm_every == 0:
            # local eval
            local_end_loss, local_end_acc = evaluate(model, local_eval_loader, device)

            # 聚合本地 eval 指标（平均到 rank0）
            t = torch.tensor([local_loss, local_end_loss, local_end_acc], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world_size
            local_loss_avg_workers = float(t[0].item())
            local_train_loss_end_avg = float(t[1].item())
            local_train_acc_end_avg = float(t[2].item())

            # FedAvg sync params
            fedavg_sync_model_(model, world_size)

            # 全局 eval（每个 rank 都算也行；这里为了简单只让 rank0 算）
            if rank == 0:
                train_loss0, train_acc0 = evaluate(model, eval_loader, device)
                lr_now = opt.param_groups[0]["lr"]

                rec_epoch.append(epoch)
                rec_lr.append(lr_now)
                rec_local_loss.append(local_loss_avg_workers)
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

    # 保存 CSV（仅 rank0）
    if rank == 0:
        os.makedirs(cfg.csv_dir, exist_ok=True)
        csv_path = f"{cfg.csv_dir}/{cfg.dataset}_W{world_size}_mom{mom}.csv"
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
                rec_local_loss,
                rec_local_train_loss_end_avg,
                rec_local_train_acc_end_avg,
                rec_train_loss_after_avg,
                rec_train_acc_after_avg,
            ):
                writer.writerow([e, lr, ll, lte, lae, tl, ta])

        torch.cuda.synchronize()
        print(f"Saved CSV to {csv_path}")
        print(f"\n[TOTAL] elapsed time: {time.perf_counter() - t0:.2f} sec")

    dist.destroy_process_group()


def run_multigpu(cfg: CFG):
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 多卡环境才能运行该版本。")

    for workers in cfg.workers_list:
        if workers > torch.cuda.device_count():
            raise RuntimeError(f"workers={workers} 但机器只有 {torch.cuda.device_count()} 张 GPU。")

        for mom in cfg.momentums:
            # world_size = workers = GPU数量（一个 worker 一张卡）
            world_size = workers
            print(f"\n=== multi-gpu FedAvg: world_size(workers)={world_size}, momentum={mom} ===")
            mp.spawn(
                ddp_worker,
                args=(world_size, cfg, mom),
                nprocs=world_size,
                join=True,
            )


def main():
    cfg = CFG()
    print(cfg)
    run_multigpu(cfg)


if __name__ == "__main__":
    main()
