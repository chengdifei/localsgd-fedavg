# localsgd-fedavg
Reference implementations of Local SGD / Periodic FedAvg for classification and regression, including both single-GPU simulated workers and true multi-GPU execution.

README
======

This repository contains experimental code for studying Local SGD / Periodic FedAvg
under different  tasks (classification and regression).

All experiments are designed such that, except for the execution mode, the algorithm,
data partitioning, optimization hyperparameters, and evaluation protocols remain
identical across implementations.

！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

The single-GPU simulated version is included to enable researchers without access to multi-GPU environments to reproduce the experiments under the same algorithmic settings.


！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！



--------------------------------------------------
Overview of Files
--------------------------------------------------

1. classification_fedavg_single_gpu.py

   - Task: Image classification (e.g., CIFAR-10 / CIFAR-100)
   - Model: ResNet-20 (CIFAR-style)
   - Execution: Single GPU, single process
   - Parallelism: Multiple workers are simulated inside one process
   - Purpose:
       * Baseline implementation
       * Easy debugging and controlled experimentation
       * Algorithmic reference for Local SGD / FedAvg


2. classification_fedavg_multi_gpu.py

   - Task: Image classification (e.g., CIFAR-10 / CIFAR-100)
   - Model: ResNet-20 (CIFAR-style)
   - Execution: Multi-GPU, multi-process
   - Parallelism: One worker per GPU (one process per GPU)
   - Communication: torch.distributed (NCCL backend)
   - Purpose:
       * Scalable implementation
       * Performance comparison with single-GPU simulation
       * Matches the single-GPU version algorithmically


3. regression_fedavg_single_gpu.py

   - Task: Regression (Boston Housing dataset)
   - Model: Fully-connected network (FC4)
   - Execution: Single GPU, single process
   - Parallelism: Multiple workers simulated in one process
   - Loss: Mean Squared Error (MSE)
   - Purpose:
       * Lightweight regression benchmark
       * Verifies Local SGD / FedAvg behavior beyond classification


4. regression_fedavg_multi_gpu.py

   - Task: Regression (Boston Housing dataset)
   - Model: Fully-connected network (FC4)
   - Execution: Multi-GPU, multi-process
   - Parallelism: One worker per GPU
   - Communication: torch.distributed (NCCL backend)
   - Purpose:
       * Multi-GPU counterpart of the regression experiment
       * Same algorithm and hyperparameters as single-GPU version


--------------------------------------------------
Common Experimental Setup
--------------------------------------------------

Across all files, the following settings are kept consistent:

- Data partitioning:
    * The dataset is split deterministically across workers using a fixed random seed.
    * Each worker trains only on its local subset.

- Optimization:
    * Optimizer: SGD
    * Momentum: configurable (e.g., 0.0, 0.5, 0.9)
    * Weight decay: fixed
    * Learning rate schedule: lr = base_lr / epoch (inverse decay)

- Training scheme:
    * Local SGD on each worker
    * Periodic model averaging (FedAvg) every 'comm_every' epochs
    * No gradient synchronization between communication rounds

- Evaluation:
    * Classification: accuracy and cross-entropy loss
    * Regression: mean squared error (MSE)
    * Global evaluation is performed after FedAvg


--------------------------------------------------
Single-GPU vs. Multi-GPU Implementations
--------------------------------------------------

Single-GPU versions:
- Use one process and one GPU.
- Multiple workers are simulated by maintaining multiple models and optimizers.
- Useful for debugging, ablation studies, and controlled experiments.

Multi-GPU versions:
- Use torch.multiprocessing and torch.distributed.
- Each worker runs in an independent process bound to one GPU.
- Parameter averaging is implemented via all-reduce.
- Algorithmically equivalent to the single-GPU versions.


--------------------------------------------------
Notes
--------------------------------------------------

- In the current implementations, BatchNorm synchronization is NOT used.
  Each worker maintains its own local BatchNorm statistics (local BN).

- The regression models do not contain BatchNorm layers, so single-GPU and
  multi-GPU implementations are fully equivalent in this respect.

- The term "worker" refers to an independent local model participating in
  Local SGD / FedAvg. In the multi-GPU setting, the number of workers equals
  the number of GPU processes (world_size).


--------------------------------------------------
Usage
--------------------------------------------------

Single-GPU experiments:
    python classification_fedavg_single_gpu.py
    python regression_fedavg_single_gpu.py

Multi-GPU experiments:
    python classification_fedavg_multi_gpu.py
    python regression_fedavg_multi_gpu.py

To restrict visible GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python classification_fedavg_multi_gpu.py


--------------------------------------------------
Summary
--------------------------------------------------

These implementations provide both a single-GPU execution that simulates multiple workers and a true multi-GPU execution for Local SGD / FedAvg, while keeping the underlying algorithm unchanged.
