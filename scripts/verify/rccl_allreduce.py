#!/usr/bin/env python3
"""RCCL all_reduce 正确性验证（torch.distributed nccl 后端）。

验证「前置条件 5：RCCL 通信」。所有 rank 各自 all_reduce 一个全 1 张量，
结果应等于 WORLD_SIZE（sum=world_size）→ ALLREDUCE OK。

用法（单机）：
  torchrun --standalone --nproc_per_node=8 rccl_allreduce.py
用法（多机，Slurm）：
  见 rccl_allreduce.sh 里的 srun 模板，通过 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT 传参。
"""
import os

rank = int(os.environ.get("RANK", "0"))
ws = int(os.environ.get("WORLD_SIZE", "1"))
print(f"[rank {rank}] python start pid={os.getpid()}", flush=True)

import torch
print(f"[rank {rank}] torch {torch.__version__} cuda_avail={torch.cuda.is_available()}",
      flush=True)

import torch.distributed as dist

dist.init_process_group(
    "nccl",  # 海光上仍叫 nccl 后端，实际走 RCCL
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    rank=rank,
    world_size=ws,
)
torch.cuda.set_device(0)

x = torch.ones(8192, device="cuda")
dist.all_reduce(x)
s = x[0].item()
verdict = "OK" if s == float(ws) else "WRONG"
print(f"[rank {rank}] ALLREDUCE {verdict} sum={s} ws={ws}", flush=True)

dist.destroy_process_group()
print(f"[rank {rank}] clean exit", flush=True)
