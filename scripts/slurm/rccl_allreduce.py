import os

rank = int(os.environ.get("RANK", "0"))
ws = int(os.environ.get("WORLD_SIZE", "1"))
print(f"[rank {rank}] python start pid={os.getpid()}", flush=True)

import torch
print(f"[rank {rank}] torch {torch.__version__} cuda_avail={torch.cuda.is_available()}", flush=True)

import torch.distributed as dist

dist.init_process_group(
    "nccl",
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
