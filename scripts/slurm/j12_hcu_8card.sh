#!/bin/bash
#SBATCH -J acc-j12-hcu8
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:8
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j12-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j12-%j.err

# J12：海光版 torch 上 —— ①8 卡四操作 ②DDP 梯度规约 ③torch 层 all_reduce 带宽
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
export PYTHONPATH=$HOME/hcu/torch2501-dtk2604-py312
export PYTHONNOUSERSITE=1
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J12 on $(hostname) $(date) ##########"

echo "===== [1] four_ops_test.py（8 卡单节点）====="
cd $WS/experiments
$PY -m torch.distributed.run --standalone --nproc_per_node=8 four_ops_test.py 2>&1 | grep -a -E "ALL MATCH|nranks|^2,|全部通过|Error|Traceback" | head -20

echo "===== [2] DDP 梯度规约（训练一步的通信）====="
cat > /tmp/ddp_hcu.py <<'EOF'
import time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
m = torch.nn.parallel.DistributedDataParallel(
        torch.nn.Linear(1024, 1024).cuda(), device_ids=[r])
x = torch.randn(64, 1024, device="cuda")
m(x).sum().backward()
g = m.module.weight.grad
same = torch.allclose(g, g[0:1].expand_as(g), atol=1e-2)
if r == 0:
    print(f"[ddp] world={w} 梯度跨卡规约一致={same}")
for mb in (16, 64, 256):
    t = torch.randn(mb * 1024 * 1024 // 4, device="cuda")
    for _ in range(3): dist.all_reduce(t)
    dist.barrier(); t0 = time.perf_counter()
    for _ in range(10): dist.all_reduce(t)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 10
    if r == 0:
        algbw = mb / 1024 / dt
        print(f"[ddp] all_reduce {mb:4d}MB: {dt*1e3:8.2f} ms  algbw={algbw:6.2f} GB/s  busbw={algbw*2*(w-1)/w:6.2f} GB/s")
dist.destroy_process_group()
EOF
$PY -m torch.distributed.run --standalone --nproc_per_node=8 /tmp/ddp_hcu.py 2>&1 | grep -a -E "^\[ddp\]|Error|Traceback" | head -10

echo "===== [3] 对照：RCCL 官方 all_reduce_perf（同 8 卡）====="
if command -v mpirun >/dev/null 2>&1; then
  mpirun --allow-run-as-root -np 8 /public/software/app/rccl/tests/all_reduce_perf \
    -b 16M -e 256M -f 4 -g 1 -w 2 -n 3 2>&1 | tail -8
else
  echo "(无 mpirun，跳过)"
fi
echo "########## J12 done $(date) ##########"
