#!/bin/bash
#SBATCH -J acc-j13-hcu8b
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:8
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j13-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j13-%j.err

# J13：干净版 —— 完整日志（不 grep）+ torch 层 all_reduce 带宽 + traceback 诊断
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
L=$WS/验收/logs
export PYTHONPATH=$HOME/hcu/torch2501-dtk2604-py312
export PYTHONNOUSERSITE=1
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J13 on $(hostname) $(date) ##########"

echo "===== [1] 完整日志：four_ops_test.py 8 卡 ====="
cd $WS/experiments
$PY -m torch.distributed.run --standalone --nproc_per_node=8 four_ops_test.py > $L/j13-four_ops-full.log 2>&1
echo "exit=$?  完整日志: $L/j13-four_ops-full.log  ($(wc -l < $L/j13-four_ops-full.log) 行)"
echo "--- 开头的 traceback 到底是什么（前 25 行）---"
head -25 $L/j13-four_ops-full.log
echo "--- 结论行 ---"
grep -a -E "ALL MATCH|全部通过|^2,|^8," $L/j13-four_ops-full.log | head -12

echo
echo "===== [2] torch 层 all_reduce 带宽（8 卡，逐档）====="
cat > /tmp/bw_hcu.py <<'EOF'
import time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
print(f"[bw] world={w} backend=nccl device={torch.cuda.get_device_name(r)}", flush=True)
for mb in (8, 16, 64, 256):
    n = mb * 1024 * 1024 // 4
    t = torch.randn(n, device="cuda")
    for _ in range(3): dist.all_reduce(t)
    torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
    for _ in range(10): dist.all_reduce(t)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 10
    if r == 0:
        a = mb / 1024 / dt
        print(f"[bw] all_reduce {mb:4d}MB  {dt*1e3:8.3f} ms  algbw={a:6.2f} GB/s  busbw={a*2*(w-1)/w:6.2f} GB/s", flush=True)
dist.destroy_process_group()
EOF
$PY -m torch.distributed.run --standalone --nproc_per_node=8 /tmp/bw_hcu.py > $L/j13-bw-full.log 2>&1
echo "exit=$?  完整日志: $L/j13-bw-full.log"
grep -a -E "^\[bw\]|Error|Traceback|Exception" $L/j13-bw-full.log | head -15
echo "--- 若有异常，看最后 20 行 ---"
tail -20 $L/j13-bw-full.log
echo "########## J13 done $(date) ##########"
