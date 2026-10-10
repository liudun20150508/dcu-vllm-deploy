#!/bin/bash
#SBATCH -J acc-j11-hcutorch
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j11-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j11-%j.err

# J11：用【海光 DTK 配套版 torch】跑通 DCU —— torch-2.5.1+das.opt1.dtk2604 (cp312)
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
TP=$HOME/hcu/torch2501-dtk2604-py312
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
echo "########## J11 on $(hostname) $(date) ##########"
echo "DTKROOT=$DTKROOT  ROCM_PATH=${ROCM_PATH:-}"
echo "HIP_PATH=${HIP_PATH:-}"

export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$TP
PY=/usr/local/python3.12/bin/python3
echo "PY=$PY  ($($PY --version 2>&1))"
echo "torch 来自: $(ls -d $TP/torch 2>/dev/null || echo '【缺失】')"

echo "===== [1] 自检：torch 能否认出 DCU ====="
$PY - <<'EOF' 2>&1 | tail -20
import os, torch
print("torch        :", torch.__version__)
print("torch.hip    :", torch.version.hip)
print("torch 路径   :", os.path.dirname(torch.__file__))
print("ROCm path    :", getattr(torch.version, "hip", None))
print("is_available :", torch.cuda.is_available())
print("device count :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0     :", torch.cuda.get_device_name(0))
    x = torch.ones(1024, device="cuda")
    print("DCU 上求和   :", x.sum().item())
    print("矩阵乘测试   :", (torch.randn(256,256,device='cuda') @ torch.randn(256,256,device='cuda')).sum().item())
    print("nccl 版本    :", torch.cuda.nccl.version())
EOF

echo "===== [2] 跑 four_ops_test.py（2 卡，四个集合操作）====="
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
cd $WS/experiments
$PY -m torch.distributed.run --standalone --nproc_per_node=2 four_ops_test.py 2>&1 | tail -35

echo "===== [3] 对照：DDP 梯度规约（训练那一步的通信）====="
cat > /tmp/ddp_smoke_hcu.py <<'EOF'
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); w = dist.get_world_size()
torch.cuda.set_device(r)
m = torch.nn.Linear(1024, 1024).cuda()
m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[r])
x = torch.randn(64, 1024, device="cuda")
loss = m(x).sum(); loss.backward()
g = m.module.weight.grad
ok = torch.allclose(g, g[0:1].expand_as(g), atol=1e-2)
if r == 0:
    print(f"[ddp] world={w} 梯度已跨卡规约(各 rank 一致)={ok} busbw≈(用 all_reduce 计时)")
import time
t = torch.randn(64*1024*1024//4, device="cuda")
dist.barrier(); t0 = time.perf_counter()
for _ in range(5): dist.all_reduce(t)
torch.cuda.synchronize(); dt = (time.perf_counter()-t0)/5
if r == 0:
    print(f"[ddp] all_reduce 64MB: {dt*1e3:.2f} ms  algbw={t.numel()*4/dt/1e9:.2f} GB/s  busbw={t.numel()*4/dt/1e9*2*(w-1)/w:.2f} GB/s")
dist.destroy_process_group()
EOF
$PY -m torch.distributed.run --standalone --nproc_per_node=2 /tmp/ddp_smoke_hcu.py 2>&1 | grep -E "\[ddp\]|Error|error" | head -10

echo "########## J11 done $(date) ##########"
