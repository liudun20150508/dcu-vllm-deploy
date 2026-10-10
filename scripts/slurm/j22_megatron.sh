#!/bin/bash
#SBATCH -J acc-j22-megatron
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH -t 00:25:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j22-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j22-%j.err

# J22（优先 6 + 5）：Megatron 打通 —— 用 PyPI triton 顶掉缺 libgcvm 的海光 triton，再跑 TP 训练
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
W=$HOME/hcu/wheels
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J22 on $(hostname) $(date) ##########"

echo "===== [1] 诊断：谁在硬 import triton ====="
grep -rn "^import triton" $HOME/.local/lib/python3.12/site-packages/megatron/core/*.py 2>/dev/null | head -5
grep -rn "^import triton" $HOME/.local/lib/python3.12/site-packages/megatron/core/**/*.py 2>/dev/null | head -5
echo "-- 顶层 import 链失败点 --"
$PY -c "import megatron.core" 2>&1 | tail -4

echo
echo "===== [2] 用 PyPI 官方 triton 顶掉海光版（后者缺 libgcvm）====="
$PY -m pip install --user --no-index --no-deps --force-reinstall $W/deps/triton-3.1.0-cp312-cp312-*.whl 2>&1 | tail -2
$PY -c "import triton; print('  triton', triton.__version__, '->', triton.__file__)" 2>&1 | tail -2

echo
echo "===== [3] Megatron 能否 import ====="
$PY -c "import megatron.core as mc; print('  megatron.core:', mc.__version__)" 2>&1 | tail -2
$PY -c "import dcu_megatron; print('  dcu_megatron: OK')" 2>&1 | tail -2

echo
echo "===== [4] Megatron 风格并行训练（TP=2，随机权重，合成数据）====="
cat > /tmp/megatron_tp.py <<'PYEOF'
import os, time, torch
import torch.distributed as dist
import megatron.core.parallel_state as mpu

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl")
# 初始化 Megatron 的并行组（TP=world, PP=1, DP=1）
mpu.initialize_model_parallel(tensor_model_parallel_size=world, pipeline_model_parallel_size=1)
tp_rank = mpu.get_tensor_model_parallel_rank()
tp_group = mpu.get_tensor_model_parallel_group()
torch.cuda.synchronize(); t0 = time.perf_counter()
x = torch.randn(1024, 1024, device="cuda") + tp_rank
for _ in range(20):
    dist.all_reduce(x, group=tp_group)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 20
if rank == 0:
    print(f"  [TP] tp_size={mpu.get_tensor_model_parallel_world_size()} "
          f"all_reduce(4MB) {dt*1e3:.2f} ms -> busbw={4/1024/dt*2*(world-1)/world:.2f} GB/s")
    print(f"  [TP] parallel_state 初始化 OK：TP group size = {torch.distributed.get_world_size(tp_group)}")
mpu.destroy_model_parallel()
dist.destroy_process_group()
PYEOF
$PY -m torch.distributed.run --standalone --nproc_per_node=2 /tmp/megatron_tp.py 2>&1 | grep -a -E "^  \[TP\]|Error|Traceback|ModuleNotFound" | head -8

echo
echo "===== [5] Megatron 的 TP 层（ColumnParallelLinear）真机前向 ====="
cat > /tmp/megatron_layer.py <<'PYEOF'
import os, torch, torch.distributed as dist
import megatron.core.parallel_state as mpu
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local); dist.init_process_group("nccl")
mpu.initialize_model_parallel(tensor_model_parallel_size=world, pipeline_model_parallel_size=1)
col = ColumnParallelLinear(256, 512, config=None, init_method=lambda t: torch.nn.init.normal_(t)).cuda()
row = RowParallelLinear(512, 256, config=None, init_method=lambda t: torch.nn.init.normal_(t)).cuda()
x = torch.randn(8, 256, device="cuda")
y, _ = col(x); z, _ = row(y)
if rank == 0:
    print(f"  [LAYER] ColumnParallelLinear {tuple(x.shape)}->{tuple(y.shape)} -> RowParallelLinear ->{tuple(z.shape)}  OK")
mpu.destroy_model_parallel(); dist.destroy_process_group()
PYEOF
$PY -m torch.distributed.run --standalone --nproc_per_node=2 /tmp/megatron_layer.py 2>&1 | grep -a -E "^  \[LAYER\]|Error|Traceback" | head -6
echo "########## J22 done $(date) ##########"
