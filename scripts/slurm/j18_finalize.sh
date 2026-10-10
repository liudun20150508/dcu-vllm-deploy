#!/bin/bash
#SBATCH -J acc-j18-final
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j18-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j18-%j.err

# J18：收尾 —— 修 pydantic-core、用正确模块名测 megatron、查 libgcvm、总验证
set -u
W=$HOME/hcu/wheels
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J18 on $(hostname) $(date) ##########"

echo "===== [1] 修 pydantic-core ====="
$PY -m pip install --user --no-index --no-deps --force-reinstall \
    $W/deps/pydantic_core-2.46.5-*.whl 2>&1 | tail -3
$PY -c "import pydantic_core, pydantic; print('pydantic', pydantic.VERSION, '/ core', pydantic_core.__version__)" 2>&1 | tail -2

echo
echo "===== [2] libgcvm 在哪（计算节点视野）====="
for d in /opt/dtk-25.04.4 /opt/dtk-26.04 /opt/hyhal /public/software/compiler/dtk-26.04 /public/software/compiler/dtk-25.04.4; do
  r=$(ls $d/lib*/libgcvm* 2>/dev/null); [ -n "$r" ] && echo "  找到: $d -> $r"
done
ls /opt 2>/dev/null | head -20
echo "  (若都没找到，说明 DTK 26.04 未随附该库)"

echo
echo "===== [3] 逐个框架最终状态 ====="
for mod in torch torchvision deepspeed triton vllm dcu_megatron transformers; do
  printf "  %-14s: " "$mod"
  $PY -c "
import importlib
m = importlib.import_module('$mod')
v = getattr(m, '__version__', 'OK')
if '$mod' == 'torch':
    v = f'{m.__version__} hip={m.version.hip} avail={m.cuda.is_available()} dev={m.cuda.get_device_name(0) if m.cuda.is_available() else \"N/A\"}'
print(v)
" 2>&1 | tail -1
done

echo
echo "===== [4] 端到端：DCU 上做一次真实训练步（torchvision 模型 + DDP + Adam）====="
cat > /tmp/train1step.py <<'EOF'
import torch, torch.distributed as dist, torchvision, time
dist.init_process_group("nccl"); r = dist.get_rank()
torch.cuda.set_device(r)
m = torchvision.models.resnet18(num_classes=10).cuda()
if dist.get_world_size() > 1:
    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[r])
opt = torch.optim.Adam(m.parameters(), lr=1e-4)
x = torch.randn(16, 3, 224, 224, device="cuda"); y = torch.randint(0, 10, (16,), device="cuda")
torch.cuda.synchronize(); t0 = time.perf_counter()
loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
if r == 0:
    print(f"  训练一步 OK: loss={loss.item():.4f}  用时={(time.perf_counter()-t0)*1e3:.1f} ms  world={dist.get_world_size()}")
dist.destroy_process_group()
EOF
$PY -m torch.distributed.run --standalone --nproc_per_node=1 /tmp/train1step.py 2>&1 | grep -a -E "训练一步|Error|Traceback" | head -5

echo
echo "===== [5] vLLM 自检（不下载模型，只看能否建引擎配置）====="
$PY -c "
import vllm
print('  vllm', vllm.__version__)
from vllm import LLM
print('  LLM 类可导入 OK（未加载权重，避免联网）')
" 2>&1 | tail -4
echo "########## J18 done $(date) ##########"
