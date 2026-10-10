#!/bin/bash
#SBATCH -J acc-j19-verify
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j19-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j19-%j.err

# J19：补齐 httpx / megatron-core，做框架栈最终验证
set -u
W=$HOME/hcu/wheels
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J19 on $(hostname) $(date) ##########"

echo "===== [1] 补装 ====="
$PY -m pip install --user --no-index --no-deps \
  $W/deps/httpx-*.whl $W/deps/httpcore-*.whl $W/deps/h11-*.whl \
  $W/deps/certifi-*.whl $W/deps/idna-*.whl $W/deps/sniffio-*.whl \
  $W/deps/megatron_core-*.whl 2>&1 | tail -4

echo
echo "===== [2] 框架栈最终清单（每项独立进程）====="
for mod in torch torchvision deepspeed vllm transformers triton httpx; do
  printf "  %-14s: " "$mod"
  $PY -c "
import importlib
m = importlib.import_module('$mod'); v = getattr(m, '__version__', 'OK')
if '$mod' == 'torch':
    v = f'{m.__version__}  hip={m.version.hip}  avail={m.cuda.is_available()}  dev={m.cuda.get_device_name(0)}'
print(v)
" 2>&1 | tail -1
done
printf "  %-14s: " "megatron.core"
$PY -c "import megatron.core as mc; print(getattr(mc,'__version__','OK'))" 2>&1 | tail -1
printf "  %-14s: " "dcu_megatron"
$PY -c "import dcu_megatron as d; print(getattr(d,'__version__','OK'))" 2>&1 | tail -1

echo
echo "===== [3] 端到端：torchvision 训练一步 + vLLM 能建引擎 ====="
$PY -c "
import torch, torchvision, time
m = torchvision.models.resnet18(num_classes=10).cuda()
opt = torch.optim.Adam(m.parameters(), lr=1e-4)
x = torch.randn(32,3,224,224,device='cuda'); y = torch.randint(0,10,(32,),device='cuda')
for i in range(3):
    torch.cuda.synchronize(); t0=time.perf_counter()
    loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); print(f'  step{i+1}: loss={loss.item():.4f} {(time.perf_counter()-t0)*1e3:.1f} ms')
" 2>&1 | grep -a -E "^  step|Error" | tail -4
$PY -c "
from vllm import LLM
print('  vLLM: LLM 入口可导入（未加载权重，避免联网）')
" 2>&1 | tail -3
echo "########## J19 done $(date) ##########"
