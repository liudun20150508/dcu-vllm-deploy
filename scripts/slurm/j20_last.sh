#!/bin/bash
#SBATCH -J acc-j20-last
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j20-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j20-%j.err

set -u
W=$HOME/hcu/wheels
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J20 on $(hostname) $(date) ##########"

echo "===== [1] 补 httpx2（huggingface_hub 2.0 需要）====="
$PY -m pip install --user --no-index --no-deps $W/deps/httpx2-*.whl 2>&1 | tail -3

echo
echo "===== [2] 框架栈最终清单 ====="
for mod in torch torchvision deepspeed vllm transformers huggingface_hub triton; do
  printf "  %-17s: " "$mod"
  $PY -c "
import importlib
m = importlib.import_module('$mod'); print(getattr(m,'__version__','OK'))
" 2>&1 | tail -1
done
printf "  %-17s: " "megatron.core"; $PY -c "import megatron.core; print('OK')" 2>&1 | tail -1
printf "  %-17s: " "dcu_megatron";    $PY -c "import dcu_megatron; print('OK')" 2>&1 | tail -1

echo
echo "===== [3] 端到端确认（torch+torchvision 训练一步）====="
$PY -c "
import torch, torchvision, time
m = torchvision.models.resnet18(num_classes=10).cuda()
opt = torch.optim.Adam(m.parameters(), lr=1e-4)
x = torch.randn(32,3,224,224,device='cuda'); y = torch.randint(0,10,(32,),device='cuda')
for i in range(3):
    torch.cuda.synchronize(); t0=time.perf_counter()
    loss = torch.nn.functional.cross_entropy(m(x), y); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); print(f'  step{i+1}: loss={loss.item():.4f}  {(time.perf_counter()-t0)*1e3:.1f} ms')
" 2>&1 | grep -a -E "^  step|Error" | tail -4
echo "########## J20 done $(date) ##########"
