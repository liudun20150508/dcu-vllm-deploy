#!/bin/bash
#SBATCH -J acc-j17-fix
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j17-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j17-%j.err

# J17：amdsmi 与海光 libamd_smi.so 不兼容 → 卸掉它，逐框架重测
set -u
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J17 on $(hostname) $(date) ##########"

echo "===== [0] 装的是哪个 amdsmi ====="
$PY -m pip show amdsmi 2>/dev/null | head -3
echo "-- 海光 libamd_smi.so 是否有 amdsmi_get_gpu_enumeration_info --"
nm -D /opt/hyhal/lib/libamd_smi.so 2>/dev/null | grep -c amdsmi_get_gpu_enumeration_info

echo
echo "===== [1] 卸载 amdsmi ====="
$PY -m pip uninstall -y amdsmi 2>&1 | tail -3

echo
echo "===== [2] 逐个框架验证（每项独立进程，失败不影响其它）====="
for mod in torch torchvision deepspeed triton vllm dcu_megatron; do
  printf "%-14s: " "$mod"
  $PY -c "
import importlib
m = importlib.import_module('$mod')
extra = ''
if '$mod' == 'torch':
    extra = f\" hip={m.version.hip} avail={m.cuda.is_available()} dev={m.cuda.get_device_name(0) if m.cuda.is_available() else 'N/A'}\"
print(getattr(m, '__version__', 'OK') + extra)
" 2>&1 | tail -1
done

echo
echo "===== [3] dcu_megatron 提供了什么模块名 ====="
$PY -c "
import site, os, glob
for p in site.getsitepackages() + [site.getusersitepackages()]:
    for pat in ('*megatron*', '*Megatron*'):
        for x in glob.glob(os.path.join(p, pat)):
            print(' ', os.path.basename(x))
" 2>&1 | head -12

echo
echo "===== [4] 真实前向：torchvision resnet18 on DCU ====="
$PY -c "
import torch, torchvision
m = torchvision.models.resnet18(num_classes=10).cuda()
y = m(torch.randn(4,3,224,224,device='cuda'))
print('resnet18 on DCU ->', tuple(y.shape), 'sum=%.3f' % y.sum().item())
" 2>&1 | tail -4

echo "########## J17 done $(date) ##########"
