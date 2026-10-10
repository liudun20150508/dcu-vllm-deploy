#!/bin/bash
#SBATCH -J acc-j16-stack
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:30:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j16-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j16-%j.err

# J16：离线安装框架栈（计算节点不能联网 → 全部走本地 find-links）
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
W=$HOME/hcu/wheels
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J16 on $(hostname) $(date) ##########"
echo "离线包：$(ls $W/deps 2>/dev/null | wc -l) 个依赖 + $(ls $W/*.whl 2>/dev/null | wc -l) 个主包"

echo "===== [1] 离线安装（全部 --no-deps：只装包、不让 pip 去解析/替换 torch）====="
WLIST=$(ls $W/deps/*.whl 2>/dev/null $W/*.whl 2>/dev/null | grep -v "/torch-")
echo "待装 whl：$(echo "$WLIST" | wc -w) 个"
$PY -m pip install --user --no-index --no-deps $WLIST 2>&1 | tail -12

echo
echo "===== [2] 验证：torch 是否还是 DCU 版 + 各框架 import ====="
$PY - <<'EOF'
import torch
print(f"torch        : {torch.__version__}  hip={torch.version.hip}  avail={torch.cuda.is_available()}  dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
for name in ("torchvision", "deepspeed", "triton"):
    try:
        m = __import__(name)
        print(f"{name:13}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{name:13}: FAIL -> {type(e).__name__}: {str(e)[:120]}")
EOF

echo "--- vllm（import 较重，单独试）---"
$PY -c "import vllm; print('vllm         :', vllm.__version__)" 2>&1 | tail -6

echo "--- dcu_megatron / megatron ---"
$PY -c "
import importlib
for name in ('dcu_megatron','megatron','megatron.core'):
    try:
        m = importlib.import_module(name); print(f'{name:14}: OK  {getattr(m,\"__version__\",\"\")}')
    except Exception as e:
        print(f'{name:14}: FAIL -> {type(e).__name__}: {str(e)[:110]}')
" 2>&1 | tail -8

echo "===== [3] 一次真实前向（torchvision 模型 + DCU）====="
$PY - <<'EOF' 2>&1 | tail -6
import torch, torchvision
m = torchvision.models.resnet18(num_classes=10).cuda()
x = torch.randn(4, 3, 224, 224, device="cuda")
y = m(x)
print("resnet18 on DCU:", tuple(y.shape), "sum=%.3f" % y.sum().item())
EOF
echo "########## J16 done $(date) ##########"
