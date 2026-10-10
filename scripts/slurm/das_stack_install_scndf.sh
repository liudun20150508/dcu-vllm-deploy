#!/bin/bash
#SBATCH -J das-install
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:30:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/das-install-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/das-install-%j.err

# DAS 1.8 框架栈离线安装（当前账号 scndfy1yuj，计算节点 glibc≥2.28 才能装 manylinux_2_28 wheel）
set -u
W=/public/home/scndfy1yuj/中科曙光/曙光环境/hcu/wheels
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## DAS install on $(hostname) $(date) ##########"
echo "DTKROOT=$DTKROOT  python=$($PY --version 2>&1)"

# 主包 + deps；排除：
#   torch（已解包进 user-site）、huggingface_hub-2.0.0（改用 1.33.0）、transformers-4.57.6（改用 5.17.0）
WLIST=$(ls $W/*.whl $W/deps/*.whl 2>/dev/null | grep -v '/torch-' | grep -v 'huggingface_hub-2.0.0' | grep -v 'transformers-4.57.6')
echo "待装 whl: $(echo "$WLIST" | wc -w) 个"
$PY -m pip install --user --no-index --no-deps $WLIST 2>&1 | tail -8

echo
echo "===== 框架栈 import 清单 ====="
$PY - <<'EOF'
import torch
print(f"torch          : {torch.__version__}  hip={torch.version.hip}  avail={torch.cuda.is_available()}  dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
for name in ("torchvision","deepspeed","vllm","transformers","huggingface_hub","numpy","sympy","triton","dcu_megatron"):
    try:
        m = __import__(name)
        print(f"{name:16}: {getattr(m,'__version__','OK')}")
    except Exception as e:
        print(f"{name:16}: FAIL -> {type(e).__name__}: {str(e)[:100]}")
EOF

echo
echo "===== resnet18 前向（真实 DCU）====="
$PY - <<'EOF' 2>&1 | tail -4
import torch, torchvision
m = torchvision.models.resnet18(num_classes=10).cuda()
x = torch.randn(4, 3, 224, 224, device="cuda")
y = m(x)
print("resnet18 on DCU:", tuple(y.shape), "sum=%.3f" % y.sum().item())
EOF
echo "########## DAS install done $(date) ##########"
