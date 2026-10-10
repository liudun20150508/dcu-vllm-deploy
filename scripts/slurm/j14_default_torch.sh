#!/bin/bash
#SBATCH -J acc-j14-default
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j14-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j14-%j.err

# J14：验证 DCU 版 torch 已装进 user-site ——【不设 PYTHONPATH】直接 import
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
# 关键：这里刻意不设 PYTHONPATH / PYTHONNOUSERSITE
unset PYTHONPATH || true
PY=/usr/local/python3.12/bin/python3
echo "########## J14 on $(hostname) $(date) ##########"
echo "PYTHONPATH='${PYTHONPATH:-<空>}'"

echo "===== [1] 默认 import 到的是哪一份？ ====="
$PY - <<'EOF' 2>&1 | grep -v "^ *$" | tail -14
import os, sys, torch
print("torch.__file__ :", torch.__file__)
print("version        :", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("is_available   :", torch.cuda.is_available())
print("device count   :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0       :", torch.cuda.get_device_name(0))
    print("DCU 求和       :", torch.ones(1024, device="cuda").sum().item())
    print("nccl 版本      :", torch.cuda.nccl.version())
EOF

echo "===== [2] four_ops_test.py（2 卡，不带任何 PYTHONPATH 设置）====="
cd $WS/experiments
$PY -m torch.distributed.run --standalone --nproc_per_node=2 four_ops_test.py 2>&1 \
  | grep -a -E "ALL MATCH|全部通过|^2,|Error|Traceback: " | head -10
echo "########## J14 done $(date) ##########"
