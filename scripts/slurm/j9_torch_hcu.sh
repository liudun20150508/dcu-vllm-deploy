#!/bin/bash
#SBATCH -J acc-j9-torch
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j9-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j9-%j.err

# J9：找 python3.12 → 用已装好的海光版 torch（2.5.1+rocm6.2）跑 four_ops_test.py
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
echo "########## J9 on $(hostname) $(date) ##########"

echo "===== [1] 找 python3.12 ====="
for p in /usr/local/python3.12/bin/python3 /usr/local/bin/python3.12 /opt/python3.12/bin/python3 \
         /public/software/apps/python/3.12*/bin/python3; do
  printf "%-45s %s\n" "$p" "$([ -x "$p" ] && echo YES || echo no)"
done
echo "-- which --"; which python3.12 python3.11 python3.10 python3 2>&1
echo "-- python3 版本 --"; python3 --version 2>&1
echo "-- /usr/local/bin 里的 python --"; ls /usr/local/bin 2>/dev/null | grep -i py || echo "(none)"
echo "-- /usr/local 顶层 --"; ls /usr/local 2>&1
echo "-- anaconda3 可用版本 --"; ls /public/software/apps/anaconda3/ 2>&1
echo "-- 用户 site-packages --"; ls -d $HOME/.local/lib/python3.12/site-packages/torch 2>&1
echo "-- /opt --"; ls /opt 2>&1 | head
echo "-- 深度查找 python3.12（上限 30s）--"
timeout 30 find /usr /opt /public/software -maxdepth 4 -name "python3.12" 2>/dev/null | head

echo "===== [2] 若有 python3.12：用海光版 torch 自检 ====="
PY=""
for c in /usr/local/python3.12/bin/python3 "$(command -v python3.12 2>/dev/null)"; do
  [ -n "$c" ] && [ -x "$c" ] && PY="$c" && break
done
if [ -z "$PY" ]; then
  echo "[MISS] 计算节点也没有 python3.12 解释器 —— torch(2.5.1+rocm6.2) 无法 import"
  exit 0
fi
echo "用 $PY"
module load compiler/dtk/26.04
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
$PY - <<'EOF'
import torch, os
print("torch        :", torch.__version__)
print("torch.hip    :", torch.version.hip)
print("hsa/hip avail:", torch.cuda.is_available())
print("device count :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name  :", torch.cuda.get_device_name(0))
    x = torch.ones(4, device="cuda"); print("tensor on DCU:", x.sum().item())
    print("nccl version :", torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else "n/a")
EOF

echo "===== [3] 跑 four_ops_test.py（2 卡）====="
cd $WS/experiments
$PY -m torch.distributed.run --standalone --nproc_per_node=2 four_ops_test.py 2>&1 | tail -30
echo "########## J9 done $(date) ##########"
