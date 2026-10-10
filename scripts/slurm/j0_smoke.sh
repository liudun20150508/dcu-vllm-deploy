#!/bin/bash
#SBATCH -J acc-j0-smoke
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH -t 00:05:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j0-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j0-%j.err

# J0：验收链路冒烟 —— 只探测环境，不做重负载
echo "===== J0 smoke ====="
date
echo "hostname : $(hostname)"
echo "cpus     : $(nproc)"
free -g | head -2
echo "===== RDMA ====="
ls /dev/infiniband/ 2>&1
ibv_devices 2>&1
ibv_devinfo 2>&1 | grep -E "hca_id|link_layer|port_lid|state" | head -20
echo "===== DCU ====="
for c in hy-smi rocm-smi; do echo "-- $c: $(command -v $c || echo MISSING)"; done
command -v hy-smi >/dev/null && hy-smi 2>&1 | head -15
echo "===== dtk / hipcc ====="
ls -d /public/software/compiler/dtk-26.04/bin/hipcc 2>&1
echo "===== python / torch ====="
for p in /usr/bin/python3 python3; do
  echo "-- $p: $($p --version 2>&1)"
  $p -c "import torch;print('   torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())" 2>&1 | head -3
done
ls /public/home/ac6ksiqdok/.local/lib/python3.12/site-packages 2>/dev/null | head -5
echo "===== done ====="
