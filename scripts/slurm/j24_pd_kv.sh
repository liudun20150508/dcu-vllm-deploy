#!/bin/bash
#SBATCH -J acc-j24-pdkv
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:1
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j24-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j24-%j.err

# J24（优先 7）：PD 分离 / KV Cache 搬运实测 —— 跨节点点对点 + 与梯度规约抢带宽
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
PY=/usr/local/python3.12/bin/python3
HOSTS=($(scontrol show hostname $SLURM_NODELIST))
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=29517
echo "########## J24 on ${HOSTS[*]} $(date) ##########"
echo "MASTER_ADDR=$MASTER_ADDR"

mpirun --allow-run-as-root -np 2 --map-by ppr:1:node \
  -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x HSA_FORCE_FINE_GRAIN_PCIE \
  -x NCCL_NET_PLUGIN -x MASTER_ADDR -x MASTER_PORT -x NCCL_TOPO_FILE -x RCCL_NET_PLANE \
  $PY $WS/experiments/pd_kv_cache_bench.py 2>&1 | grep -a -v -E "^\[|Warning|warn|^$" | tail -40
echo "########## J24 done $(date) ##########"
