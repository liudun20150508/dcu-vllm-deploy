#!/bin/bash
#SBATCH -J multi-train
#SBATCH -p hx1hdnormal
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:30:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/multi-train-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/multi-train-%j.err
# ============================================================
# 多节点分布式训练（JD 优先 5「大规模训练」）
#   8 节点 × 8 卡 = 64 卡（账号限额 80 卡内），纯数据并行（DP=world）。
#   量出「跨节点 DP 梯度 AllReduce」的带宽 —— 训练里最大、最同步的通信流。
#   启动：mpirun 直启 64 个 python 进程，用 OpenMPI 的 rank 环境变量注入 torch 的 env:// 汇聚。
# ============================================================
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
export NCCL_IB_HCA=shca
PY=/usr/local/python3.12/bin/python3
export TP_SIZE=8

# 主节点地址：取第一个节点的 IPv4（管理网可达），供 torch env:// 汇聚用
MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
MASTER_ADDR=$(scontrol show node "$MASTER_NODE" 2>/dev/null | grep -oP 'NodeAddr=\K\S+' | head -1)
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=$(getent hosts "$MASTER_NODE" | awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+/) {print $i; exit}}')
export MASTER_ADDR
export MASTER_PORT=29511
echo "主节点 $MASTER_NODE -> $MASTER_ADDR:$MASTER_PORT"

cd "$WS/experiments"
# 先确认数据集存在
[ -f train_X.npy ] || { echo "缺数据集，先跑 make_dataset.py"; exit 1; }

echo "===== 串行预编译 torch dlpack 扩展（避免 16 进程抢锁死锁）====="
$PY -c "import torch; import torch.utils.dlpack" 2>&1 | tail -2 || echo "(预编译跳过)"

echo "===== 跨节点分布式训练（16 进程 = 2 节点 × 8 卡）====="
mpirun --allow-run-as-root -np 64 --map-by ppr:8:node \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x HSA_FORCE_FINE_GRAIN_PCIE \
  -x NCCL_NET_PLUGIN -x NCCL_IB_HCA -x TP_SIZE -x MASTER_ADDR -x MASTER_PORT \
  bash -c 'export RANK=$OMPI_COMM_WORLD_RANK \
                 WORLD_SIZE=$OMPI_COMM_WORLD_SIZE \
                 LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK; \
           exec '"$PY"' distributed_train.py' 2>&1
echo "===== exit=$? ====="
