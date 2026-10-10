#!/bin/bash
#SBATCH -J acc-j6-mpi
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:10:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j6-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j6-%j.err

# J6：修正版 —— ①MPI 跨节点（二进制放共享 FS）②tuner 插件挂载的完整日志
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
BIN=/public/home/ac6ksiqdok/acc-bin
mkdir -p $BIN
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
echo "########## J6 on $(scontrol show hostname $SLURM_NODELIST | tr '\n' ' ') $(date) ##########"

echo "===== [1] MPI 跨节点 16 进程（二进制在共享 FS）====="
mpicxx -O2 -o $BIN/mpi_bench $WS/mpi-lab/mpi_bench.cpp 2>&1 | tail -3
mpirun --allow-run-as-root -np 16 --map-by ppr:8:node $BIN/mpi_bench 2>&1 | tail -22

echo "===== [2] tuner 插件挂载：完整 NCCL 日志 ====="
cp -f $WS/nccl-tuner-plugin/libnccl-tuner-mycard.so /tmp/librccl-tuner.so
cp -f /tmp/librccl-tuner.so $BIN/librccl-tuner.so
NCCL_NET_PLUGIN=$PLUGIN NCCL_TUNER_PLUGIN=$BIN/librccl-tuner.so \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING,INIT \
mpirun --allow-run-as-root -np 2 --map-by ppr:2:node \
  -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
  -x NCCL_TUNER_PLUGIN -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  /public/software/app/rccl/tests/all_reduce_perf -b 8M -e 8M -f 1 -g 1 -w 1 -n 2 2>&1 \
  | grep -i -E "TUNER|tuner|Avg bus|Out of bounds" | head -30
echo "########## J6 done $(date) ##########"
