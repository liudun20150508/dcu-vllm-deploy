#!/bin/bash
#SBATCH -J acc-j8-tuner
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j8-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j8-%j.err

# J8：①tuner 插件完整日志（不过滤，含 backtrace）②稳定性逐轮 8 次
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
BIN=/public/home/ac6ksiqdok/acc-bin
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
AR=/public/software/app/rccl/tests/all_reduce_perf
echo "########## J8 on $(scontrol show hostname $SLURM_NODELIST | tr '\n' ' ') $(date) ##########"

echo "===== [1] tuner 插件：完整日志（末 60 行）====="
NCCL_NET_PLUGIN=$PLUGIN NCCL_TUNER_PLUGIN=$BIN/librccl-tuner.so \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING,INIT \
mpirun --allow-run-as-root -np 2 --map-by ppr:2:node \
  -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
  -x NCCL_TUNER_PLUGIN -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  $AR -b 8M -e 8M -f 1 -g 1 -w 1 -n 2 > /tmp/j8_tuner.log 2>&1
echo "exit=$?"
tail -60 /tmp/j8_tuner.log

echo "===== [2] 稳定性：128MB AllReduce 逐轮 8 次 ====="
for i in 1 2 3 4 5 6 7 8; do
  V=$(NCCL_NET_PLUGIN=$PLUGIN \
      mpirun --allow-run-as-root -np 16 --map-by ppr:8:node \
      -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
      $AR -b 128M -e 128M -f 1 -g 1 -w 1 -n 3 2>&1 | grep "Avg bus bandwidth")
  echo "  轮 $i : $V"
done
echo "########## J8 done $(date) ##########"
