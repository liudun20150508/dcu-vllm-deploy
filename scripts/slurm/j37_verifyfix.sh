#!/bin/bash
#SBATCH -J acc-j37-verifyfix
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j37-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j37-%j.err

# J37：验证两个已修的"小问题"（对应验收报告 §5.1 问题 2、4）：
#   ① tuner 扁平数组修复 —— 真 RCCL 上 getCollInfo 不再 SIGSEGV
#   ② train_comm_sim —— MPI_Bcast 替代文件 bootstrap，不再 ncclCommInitRank invalid usage
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
AR=/public/software/app/rccl/tests/all_reduce_perf
echo "########## J37 on $(scontrol show hostname $SLURM_NODELIST | tr '\n' ' ') $(date) ##########"

echo "===== [1] tuner 插件：从修复后源码重编 + 真 RCCL 挂载 ====="
cd "$WS/nccl-tuner-plugin"
make clean >/dev/null 2>&1
make >/dev/null 2>&1 || echo "[FAIL] tuner make"
cp -f libnccl-tuner-mycard.so /tmp/librccl-tuner.so
NCCL_NET_PLUGIN=$PLUGIN NCCL_TUNER_PLUGIN=/tmp/librccl-tuner.so \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING,INIT \
mpirun --allow-run-as-root -np 2 --map-by ppr:2:node \
  -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
  -x NCCL_TUNER_PLUGIN -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  $AR -b 8M -e 8M -f 1 -g 1 -w 1 -n 2 > /tmp/j37_tuner.log 2>&1
echo "tuner exit=$?"
echo "--- 关键行（应出现 [tuner] 决策 + Avg bus，且无 SIGSEGV/Segmentation）---"
grep -iE "tuner|plugin|SIGSEGV|Segmentation|Avg bus|Out of bounds|Caught signal" /tmp/j37_tuner.log | head -20

echo ""
echo "===== [2] train_comm_sim：MPI_Bcast bootstrap ====="
cd "$WS/experiments"
hipcc -O2 --offload-arch=gfx936 -o /tmp/train_comm_sim train_comm_sim.cpp \
  $(mpicxx --showme:compile) $(mpicxx --showme:link) -L"$RCCL_HOME"/lib -lrccl 2>&1 | tail -5
if [ -x /tmp/train_comm_sim ]; then
  mpirun --allow-run-as-root -np 2 --map-by ppr:2:node \
    -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
    /tmp/train_comm_sim 2>&1 | tail -25
  echo "--- train_comm_sim exit=$? ---"
else
  echo "[FAIL] train_comm_sim 编译失败"
fi
echo "########## J37 done $(date) ##########"
