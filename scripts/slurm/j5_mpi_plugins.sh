#!/bin/bash
#SBATCH -J acc-j5-mpi
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j5-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j5-%j.err

# J5（修正版，替代 J4 的失败段）：
#   ① MPI 跨节点 16 进程（module purge 后再 load，避免 mpi/hpcx 冲突）
#   ② 我写的 tuner 插件在真 RCCL 上的挂载
#   ③ 我写的 NET 插件 v12 在 RCCL(v8) 上的加载结果
#   ④ train_comm_sim（已知待修项）真机表现
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
echo "mpirun = $(command -v mpirun)"
echo "mpicxx = $(command -v mpicxx)"
export HSA_FORCE_FINE_GRAIN_PCIE=1
PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
AR=/public/software/app/rccl/tests/all_reduce_perf
echo "########## J5 on $(scontrol show hostname $SLURM_NODELIST | tr '\n' ' ') $(date) ##########"

echo "===== [1] MPI 跨节点 16 进程 ====="
mpicxx -O2 -o /tmp/mpi_bench $WS/mpi-lab/mpi_bench.cpp 2>&1 | tail -3
mpirun --allow-run-as-root -np 16 --map-by ppr:8:node /tmp/mpi_bench 2>&1 | tail -22

echo "===== [2] tuner plugin 挂载（NCCL_TUNER_PLUGIN）====="
cp -f $WS/nccl-tuner-plugin/libnccl-tuner-mycard.so /tmp/librccl-tuner.so
nm -D --defined-only /tmp/librccl-tuner.so | grep -i tunerplugin
NCCL_NET_PLUGIN=$PLUGIN NCCL_TUNER_PLUGIN=/tmp/librccl-tuner.so \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING,INIT \
mpirun --allow-run-as-root -np 2 --map-by ppr:2:node \
  -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
  -x NCCL_TUNER_PLUGIN -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  $AR -b 8M -e 8M -f 1 -g 1 -w 1 -n 2 2>&1 | grep -i -E "tuner|plugin|Avg bus" | head -12
echo "--- 对照：不指定 tuner 插件 ---"
NCCL_NET_PLUGIN=$PLUGIN NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING \
mpirun --allow-run-as-root -np 2 --map-by ppr:2:node \
  -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  $AR -b 8M -e 8M -f 1 -g 1 -w 1 -n 2 2>&1 | grep -i -E "tuner|plugin" | head -8

echo "===== [3] NET plugin v12 在 RCCL(v8) 上的加载结果 ====="
SHIM=/tmp/nccl-shim-j5; mkdir -p $SHIM
cat > $SHIM/nccl.h <<'EOF'
#ifndef NCCL_H_
#define NCCL_H_
#include <stdint.h>
#include <stddef.h>
typedef enum { ncclSuccess=0, ncclUnhandledCudaError=1, ncclSystemError=2,
  ncclInternalError=3, ncclInvalidArgument=4, ncclInvalidUsage=5,
  ncclRemoteError=6, ncclInProgress=7, ncclNumResults=8 } ncclResult_t;
typedef enum { ncclInt8=0, ncclChar=0, ncclUint8=1, ncclInt32=2, ncclInt=2,
  ncclUint32=3, ncclInt64=4, ncclUint64=5, ncclFloat16=6, ncclHalf=6,
  ncclFloat32=7, ncclFloat=7, ncclFloat64=8, ncclDouble=8, ncclBfloat16=9,
  ncclNumTypes=10 } ncclDataType_t;
typedef enum { ncclSum=0, ncclProd=1, ncclMax=2, ncclMin=3, ncclAvg=4, ncclNumOps=5 } ncclRedOp_t;
#endif
EOF
g++ -std=c++17 -O2 -fPIC -shared -fpermissive -I$SHIM \
    -I$WS/refs/nccl/src/include/plugin -I$WS/refs/nccl/src/include \
    $WS/nccl-net-plugin/mycard_net.cpp -o /tmp/librccl-net-mycard.so -lpthread 2>&1 | tail -3
ls -la /tmp/librccl-net-mycard.so 2>&1
nm -D --defined-only /tmp/librccl-net-mycard.so 2>/dev/null | grep -i netplugin
NCCL_NET_PLUGIN=/tmp/librccl-net-mycard.so NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET,INIT \
mpirun --allow-run-as-root -np 2 --map-by ppr:2:node \
  -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  $AR -b 8M -e 8M -f 1 -g 1 -w 1 -n 2 2>&1 | grep -i -E "plugin|NET/|Avg bus|Out of bounds|symbol" | head -15

echo "===== [4] train_comm_sim（已修：MPI_Bcast 替代文件 bootstrap）====="
cd $WS/experiments
hipcc -O2 --offload-arch=gfx936 -o /tmp/train_comm_sim train_comm_sim.cpp \
  $(mpicxx --showme:compile) $(mpicxx --showme:link) -L$RCCL_HOME/lib -lrccl 2>&1 | tail -8
if [ -x /tmp/train_comm_sim ]; then
  mpirun --allow-run-as-root -np 2 --map-by ppr:2:node \
    -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
    /tmp/train_comm_sim 2>&1 | tail -35
  echo "--- exit=$? ---"
else
  echo "[FAIL] train_comm_sim 编译失败"
fi

echo "########## J5 done $(date) ##########"
