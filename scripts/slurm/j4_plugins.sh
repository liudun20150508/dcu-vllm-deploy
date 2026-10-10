#!/bin/bash
#SBATCH -J acc-j4-plug
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:2
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j4-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j4-%j.err

# J4：插件三连 —— ①我写的 tuner 插件能否被真 RCCL 挂载
#                        ②我写的 NET 插件(v12) 在真 RCCL(v8) 上会怎样
#                        ③训练通信模拟器（已知待修项）在真机上的表现
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
AR=/public/software/app/rccl/tests/all_reduce_perf
EX="--allow-run-as-root --map-by ppr:2:node"
X="-x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN -x NCCL_TUNER_PLUGIN -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS"
echo "########## J4 on $(hostname) $(date) ##########"

echo "===== [1] 我写的 tuner plugin 挂载验证 ====="
cp -f $WS/nccl-tuner-plugin/libnccl-tuner-mycard.so /tmp/librccl-tuner.so
nm -D --defined-only /tmp/librccl-tuner.so | grep -i tunerplugin || true
NCCL_TUNER_PLUGIN=/tmp/librccl-tuner.so NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING,INIT \
mpirun $EX -np 2 $X $AR -b 8M -e 8M -f 1 -g 1 -w 1 -n 2 2>&1 | grep -i -E "tuner|plugin|busbw|Avg bus" | head -20
echo "--- 对照：不指定 tuner 插件 ---"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING \
mpirun $EX -np 2 -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
$AR -b 8M -e 8M -f 1 -g 1 -w 1 -n 2 2>&1 | grep -i -E "tuner|plugin" | head -10

echo "===== [2] NET plugin v12 在 RCCL(v8) 上的加载结果 ====="
cp -f /tmp/libnccl-net.so /tmp/librccl-net-mycard.so 2>/dev/null || echo "(v12 .so 不在 /tmp，重新编译)"
if [ ! -f /tmp/librccl-net-mycard.so ]; then
  g++ -std=c++17 -O2 -fPIC -shared -fpermissive -I/tmp/nccl-shim \
      -I$WS/refs/nccl/src/include/plugin -I$WS/refs/nccl/src/include \
      $WS/nccl-net-plugin/mycard_net.cpp -o /tmp/librccl-net-mycard.so -lpthread 2>&1 | tail -3
fi
nm -D --defined-only /tmp/librccl-net-mycard.so | grep -i netplugin
NCCL_NET_PLUGIN=/tmp/librccl-net-mycard.so NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET,INIT \
mpirun $EX -np 2 -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
$AR -b 8M -e 8M -f 1 -g 1 -w 1 -n 2 2>&1 | grep -i -E "plugin|NET/|Avg bus|Out of bounds" | head -15

echo "===== [3] 训练通信模拟器 train_comm_sim（已知待修项）====="
cd $WS/experiments
hipcc -O2 --offload-arch=gfx936 -o /tmp/train_comm_sim train_comm_sim.cpp -L$RCCL_HOME/lib -lrccl 2>&1 | tail -8
if [ -x /tmp/train_comm_sim ]; then
  echo "--- 编译成功，运行（2 卡）---"
  mpirun $EX -np 2 -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
    /tmp/train_comm_sim 2>&1 | tail -30
  echo "--- 退出码 $? ---"
else
  echo "[FAIL] train_comm_sim 编译失败"
fi

echo "########## J4 done $(date) ##########"
