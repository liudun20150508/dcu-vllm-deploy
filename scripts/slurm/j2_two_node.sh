#!/bin/bash
#SBATCH -J acc-j2-two
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j2-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j2-%j.err

# J2：2 节点 16 卡 —— ①基线(退化Socket) ②调优后 ③多网卡单变量对照 ④稳定性 ⑤MPI 跨节点
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
AR=/public/software/app/rccl/tests/all_reduce_perf
EX="--allow-run-as-root --map-by ppr:8:node"
X="-x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN -x NCCL_IB_HCA -x NCCL_TOPO_FILE -x RCCL_NET_PLANE -x RCCL_PXN_GPU_BALANCE -x NCCL_DEBUG"
echo "########## J2 on $(scontrol show hostname $SLURM_NODELIST | tr '\n' ' ') $(date) ##########"

echo "===== [1] 复现基线：不指定 NET 插件（应退化为 Socket / 低带宽）====="
NCCL_NET_PLUGIN= NCCL_DEBUG=WARN \
mpirun $EX -np 16 $X $AR -b 8 -e 256M -f 4 -g 1 -w 2 -n 3 2>&1 | tail -22

echo "===== [2] 调优后：加载 librccl-net-shca.so ====="
NCCL_NET_PLUGIN=$PLUGIN \
mpirun $EX -np 16 $X $AR -b 8 -e 256M -f 4 -g 1 -w 2 -n 3 2>&1 | tail -22

echo "===== [3] 多 RDMA 单变量对照（只改 NCCL_IB_HCA）====="
for hca in "shca_0" "shca_0,shca_1" "shca"; do
  echo "----- NCCL_IB_HCA=$hca -----"
  NCCL_IB_HCA=$hca NCCL_NET_PLUGIN=$PLUGIN \
  mpirun $EX -np 16 $X $AR -b 8 -e 256M -f 4 -g 1 -w 2 -n 3 2>&1 | tail -12
done
echo "----- 四卡 + 双平面 RCCL_NET_PLANE -----"
NCCL_IB_HCA=shca RCCL_NET_PLANE='shca_0,shca_3|shca_1,shca_2' NCCL_NET_PLUGIN=$PLUGIN \
mpirun $EX -np 16 $X $AR -b 8 -e 256M -f 4 -g 1 -w 2 -n 3 2>&1 | tail -12

echo "===== [4] 稳定性：128MB × 8 轮 ====="
NCCL_NET_PLUGIN=$PLUGIN \
mpirun $EX -np 16 $X $AR -b 128M -e 128M -f 1 -g 1 -w 1 -n 8 2>&1 | tail -16

echo "===== [5] MPI 跨节点 16 进程 ====="
mpicxx -O2 -o /tmp/mpi_bench $WS/mpi-lab/mpi_bench.cpp 2>&1 | tail -3
mpirun --allow-run-as-root -np 16 --map-by ppr:8:node /tmp/mpi_bench 2>&1 | tail -22

echo "########## J2 done $(date) ##########"
