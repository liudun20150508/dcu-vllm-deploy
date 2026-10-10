#!/bin/bash
#SBATCH -J acc-j3-scal
#SBATCH -p hx1hdnormal
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j3-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j3-%j.err

# J3：4 节点 32 卡 —— 扩展性曲线（32 卡档）
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
AR=/public/software/app/rccl/tests/all_reduce_perf
EX="--allow-run-as-root --map-by ppr:8:node"
X="-x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN -x NCCL_IB_HCA -x NCCL_TOPO_FILE -x RCCL_NET_PLANE -x RCCL_PXN_GPU_BALANCE"
echo "########## J3 on $(scontrol show hostname $SLURM_NODELIST | tr '\n' ' ') $(date) ##########"
echo "===== RCCL AllReduce 32 卡（4 节点）====="
mpirun $EX -np 32 $X $AR -b 8 -e 256M -f 4 -g 1 -w 2 -n 3 2>&1 | tail -22
echo "===== MPI 32 进程跨 4 节点 ====="
mpicxx -O2 -o /tmp/mpi_bench $WS/mpi-lab/mpi_bench.cpp 2>&1 | tail -3
mpirun --allow-run-as-root -np 32 --map-by ppr:8:node /tmp/mpi_bench 2>&1 | tail -20
echo "########## J3 done $(date) ##########"
