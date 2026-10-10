#!/bin/bash
#SBATCH -J acc-j7-scal8
#SBATCH -p hx1hdnormal
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j7-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j7-%j.err

# J7：8 节点 64 卡 —— 扩展性曲线最高一档
set -u
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
echo "########## J7 nodes: $(scontrol show hostname $SLURM_NODELIST | wc -l) $(date) ##########"
mpirun --allow-run-as-root -np 64 --map-by ppr:8:node \
  -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
  -x NCCL_TOPO_FILE -x RCCL_NET_PLANE -x RCCL_PXN_GPU_BALANCE \
  /public/software/app/rccl/tests/all_reduce_perf -b 8 -e 256M -f 4 -g 1 -w 2 -n 3 2>&1 | tail -22
echo "########## J7 done $(date) ##########"
