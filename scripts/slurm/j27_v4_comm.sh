#!/bin/bash
#SBATCH -J acc-j27-v4comm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j27-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j27-%j.err

# J27（优先 7）：按 DeepSeek-V4-Flash / V4-Pro 真实参数 —— 并行策略的通信场景实测
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
export MASTER_PORT=29527
echo "########## J27 on ${HOSTS[*]} ($(date)) ##########"

mpirun --allow-run-as-root -np 8 --map-by ppr:8:node \
  -x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN \
  -x MASTER_ADDR -x MASTER_PORT -x NCCL_TOPO_FILE -x RCCL_NET_PLANE \
  -x OMPI_COMM_WORLD_LOCAL_RANK -x SLURM_LOCALID -x HIP_VISIBLE_DEVICES \
  $PY $WS/experiments/v4_comm_lab.py 2>&1 \
  | grep -a -v -E "ProcessGroupNCCL|watchdog|heart beat|Warning|warn|^\[|numpy|NumPy|NCCLUtils|Aborting|frame #" | tail -60
echo "########## J27 done $(date) ##########"
