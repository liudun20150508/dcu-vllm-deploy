#!/bin/bash
#SBATCH -J acc-j29-mpirocm
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j29-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j29-%j.err

# J29（优先 2/3 + 职责 2）：GPU-aware MPI —— MPI 直接对 DCU 显存做 AllReduce
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
export HSA_FORCE_FINE_GRAIN_PCIE=1
HOSTS=($(scontrol show hostname $SLURM_NODELIST))
echo "########## J29 on ${HOSTS[*]} ($(date)) ##########"

echo "=== 编译 ==="
cd $WS/mpi-lab
mpicxx -O2 -std=c++17 -Wall -Wextra -D__HIP_PLATFORM_AMD__ \
  -L$DTKROOT/hip/lib -o $WS/mpi-lab/mpi_rocm_allreduce mpi_rocm_allreduce.cpp -lamdhip64 2>&1 | head -6
ls -la $WS/mpi-lab/mpi_rocm_allreduce

echo
echo "=== 跑：默认（不带 GPU 加速组件）==="
mpirun --allow-run-as-root -np 2 --map-by ppr:1:node \
  -x LD_LIBRARY_PATH -x HSA_FORCE_FINE_GRAIN_PCIE \
  $WS/mpi-lab/mpi_rocm_allreduce 2>&1 | head -25

echo
echo "=== 再跑：显式开启 accelerator/rocp 组件 ==="
mpirun --allow-run-as-root -np 2 --map-by ppr:1:node \
  --mca pml ucx --mca accelerator rocm --mca opal_cuda_support 1 \
  -x LD_LIBRARY_PATH -x HSA_FORCE_FINE_GRAIN_PCIE \
  $WS/mpi-lab/mpi_rocm_allreduce 2>&1 | head -25
echo "########## J29 done $(date) ##########"
