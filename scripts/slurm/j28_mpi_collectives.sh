#!/bin/bash
#SBATCH -J acc-j28-mpicoll
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j28-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j28-%j.err

# J28（优先 4 + 职责 2）：MPI 手写集合通信算法 —— 跨节点 16 进程
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
HOSTS=($(scontrol show hostname $SLURM_NODELIST))
echo "########## J28 on ${HOSTS[*]} ($(date)) ##########"

echo "=== 编译（mpicxx -Wall -Wextra）==="
cd $WS/mpi-lab
mpicxx -O2 -std=c++17 -Wall -Wextra -o $WS/mpi-lab/mpi_collectives mpi_collectives.cpp 2>&1 | head -5
ls -la $WS/mpi-lab/mpi_collectives

echo
echo "=== 跨节点 16 进程（走 shca 网卡）==="
mpirun --allow-run-as-root -np 16 --map-by ppr:8:node $WS/mpi-lab/mpi_collectives 2>&1 | head -45

echo
echo "=== 顺带：MPI 的 GPU-aware 支持探测 ==="
cat > /tmp/mpi_gpu_probe.c <<'EOF'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int r; MPI_Comm_rank(MPI_COMM_WORLD, &r);
#ifdef MPIX_CUDA_AWARE_SUPPORT
    if (r == 0) printf("  MPIX_CUDA_AWARE_SUPPORT = %d\n", MPIX_CUDA_AWARE_SUPPORT);
#endif
    /* OpenMPI 运行时查询 */
    if (r == 0) {
        printf("  (OpenMPI 编译期 extensions 见 ompi_info：affinity,cuda,ftmpi,rocm)\n");
        printf("  运行期实际支持需 mpirun --mca pml ucx 或 opal_cuda 组件\n");
    }
    MPI_Finalize();
    return 0;
}
EOF
mpicc -O2 -o /tmp/mpi_gpu_probe /tmp/mpi_gpu_probe.c 2>&1 | head -3
mpirun --allow-run-as-root -np 2 /tmp/mpi_gpu_probe 2>&1 | head -5
echo "########## J28 done $(date) ##########"
