#!/bin/bash
#SBATCH -J acc-j32-dcu
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j32-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j32-%j.err

# J32（优先 3）：DCU(HIP) 编程 —— wave64 的坑 / 显存带宽 / 归约 kernel / P2P
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
echo "########## J32 on $(hostname) ($(date)) ##########"
echo "hipcc: $(command -v hipcc)  $(hipcc --version 2>&1 | head -1)"

cd $WS/experiments
echo
echo "=== 编译（--offload-arch=gfx936）==="
hipcc -O2 --offload-arch=gfx936 -o /tmp/dcu_kernel_lab dcu_kernel_lab.hip 2>&1 | tail -6
ls -la /tmp/dcu_kernel_lab 2>&1

echo
echo "=== 运行 ==="
/tmp/dcu_kernel_lab 2>&1 | head -50

echo
echo "=== 顺带：hipify 能把 CUDA 源码一键转 DCU 吗 ==="
cat > /tmp/cuda_demo.cu <<'EOF'
#include <cuda_runtime.h>
__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  float *a, *b, *c; cudaMalloc(&a, 1024); cudaMemset(a, 0, 1024);
  cudaDeviceSynchronize();
  return 0;
}
EOF
hipify-perl /tmp/cuda_demo.cu > /tmp/cuda_demo.hip 2>/dev/null || hipify /tmp/cuda_demo.cu > /tmp/cuda_demo.hip 2>/dev/null
echo "--- hipify 的输出（只看关键行）---"
grep -n -E "hip|include" /tmp/cuda_demo.hip 2>/dev/null | head -8
echo "########## J32 done $(date) ##########"
