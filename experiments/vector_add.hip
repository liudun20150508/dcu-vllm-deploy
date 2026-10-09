// vector_add.hip —— 最小 HIP kernel（海光 DCU 的 GPU 编程模型验证）
//
// 目的：确认 DTK 的工具链能编译并运行 kernel，顺便摸清海光编程模型和 CUDA 的差异。
//
// 编译运行：
//     source /opt/dtk-26.04/env.sh
//     hipcc -O2 -o vector_add vector_add.hip && ./vector_add
//
// 和 CUDA 的对应关系（HIP 是"照抄 CUDA 的 API"）：
//     核函数声明   __global__               → 一样
//     线程索引     threadIdx/blockIdx/...   → 一样
//     设备内存     cudaMalloc               → hipMalloc
//     内存拷贝     cudaMemcpy               → hipMemcpy
//     设备同步     cudaDeviceSynchronize    → hipDeviceSynchronize
//     设备属性     cudaGetDeviceProperties  → hipGetDeviceProperties
//     启动语法     <<<grid, block>>>        → hipLaunchKernelGGL(...)
//
// 所以"从 CUDA 迁到 HIP"在核函数层面基本是机械替换 —— 但设备属性、
// 互联拓扑、性能特征这些**架构层面的东西，两家完全不同**。

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
  return;  // hipcc 对 __global__ void 会误报 -Wreturn-type，显式 return 消掉
}

int main() {
  const int n = 1 << 20;                 // 1M 个元素
  const std::size_t bytes = n * sizeof(float);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  hipMalloc(&d_a, bytes);
  hipMalloc(&d_b, bytes);
  hipMalloc(&d_c, bytes);

  float* h = static_cast<float*>(std::malloc(bytes));
  for (int i = 0; i < n; ++i) h[i] = 1.0f;

  hipMemcpy(d_a, h, bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_b, h, bytes, hipMemcpyHostToDevice);

  // 启动配置：和 CUDA 一样是 (grid, block)
  hipLaunchKernelGGL(vector_add, dim3((n + 255) / 256), dim3(256), 0, 0, d_a, d_b, d_c, n);
  hipError_t err = hipDeviceSynchronize();

  hipMemcpy(h, d_c, bytes, hipMemcpyDeviceToHost);

  // ---- 打印设备和结果 ----
  int dev = 0;
  hipGetDevice(&dev);
  hipDeviceProp_t prop{};
  hipGetDeviceProperties(&prop, dev);

  std::printf("=== HIP kernel 执行结果 ===\n");
  std::printf("hipDeviceSynchronize: %s\n", hipGetErrorString(err));
  std::printf("device        : %s\n", prop.name);
  std::printf("gcnArchName   : %s\n", prop.gcnArchName);      // ← 海光/AMD 特有的架构名
  std::printf("multiProcessor: %d\n", prop.multiProcessorCount);
  std::printf("warpSize      : %d\n", prop.warpSize);
  std::printf("totalGlobalMem: %.1f GB\n", prop.totalGlobalMem / 1073741824.0);
  std::printf("c[0]          = %.1f (期望 2.0)\n", h[0]);
  std::printf("c[n-1]        = %.1f (期望 2.0)\n", h[n - 1]);

  bool ok = (h[0] == 2.0f && h[n - 1] == 2.0f);
  std::printf("%s\n", ok ? "[PASS] 1M 元素向量加法正确" : "[FAIL] 结果不对");

  hipFree(d_a); hipFree(d_b); hipFree(d_c);
  std::free(h);
  return ok ? 0 : 1;
}
