// dcu_kernel_lab.hip —— DCU(HIP) 编程实验台：说清"为什么这不是 CUDA"
//
// 【这份代码要回答什么】
//   JD 优先条件 3 写的是「熟悉 CUDA、GPU 编程模型及 GPU 通信」。
//   但 **海光 DCU 跑的不是 CUDA —— 是 HIP（DTK/ROCm 栈）**。
//   很多人以为"HIP 就是 CUDA 改个名字"，**API 层面确实如此，但架构层面完全不同**。
//   这份实验就把那些"改了名字也搬不过去"的地方跑出来：
//
//     ① wave64：DCU 的 warp 是 **64 个 lane**，不是 32 —— 写死 32 的归约会静默算错
//     ② 显存带宽：摸 gfx938 的实际带宽（平台基线上是 1.35 TB/s）
//     ③ 归约 kernel：这是集合通信的 GPU 端基础（NCCL/RCCL 的内核就干这个）
//     ④ P2P：两块 DCU 之间直接搬数据，不走主机
//
// 编译运行（在计算节点）：
//     module load compiler/dtk/26.04
//     hipcc -O2 --offload-arch=gfx938 -o dcu_kernel_lab dcu_kernel_lab.hip && ./dcu_kernel_lab

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_OK(expr, what)                                                    \
  do {                                                                        \
    hipError_t _rc = (expr);                                                  \
    if (_rc != hipSuccess) {                                                  \
      std::fprintf(stderr, "[hip] %s -> %s\n", what, hipGetErrorString(_rc));  \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// ============================================================ ① 波前宽度的坑
// CUDA 里"一个 warp 32 个 lane"是常识，很多人会这么写归约：
__device__ float warp_reduce_cuda_style(float v) {
  // ❌ 写死 32 —— 在 DCU 上只归约了半个 wave，结果是错的
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
  return v;
}

// DCU 正确写法：用 warpSize（gfx9 上是 64）而不是字面量
__device__ float warp_reduce_portable(float v) {
  // ✅ 从 warpSize/2 开始 —— 这行代码在 NVIDIA(32) 和 DCU(64) 上都对
  for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down(v, off);
  return v;
}

__global__ void warp_sum_kernel(const float* in, float* out, int n, int mode) {
  // 每个 wave 负责一段，长度恰好是一个 warpSize
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % warpSize;
  float v = (tid < n) ? in[tid] : 0.0f;
  float s = (mode == 0) ? warp_reduce_cuda_style(v) : warp_reduce_portable(v);
  if (lane == 0 && (blockIdx.x * blockDim.x / warpSize) < 4) {
    out[blockIdx.x * blockDim.x / warpSize] = s;  // 前 4 个 wave 的和，便于观察
  }
  return;
}

// ============================================================ ② 显存带宽
__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                            std::size_t n4) {
  std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) out[i] = in[i];
  return;
}

__global__ void triad_kernel(float4* a, const float4* b, const float4* c, std::size_t n4) {
  std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 x = b[i], y = c[i];
    float4 r{x.x + 2.0f * y.x, x.y + 2.0f * y.y, x.z + 2.0f * y.z, x.w + 2.0f * y.w};
    a[i] = r;
  }
  return;
}

// ============================================================ ③ 归约（集合通信的 GPU 端基础）
// 两级归约：wave 内用 shuffle，wave 间用 shared memory
__global__ void block_reduce_kernel(const float* in, float* out, int n) {
  extern __shared__ float smem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // ★ grid-stride：每个线程负责多个元素 —— 否则 grid 开小了会**静默只算一部分**
  //   （第一版就是这里错了：4096 block×256 线程 = 1M < 16M，结果正好差 16 倍）
  float v = 0.0f;
  for (int i = tid; i < n; i += gridDim.x * blockDim.x) v += in[i];
  v = warp_reduce_portable(v);

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  if (wid == 0) {
    int nwarps = blockDim.x / warpSize;
    v = (lane < nwarps) ? smem[lane] : 0.0f;
    v = warp_reduce_portable(v);
    if (lane == 0) atomicAdd(out, v);
  }
  return;
}

// ============================================================ ④ P2P：两块 DCU 直接搬
__global__ void peer_read_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                 std::size_t n) {
  std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];  // src 可以是指向**另一块卡**的指针（P2P）
}

int main() {
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0), "get prop");
  int ndev = 0;
  HIP_OK(hipGetDeviceCount(&ndev), "count");

  std::printf("=== DCU(HIP) 编程实验台 ===\n");
  std::printf("设备      : %s\n", prop.name);
  std::printf("架构      : %s\n", prop.gcnArchName);
  std::printf("CU 数     : %d\n", prop.multiProcessorCount);
  std::printf("★ warpSize: %d   ← CUDA 是 32，这里是 64（wave64）\n", prop.warpSize);
  std::printf("显存      : %.1f GB\n", prop.totalGlobalMem / 1024.0 / 1024 / 1024);
  std::printf("LDS/block : %zu KB\n", prop.sharedMemPerBlock / 1024);
  std::printf("卡数      : %d\n\n", ndev);

  // ---------------- ① wave64 的坑 ----------------
  {
    const int n = 128;
    std::vector<float> h(n, 0.0f);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    float *d_in = nullptr, *d_out = nullptr;
    HIP_OK(hipMalloc(&d_in, n * sizeof(float)), "malloc in");
    HIP_OK(hipMalloc(&d_out, 4 * sizeof(float)), "malloc out");
    HIP_OK(hipMemcpy(d_in, h.data(), n * sizeof(float), hipMemcpyHostToDevice), "H2D");

    float res[4] = {0, 0, 0, 0};
    hipLaunchKernelGGL(warp_sum_kernel, dim3(n / 64), dim3(64), 0, 0, d_in, d_out, n, 0);
    HIP_OK(hipDeviceSynchronize(), "sync");
    HIP_OK(hipMemcpy(res, d_out, sizeof(res), hipMemcpyDeviceToHost), "D2H");
    std::printf("① wave64 验证（每个 wave 64 个 1.0f，正确的和应是 64）\n");
    std::printf("   按 CUDA 习惯写死 32 : 前 4 个 wave 的和 = %.0f %.0f %.0f %.0f  ← 只有一半\n",
                res[0], res[1], res[2], res[3]);

    hipLaunchKernelGGL(warp_sum_kernel, dim3(n / 64), dim3(64), 0, 0, d_in, d_out, n, 1);
    HIP_OK(hipDeviceSynchronize(), "sync");
    HIP_OK(hipMemcpy(res, d_out, sizeof(res), hipMemcpyDeviceToHost), "D2H");
    std::printf("   用 warpSize/2 写    : 前 4 个 wave 的和 = %.0f %.0f %.0f %.0f  ← 正确\n\n",
                res[0], res[1], res[2], res[3]);
    hipFree(d_in); hipFree(d_out);
  }

  // ---------------- ② 显存带宽 ----------------
  {
    const std::size_t bytes = 1024ull * 1024 * 1024;  // 1 GB
    const std::size_t n4 = bytes / sizeof(float4);
    float4 *a = nullptr, *b = nullptr, *c = nullptr;
    HIP_OK(hipMalloc(&a, bytes), "malloc a");
    HIP_OK(hipMalloc(&b, bytes), "malloc b");
    HIP_OK(hipMalloc(&c, bytes), "malloc c");
    HIP_OK(hipMemset(a, 0, bytes), "memset");

    int blocks = prop.multiProcessorCount * 32;
    float ms = 0;
    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0), "ev"); HIP_OK(hipEventCreate(&t1), "ev");
    const int iters = 10;

    HIP_OK(hipEventRecord(t0), "rec");
    for (int i = 0; i < iters; ++i)
      hipLaunchKernelGGL(copy_kernel, dim3(blocks), dim3(256), 0, 0, b, a, n4);
    HIP_OK(hipEventRecord(t1), "rec");
    HIP_OK(hipEventSynchronize(t1), "sync");
    HIP_OK(hipEventElapsedTime(&ms, t0, t1), "elapsed");
    // copy：读 1GB + 写 1GB
    std::printf("② 显存带宽（1 GB buffer，grid=%d×256）\n", blocks);
    std::printf("   copy  : %.2f ms → %.1f GB/s（读+写各 1GB）\n", ms / iters,
                2.0 * bytes / (ms / iters / 1e3) / 1e9);

    HIP_OK(hipEventRecord(t0), "rec");
    for (int i = 0; i < iters; ++i)
      hipLaunchKernelGGL(triad_kernel, dim3(blocks), dim3(256), 0, 0, a, b, c, n4);
    HIP_OK(hipEventRecord(t1), "rec");
    HIP_OK(hipEventSynchronize(t1), "sync");
    HIP_OK(hipEventElapsedTime(&ms, t0, t1), "elapsed");
    std::printf("   triad : %.2f ms → %.1f GB/s（读 2GB + 写 1GB）\n", ms / iters,
                3.0 * bytes / (ms / iters / 1e3) / 1e9);
    std::printf("   （平台自查基线 dcu_stream Copy = 1350000 MB/s = 1350 GB/s）\n\n");
    hipFree(a); hipFree(b); hipFree(c);
    hipEventDestroy(t0); hipEventDestroy(t1);
  }

  // ---------------- ③ 归约 kernel（NCCL 内核干的就是这个）----------------
  {
    const int n = 1 << 24;  // 16M 个 float
    const std::size_t bytes = (std::size_t)n * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;
    HIP_OK(hipMalloc(&d_in, bytes), "malloc");
    HIP_OK(hipMalloc(&d_out, sizeof(float)), "malloc");

    // 构造可解析的数据：每个元素都是 1.0 → 和 = n
    std::vector<float> h(n, 1.0f);
    HIP_OK(hipMemcpy(d_in, h.data(), bytes, hipMemcpyHostToDevice), "H2D");
    HIP_OK(hipMemset(d_out, 0, sizeof(float)), "memset");

    const int thread = 256;                 // 4 个 wave/block
    const int block = 4096;
    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0), "ev"); HIP_OK(hipEventCreate(&t1), "ev");
    HIP_OK(hipEventRecord(t0), "rec");
    for (int i = 0; i < 20; ++i) {
      HIP_OK(hipMemset(d_out, 0, sizeof(float)), "memset");
      hipLaunchKernelGGL(block_reduce_kernel, dim3(block), dim3(thread),
                         (thread / 64) * sizeof(float), 0, d_in, d_out, n);
    }
    HIP_OK(hipEventRecord(t1), "rec");
    HIP_OK(hipEventSynchronize(t1), "sync");
    float ms = 0;
    HIP_OK(hipEventElapsedTime(&ms, t0, t1), "elapsed");

    float got = 0;
    HIP_OK(hipMemcpy(&got, d_out, sizeof(float), hipMemcpyDeviceToHost), "D2H");
    std::printf("③ 归约 kernel（16M 个 1.0f，两级规约：wave shuffle + smem）\n");
    std::printf("   结果 = %.0f  （期望 %.0f）%s\n", got, (float)n,
                std::fabs(got - (float)n) < 1.0f ? "  ✓" : "  ✗");
    std::printf("   单次 %.3f ms → %.1f GB/s\n\n", ms / 20, bytes / (ms / 20 / 1e3) / 1e9);
    hipFree(d_in); hipFree(d_out);
    hipEventDestroy(t0); hipEventDestroy(t1);
  }

  // ---------------- ④ P2P：两块 DCU 直接搬 ----------------
  {
    std::printf("④ P2P（两块 DCU 直接互访，不走主机内存）\n");
    if (ndev < 2) {
      std::printf("   只有 %d 块卡，跳过（需要 --gres=dcu:2）\n", ndev);
    } else {
      int can = 0;
      HIP_OK(hipDeviceCanAccessPeer(&can, 0, 1), "query p2p");
      std::printf("   hipDeviceCanAccessPeer(0,1) = %d\n", can);
      if (can) {
        const std::size_t bytes = 256ull * 1024 * 1024;
        float* d0 = nullptr;
        float* d1 = nullptr;
        HIP_OK(hipSetDevice(0), "set0");
        HIP_OK(hipDeviceEnablePeerAccess(1, 0), "enable p2p 0->1");
        HIP_OK(hipMalloc(&d0, bytes), "malloc d0");
        HIP_OK(hipSetDevice(1), "set1");
        HIP_OK(hipMalloc(&d1, bytes), "malloc d1");
        HIP_OK(hipMemset(d1, 1, bytes), "memset d1");
        HIP_OK(hipDeviceEnablePeerAccess(0, 0), "enable p2p 1->0");

        // 用 hipMemcpyPeer（API 内部处理两个设备上下文，比手搓 event+kernel 稳）
        HIP_OK(hipSetDevice(0), "set0");
        hipEvent_t t0 = nullptr, t1 = nullptr;
        HIP_OK(hipEventCreate(&t0), "ev");
        HIP_OK(hipEventCreate(&t1), "ev");
        const int iters = 5;
        HIP_OK(hipEventRecord(t0, nullptr), "rec");
        for (int i = 0; i < iters; ++i) {
          HIP_OK(hipMemcpyPeer(d0, 0, d1, 1, bytes), "memcpyPeer");  // D1 → D0，直连
        }
        HIP_OK(hipEventRecord(t1, nullptr), "rec");
        HIP_OK(hipEventSynchronize(t1), "sync");
        float ms = 0;
        HIP_OK(hipEventElapsedTime(&ms, t0, t1), "elapsed");
        std::printf("   hipMemcpyPeer D1→D0 256 MB：%.3f ms → %.2f GB/s\n", ms / iters,
                    bytes / (ms / iters / 1e3) / 1e9);

        // 再试 kernel 直接读对端显存（需要两边都 enable peer access）
        hipEventDestroy(t0); hipEventDestroy(t1);
        HIP_OK(hipEventCreate(&t0), "ev");
        HIP_OK(hipEventCreate(&t1), "ev");
        const std::size_t nf = bytes / sizeof(float);
        const int blocks = prop.multiProcessorCount * 16;
        HIP_OK(hipEventRecord(t0, nullptr), "rec");
        for (int i = 0; i < iters; ++i) {
          hipLaunchKernelGGL(peer_read_kernel, dim3(blocks), dim3(256), 0, nullptr, d1, d0, nf);
        }
        HIP_OK(hipEventRecord(t1, nullptr), "rec");
        HIP_OK(hipEventSynchronize(t1), "sync");
        HIP_OK(hipEventElapsedTime(&ms, t0, t1), "elapsed");
        std::printf("   kernel 直接读对端显存 256 MB：%.3f ms → %.2f GB/s\n", ms / iters,
                    bytes / (ms / iters / 1e3) / 1e9);

        hipFree(d0);
        HIP_OK(hipSetDevice(1), "set1");
        hipFree(d1);
        hipEventDestroy(t0); hipEventDestroy(t1);
      }
    }
  }

  std::printf("\n【小结：DCU 编程与 CUDA 的差异】\n");
  std::printf("  · API 层几乎一一对应（hipMalloc/hipMemcpy/<<<>>> 都能用）→ 移植很机械\n");
  std::printf("  · 但**架构层不同**：warpSize=64（wave64）、gcnArchName=gfx9xx、\n");
  std::printf("    LDS 大小、P2P 能力、内存层级都和 NVIDIA 不一样 —— \n");
  std::printf("    **写死 32 的代码会静默算错**，这就是'不是 CUDA 编程'最实际的一处\n");
  return 0;
}
