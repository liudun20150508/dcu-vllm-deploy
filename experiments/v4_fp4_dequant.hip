// v4_fp4_dequant.hip —— DeepSeek-V4 的 FP4 专家权重反量化 kernel（HIP 版）
//
// 【为什么写这个】
//   V4 的专家权重用 FP4（config 里 expert_dtype: fp4），格式是 float4_e2m1fn，
//   块大小 fp4_block_size=32，每个块一个 fp32 缩放系数。
//   反量化 = 把 4-bit 整数按 e2m1fn 语义还原成浮点，再乘块缩放。
//   这是 V4 官方推理代码 kernel.py 里 fp4_gemm 的第一步（我读源码读到的）。
//
// 【e2m1fn 格式】（1 符号 + 2 指数 + 1 尾数，无 inf/nan）
//   bit3=符号 s，bit[2:1]=指数 e，bit0=尾数 m
//     e=0: mag = 0.5*m            → 0 或 0.5
//     e=1: mag = 1.0 + 0.5*m      → 1.0 或 1.5
//     e=2: mag = 2.0 + 1.0*m      → 2.0 或 3.0
//     e=3: mag = 4.0 + 2.0*m      → 4.0 或 6.0   （fn=有限，无 inf/nan）
//
// 编译（曙光上，DTK 环境）：
//   module load compiler/dtk/26.04
//   hipcc -O2 -o v4_fp4_dequant v4_fp4_dequant.hip

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>

#define CHK(call) do { hipError_t e = (call); \
  if (e != hipSuccess) { std::fprintf(stderr, "[hip] %s: %s\n", #call, hipGetErrorString(e)); std::exit(1); } } while (0)

constexpr int FP4_BLOCK = 32;   // V4 的 fp4_block_size

// ---- CPU 参考：e2m1fn 反量化（用于对照验证）----
__host__ __device__ inline float fp4_to_float(uint8_t x) {
    int s = (x >> 3) & 1;
    int e = (x >> 1) & 0x3;
    int m = x & 0x1;
    float mag;
    if (e == 0)      mag = 0.5f * m;
    else if (e == 1) mag = 1.0f + 0.5f * m;
    else if (e == 2) mag = 2.0f + 1.0f * m;
    else             mag = 4.0f + 2.0f * m;   // e=3，fn 格式无 inf
    return (s ? -mag : mag);
}

// ---- GPU kernel：反量化 FP4 权重（packed 2 个/字节）----
// packed: 每字节存 2 个 fp4 值（低 4 位在前）
// scales: 每 FP4_BLOCK 个值一个 fp32 缩放
// out:    反量化后的 fp32（或 bf16）结果
__global__ void fp4_dequant_kernel(const uint8_t* __restrict__ packed,
                                   const float* __restrict__ scales,
                                   float* __restrict__ out,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // 第 i 个 fp4 值在字节 i/2 里，i 偶数取低 4 位、奇数取高 4 位
    uint8_t byte = packed[i >> 1];
    uint8_t x = (i & 1) ? (byte >> 4) : (byte & 0xF);
    float scale = scales[i / FP4_BLOCK];
    out[i] = fp4_to_float(x) * scale;
}

int main() {
    constexpr int N = 4096;                 // 4096 个 fp4 值
    constexpr int NBYTES = N / 2;           // packed 字节数
    constexpr int NBLOCKS = N / FP4_BLOCK;  // 块数 = 缩放系数个数

    // 造一点测试数据（固定 seed，可复现）
    srand(42);
    uint8_t* h_packed = new uint8_t[NBYTES];
    float*   h_scales = new float[NBLOCKS];
    for (int b = 0; b < NBYTES; ++b) h_packed[b] = (uint8_t)(rand() & 0xFF);  // 每字节 2 个随机 fp4
    for (int b = 0; b < NBLOCKS; ++b) h_scales[b] = (rand() % 1000) / 100.0f; // 随机缩放

    // 设备内存
    uint8_t *d_packed; float *d_scales, *d_out;
    CHK(hipMalloc(&d_packed, NBYTES));
    CHK(hipMalloc(&d_scales, NBLOCKS * sizeof(float)));
    CHK(hipMalloc(&d_out, N * sizeof(float)));
    CHK(hipMemcpy(d_packed, h_packed, NBYTES, hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_scales, h_scales, NBLOCKS * sizeof(float), hipMemcpyHostToDevice));

    // 启动 kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(fp4_dequant_kernel, dim3(blocks), dim3(threads), 0, 0,
                       d_packed, d_scales, d_out, N);
    CHK(hipDeviceSynchronize());

    // 取回结果
    float* h_out = new float[N];
    CHK(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));

    // ---- CPU 对照验证 ----
    int mismatch = 0;
    for (int i = 0; i < N; ++i) {
        uint8_t byte = h_packed[i >> 1];
        uint8_t x = (i & 1) ? (byte >> 4) : (byte & 0xF);
        float expect = fp4_to_float(x) * h_scales[i / FP4_BLOCK];
        if (std::fabs(h_out[i] - expect) > 1e-6f) {
            if (mismatch < 5) std::printf("  mismatch @%d: gpu=%f cpu=%f\n", i, h_out[i], expect);
            ++mismatch;
        }
    }

    if (mismatch == 0) {
        std::printf("[PASS] FP4 反量化 kernel：%d 个值 GPU==CPU 全部一致\n", N);
        std::printf("       样例（前 4 个）：%f %f %f %f\n",
                    h_out[0], h_out[1], h_out[2], h_out[3]);
    } else {
        std::printf("[FAIL] %d 个不一致\n", mismatch);
        return 1;
    }

    // 清理
    hipFree(d_packed); hipFree(d_scales); hipFree(d_out);
    delete[] h_packed; delete[] h_scales; delete[] h_out;
    return 0;
}
