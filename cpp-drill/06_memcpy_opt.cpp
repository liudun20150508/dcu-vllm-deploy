// 06_memcpy_opt.cpp —— 对齐优化的内存拷贝
//
// 【为什么这个岗会考】
//   通信库的数据搬运路径（pack/unpack、GPU 与 host 之间的 staging）都在拷内存；
//   一次 allreduce 要搬 GB 级数据，拷贝效率直接影响 busbw。
//
// 【考点】
//   ① 为什么按机器字（8 字节）拷贝比逐字节快？—— 逐字节每次都要做地址计算 + 循环判断；
//      按字拷贝是"一次 load + 一次 store"搬 8 字节
//   ② 为什么还有"头部/尾部不对齐"的残余要单独处理？—— 目标地址可能不是 8 字节对齐的，
//      直接按字写会越界或者触发非对齐访问
//   ③ 能不能更快？—— SIMD（AVX2/AVX512 一次 32/64 字节）、非临时存储（streaming store）、
//      甚至 DMA/GPU kernel
//   ④ 什么情况下 memcpy 反而不该手写？—— 大块数据应该让 libc 的 SIMD 版干，或者直接 DMA
//
// 【易错点】
//   - 按字拷贝但没处理非对齐 → UBSan 报错 / ARM 上直接崩
//   - 越界（把尾部多写了）
//   - 忘记处理重叠区域（memmove vs memcpy 的区别）
//
// 编译： g++ -std=c++17 -Wall -Wextra -O2 -o 06 06_memcpy_opt.cpp

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace drill {

// 朴素版：逐字节（作为正确性基准 + 性能对照）
void* memcpy_naive(void* dst, const void* src, std::size_t n) {
  char* d = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);
  for (std::size_t i = 0; i < n; ++i) d[i] = s[i];
  return dst;
}

// 优化版：头对齐 → 按 8 字节搬 → 尾残余
void* memcpy_aligned(void* dst, const void* src, std::size_t n) {
  auto* d = static_cast<unsigned char*>(dst);
  const auto* s = static_cast<const unsigned char*>(src);

  // ① 头部：先把 dst 凑到 8 字节对齐（src 可能要非对齐读，但那是字节读，安全）
  while (n > 0 && (reinterpret_cast<std::uintptr_t>(d) & 7) != 0) {
    *d++ = *s++;
    --n;
  }
  // ② 主体：按 8 字节搬
  std::size_t words = n / 8;
  if (words > 0) {
    // 这里用 memcpy 表达"取 8 字节"，避免严格别名（strict aliasing）问题，
    // 编译器会把它优化成一条 load/store 指令。
    std::uint64_t tmp;
    for (std::size_t i = 0; i < words; ++i) {
      std::memcpy(&tmp, s, 8);
      std::memcpy(d, &tmp, 8);
      s += 8;
      d += 8;
    }
    n -= words * 8;
  }
  // ③ 尾部残余
  while (n > 0) {
    *d++ = *s++;
    --n;
  }
  return dst;
}

}  // namespace drill

// ============================ 测试 ============================

namespace {

std::vector<unsigned char> make_data(std::size_t n) {
  std::vector<unsigned char> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<unsigned char>((i * 31 + 7) & 0xff);
  return v;
}

void test_correctness_all_sizes() {
  // 覆盖各种长度 × 各种源/目的对齐组合
  for (std::size_t n = 0; n <= 130; ++n) {
    auto src = make_data(n + 16);
    for (std::size_t off_s = 0; off_s < 8; ++off_s) {
      for (std::size_t off_d = 0; off_d < 8; ++off_d) {
        std::vector<unsigned char> dst(n + 16, 0xAB);
        drill::memcpy_aligned(dst.data() + off_d, src.data() + off_s, n);
        assert(std::memcmp(dst.data() + off_d, src.data() + off_s, n) == 0);
        // 越界检查：前后哨兵不能被改
        if (off_d > 0) assert(dst[off_d - 1] == 0xAB);
        assert(dst[off_d + n] == 0xAB);
      }
    }
  }
  std::printf("  [ok] 0~130 字节 × 8×8 种源/目的对齐组合，全部正确且不越界\n");
}

void test_large() {
  const std::size_t n = 1 << 20;
  auto a = make_data(n);
  std::vector<unsigned char> b(n, 0);
  drill::memcpy_aligned(b.data(), a.data(), n);
  assert(std::memcmp(a.data(), b.data(), n) == 0);
  std::printf("  [ok] 1MB 拷贝正确\n");
}

void bench() {
  const std::size_t n = 64 << 20;   // 64MB
  const int iters = 20;
  auto a = make_data(n);
  std::vector<unsigned char> b(n, 0);

  auto run = [&](auto fn, const char* name) {
    fn(b.data(), a.data(), n);      // 预热
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn(b.data(), a.data(), n);
    const double dt = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
    const double gbs = static_cast<double>(n) * iters / dt / 1e9;
    std::printf("  %-16s %7.2f GB/s\n", name, gbs);
    return gbs;
  };

  std::printf("  --- 64MB × %d 次 ---\n", iters);
  const double g_naive = run(drill::memcpy_naive, "逐字节");
  const double g_align = run(drill::memcpy_aligned, "按8字节");
  const double g_libc = run(static_cast<void* (*)(void*, const void*, std::size_t)>(std::memcpy),
                            "libc memcpy");
  std::printf("  → 按字 / 逐字节 = %.2f×；libc / 按字 = %.1f×\n",
              g_align / g_naive, g_libc / g_align);
  std::printf("\n  【这里必须诚实说明两个反直觉的现象】\n");
  std::printf("   ① -O2 下编译器会把朴素逐字节循环**自动向量化**，所以「逐字节」反而不慢；\n");
  std::printf("      只有 -O0 手动对比时，按字才明显快于逐字节。\n");
  std::printf("   ② 我手写的按字版输给 libc 一个量级——因为 libc 用的是 AVX2/AVX512\n");
  std::printf("      + 非临时存储（non-temporal store，绕过 cache 直接进内存），\n");
  std::printf("      它一次搬 32/64 字节，而我的循环一次只搬 8 字节。\n");
  std::printf("   ⇒ 结论：**对齐优化在「关闭优化 / 窄架构」时有意义，\n");
  std::printf("      但在 x86-64 上应该直接用 libc 或写 SIMD**——知道原理，别重复造轮子。\n");
}

}  // namespace

int main() {
  std::printf("=== 06 memcpy 对齐优化 ===\n");
  test_correctness_all_sizes();
  test_large();
  bench();
  std::printf("[PASS] 06 全部通过\n");
  return 0;
}
