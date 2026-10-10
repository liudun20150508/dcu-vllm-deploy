// 02_ring_buffer.cpp —— SPSC 无锁环形缓冲区
//
// 【为什么通信库必考这个】
//   NCCL 的 ring allreduce、网卡的收发队列、你的 nccl-net plugin 里的 req 队列，
//   底层都是"一个生产者一个消费者"的环形缓冲。
//
// 【考点】
//   ① 为什么 capacity 取 2 的幂？—— 用 & (cap-1) 代替 % cap（取模要除法，几十个周期）
//   ② 为什么 head/tail 要分开缓存行（padding）？—— 避免"伪共享"（false sharing）
//      两个核心各改一个变量，若在同一 cache line，会不停互相 invalidate
//   ③ 为什么用 acquire/release 而不是 seq_cst？—— 只要保证"我写完数据再发 tail"这对
//      顺序，不需要全局总序，开销更小
//   ④ 为什么 max_size 是 cap-1 而不是 cap？—— head==tail 无法区分"空"和"满"，
//      所以永远浪费一个槽
//
// 【易错点】
//   - 用 % 而不是 & （忘了 capacity 必须是 2 的幂）
//   - head/tail 不加 padding → 性能比加锁还差
//   - release/acquire 配错 → 消费者可能读到没写完的数据
//   - 当成 MPMC 用 → SPSC 只保证单生产者单消费者
//
// 编译： g++ -std=c++17 -Wall -Wextra -pthread -O2 -o 02 02_ring_buffer.cpp
#if defined(__x86_64__) || defined(__i386__)
#  include <emmintrin.h>
#endif

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>

namespace drill {

constexpr std::size_t kCacheLine = 64;

template <typename T, std::size_t Cap>
class SpscRing {
  static_assert(Cap >= 2, "容量至少 2");
  static_assert((Cap & (Cap - 1)) == 0, "★ 容量必须是 2 的幂，否则 & 掩码不成立");

 public:
  bool push(const T& v) {
    const std::size_t t = tail_.load(std::memory_order_relaxed);
    const std::size_t next = (t + 1) & kMask;          // ★ 用掩码代替取模
    if (next == head_.load(std::memory_order_acquire)) return false;  // 满（浪费一个槽）
    buf_[t] = v;
    tail_.store(next, std::memory_order_release);      // ★ 写数据在前，发 tail 在后
    return true;
  }

  bool pop(T& out) {
    const std::size_t h = head_.load(std::memory_order_relaxed);
    if (h == tail_.load(std::memory_order_acquire)) return false;     // 空
    out = buf_[h];
    head_.store((h + 1) & kMask, std::memory_order_release);
    return true;
  }

  std::size_t size() const {
    return (tail_.load(std::memory_order_acquire) - head_.load(std::memory_order_acquire)) & kMask;
  }
  static constexpr std::size_t capacity() { return Cap - 1; }  // ★ 可用容量 = Cap-1

 private:
  static constexpr std::size_t kMask = Cap - 1;

  // ★ padding：让 head_ 和 tail_ 落在不同 cache line
  alignas(kCacheLine) std::atomic<std::size_t> head_{0};
  alignas(kCacheLine) std::atomic<std::size_t> tail_{0};
  alignas(kCacheLine) T buf_[Cap]{};
};

}  // namespace drill

// ============================ 测试 ============================

namespace {

void test_fifo_order() {
  drill::SpscRing<int, 8> q;
  assert(q.capacity() == 7);
  for (int i = 0; i < 7; ++i) assert(q.push(i));
  assert(!q.push(99));                    // ★ 满了：只有 7 个槽可用
  int v = 0;
  for (int i = 0; i < 7; ++i) {
    assert(q.pop(v));
    assert(v == i);                       // ★ 严格 FIFO
  }
  assert(!q.pop(v));                      // 空了
  std::printf("  [ok] FIFO 顺序 / 满空判定 / 可用容量 = Cap-1\n");
}

void test_wraparound() {
  drill::SpscRing<int, 4> q;              // 可用 3
  int v = 0;
  for (int round = 0; round < 100; ++round) {   // ★ 反复回绕，验证掩码正确
    for (int i = 0; i < 3; ++i) assert(q.push(round * 3 + i));
    for (int i = 0; i < 3; ++i) {
      assert(q.pop(v));
      assert(v == round * 3 + i);
    }
  }
  std::printf("  [ok] 回绕 100 轮仍保序\n");
}

void test_spsc_multithread() {
  constexpr int kN = 200000;
  drill::SpscRing<int, 1024> q;
  std::atomic<long> sum{0};
  std::atomic<bool> done{false};

  std::thread producer([&] {
    for (int i = 0; i < kN; ++i) {
      while (!q.push(i)) {                // 满了就自旋
#if defined(__x86_64__)
        _mm_pause();
#endif
      }
    }
    done.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    long local = 0;
    int v = 0, got = 0;
    while (got < kN) {
      if (q.pop(v)) { local += v; ++got; }
#if defined(__x86_64__)
      else _mm_pause();
#endif
    }
    sum.store(local, std::memory_order_release);
  });

  producer.join();
  consumer.join();
  assert(done.load());
  // 0+1+...+(N-1)
  const long expect = static_cast<long>(kN) * (kN - 1) / 2;
  assert(sum.load() == expect);           // ★ 一个都不能丢、不能重
  std::printf("  [ok] 单生产单消费 %d 条，无丢失无重复（和=%ld）\n", kN, sum.load());
}

}  // namespace

int main() {
  std::printf("=== 02 环形缓冲区（SPSC lock-free）===\n");
  test_fifo_order();
  test_wraparound();
  test_spsc_multithread();
  std::printf("[PASS] 02 全部通过\n");
  return 0;
}
