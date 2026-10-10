// 05_lockfree_queue.cpp —— MPMC 有界无锁队列（基于数组 + 序号）
//
// 【为什么学这个】
//   通信库的完成队列（CQ）、请求池，多线程下都要无锁；
//   NCCL 内部大量用原子操作做"轻量同步"，而不是全靠 mutex。
//
// 【考点】
//   ① SPSC 只需 head/tail 两个指针；MPMC 为什么不够？
//      —— 多个生产者会抢同一个 tail，必须用 CAS 抢占"槽位"
//   ② 怎么区分"槽位被占了但数据还没写完"？
//      —— 每个槽配一个 sequence 序号（这是 Dmitry Vyukov 的经典解法）
//   ③ push/pop 的 CAS 失败怎么办？—— 循环重试（reload 最新值）
//   ④ 为什么容量取 2 的幂？—— 又是掩码代替取模
//
// 【易错点】
//   - 只在 push 里 CAS tail，忘了 slot 的 seq 也要原子
//   - 用 memory_order_relaxed 发布数据 → 消费者可能读到半截数据
//   - 队列满/空时错误地自增序号 → 死锁
//
// 编译： g++ -std=c++17 -Wall -Wextra -pthread -O2 -o 05 05_lockfree_queue.cpp

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>

namespace drill {

template <typename T, std::size_t Cap>
class MpmcQueue {
  static_assert((Cap & (Cap - 1)) == 0, "容量必须是 2 的幂");

  struct Slot {
    std::atomic<std::size_t> seq;
    T data;
  };

 public:
  MpmcQueue() {
    for (std::size_t i = 0; i < Cap; ++i) buf_[i].seq.store(i, std::memory_order_relaxed);
    enq_.store(0, std::memory_order_relaxed);
    deq_.store(0, std::memory_order_relaxed);
  }

  bool push(const T& v) {
    for (;;) {
      std::size_t pos = enq_.load(std::memory_order_relaxed);
      Slot& s = buf_[pos & kMask];
      const std::size_t seq = s.seq.load(std::memory_order_acquire);
      const std::intptr_t diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);
      if (diff == 0) {
        // 槽位轮到我：CAS 抢 enq_，成功才算占住
        if (enq_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          s.data = v;
          s.seq.store(pos + 1, std::memory_order_release);   // ★ release：数据先写好
          return true;
        }
      } else if (diff < 0) {
        return false;                                        // 队列满
      } else {
        // 别人抢先了，重新读 enq_
      }
    }
  }

  bool pop(T& out) {
    for (;;) {
      std::size_t pos = deq_.load(std::memory_order_relaxed);
      Slot& s = buf_[pos & kMask];
      const std::size_t seq = s.seq.load(std::memory_order_acquire);
      const std::intptr_t diff =
          static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos + 1);
      if (diff == 0) {
        if (deq_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          out = s.data;
          s.seq.store(pos + Cap, std::memory_order_release);  // 还给生产者（下一轮）
          return true;
        }
      } else if (diff < 0) {
        return false;                                        // 队列空
      }
    }
  }

  static constexpr std::size_t capacity() { return Cap; }

 private:
  static constexpr std::size_t kMask = Cap - 1;
  alignas(64) std::atomic<std::size_t> enq_{0};
  alignas(64) std::atomic<std::size_t> deq_{0};
  alignas(64) Slot buf_[Cap];
};

}  // namespace drill

// ============================ 测试 ============================

namespace {

void test_single_thread() {
  drill::MpmcQueue<int, 8> q;
  for (int i = 0; i < 8; ++i) assert(q.push(i));   // ★ MPMC 可以用满全部槽位
  assert(!q.push(99));                             // 满了
  int v = 0;
  for (int i = 0; i < 8; ++i) {
    assert(q.pop(v));
    assert(v == i);
  }
  assert(!q.pop(v));                               // 空了
  std::printf("  [ok] 单线程：可以填满 8 槽，FIFO 顺序正确\n");
}

void test_multiproducer_multiconsumer() {
  constexpr int kProd = 4, kCons = 4, kPer = 50000;
  constexpr long kTotal = static_cast<long>(kProd) * kPer;

  drill::MpmcQueue<long, 1024> q;
  std::atomic<long> consumed{0};
  std::atomic<long> checksum{0};
  std::atomic<bool> prod_done{false};
  std::atomic<int> prod_left{kProd};

  std::vector<std::thread> threads;
  for (int p = 0; p < kProd; ++p) {
    threads.emplace_back([&, p] {
      for (int i = 0; i < kPer; ++i) {
        const long v = static_cast<long>(p) * kPer + i;
        while (!q.push(v)) std::this_thread::yield();
      }
      if (prod_left.fetch_sub(1) == 1) prod_done.store(true, std::memory_order_release);
    });
  }
  for (int c = 0; c < kCons; ++c) {
    threads.emplace_back([&] {
      long local_sum = 0;
      long v = 0;
      while (consumed.load(std::memory_order_acquire) < kTotal) {
        if (q.pop(v)) {
          local_sum += v;
          consumed.fetch_add(1, std::memory_order_release);
        } else {
          if (prod_done.load(std::memory_order_acquire) && consumed.load() >= kTotal) break;
          std::this_thread::yield();
        }
      }
      checksum.fetch_add(local_sum);
    });
  }
  for (auto& t : threads) t.join();

  const long expect = kTotal * (kTotal - 1) / 2;    // 0+1+...+kTotal-1
  assert(consumed.load() == kTotal);                // ★ 一条不多一条不少
  assert(checksum.load() == expect);                // ★ 内容也没错
  std::printf("  [ok] %d 生产者 × %d 消费者 × %d 条 = %ld，无丢失无重复（校验和正确）\n",
              kProd, kCons, kPer, kTotal);
}

}  // namespace

int main() {
  std::printf("=== 05 无锁队列（MPMC）===\n");
  test_single_thread();
  test_multiproducer_multiconsumer();
  std::printf("[PASS] 05 全部通过\n");
  return 0;
}
