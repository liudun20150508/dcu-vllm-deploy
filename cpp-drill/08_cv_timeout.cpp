// 08_cv_timeout.cpp —— 带超时的条件变量等待（事件/完成通知）
//
// 【为什么这个岗会考】
//   你的 rdma-demo 里 wait_one() 是**裸轮询** CQ（注释里我自己也写了"真实系统不会这样"）。
//   真实通信库要么用 ibv_create_comp_channel + 事件，要么用条件变量/自旋混合。
//   这道题就是把"等一个完成事件，带超时"这件事写对。
//
// 【考点】
//   ① 条件变量为什么必须和 mutex 一起用？—— 检查条件 + 进入等待必须是一个原子动作，
//      否则会丢通知（lost wakeup）
//   ② 为什么必须用 while(pred) 而不是 if(pred)？—— **虚假唤醒**（spurious wakeup）
//   ③ wait_for 的返回值是什么语义？—— 返回 bool：true=谓词成立，false=超时
//   ④ 为什么不能用 sleep？—— sleep 是固定等，既不能提前醒也没法保证精度
//   ⑤ 自旋 + 条件变量怎么组合？—— 短等待自旋（省 syscall），长了再挂起（省 CPU）
//
// 【易错点】
//   - 持锁调用 notify（性能差，虽不致命）
//   - 忘记在修改条件时持锁
//   - 把 wait_for 的返回值忽略 → 超时了还当成功
//   - 谓词里读的条件变量没被 mutex 保护 → 数据竞争
//
// 编译： g++ -std=c++17 -Wall -Wextra -pthread -O2 -o 08 08_cv_timeout.cpp

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

namespace drill {

// 一个"等待完成信号"的小工具：真实系统里等的是 CQ 事件、ACK、门铃……
class Completion {
 public:
  // 等一个完成；返回 true = 等到了，false = 超时
  bool wait_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mtx_);
    // ★ 谓词形式：自动处理虚假唤醒，返回 false 表示超时
    const bool got = cv_.wait_for(lk, timeout, [this] { return pending_ > 0; });
    if (got) --pending_;              // 消费掉一个完成
    else ++timeouts_;
    return got;
  }

  void complete() {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      ++pending_;                     // ★ 计数而非布尔：N 次 complete 对应 N 次消费
    }
    cv_.notify_one();                 // ★ 先解锁再 notify（减少惊群后的竞争）
  }

  int timeouts() const { return timeouts_; }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  // ★ 用计数器而不是 bool：布尔值在两次 complete 之间、两次 wait 之间会"塌缩"——
  //   complete() 把 done_ 置 true，若消费者还没消费就再来一次 complete()，
  //   第二次信号会被第一次覆盖，造成"1000 次通知只收到 <1000 次"的丢信号。
  int pending_{0};
  int timeouts_{0};
};

// 进阶：短等待自旋、长等待挂起（真实通信库的常见做法）
class HybridWaiter {
 public:
  bool wait(std::chrono::microseconds total, std::chrono::microseconds spin) {
    const auto deadline = std::chrono::steady_clock::now() + total;
    // ① 先自旋：省掉"挂起+唤醒"的两次上下文切换（约 1~5us）
    while (std::chrono::steady_clock::now() - (deadline - total) < spin) {
      if (ready_.load(std::memory_order_acquire)) return true;
    }
    // ② 还不行就交给条件变量挂起，避免空转烧 CPU
    std::unique_lock<std::mutex> lk(mtx_);
    return cv_.wait_until(lk, deadline, [this] { return ready_.load(std::memory_order_acquire); });
  }

  void signal() {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      ready_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
  }

 private:
  std::atomic<bool> ready_{false};
  std::mutex mtx_;
  std::condition_variable cv_;
};

}  // namespace drill

// ============================ 测试 ============================

namespace {

void test_timeout_returns_false() {
  drill::Completion c;
  const auto t0 = std::chrono::steady_clock::now();
  const bool got = c.wait_for(std::chrono::milliseconds(50));   // 没人 complete
  const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
  assert(!got);                                                  // ★ 超时要返回 false
  assert(dt >= 45 && dt < 200);                                  // 大致等够了 50ms
  assert(c.timeouts() == 1);
  std::printf("  [ok] 超时返回 false（等了 %lld ms，不是立即返回，也不是死等）\n",
              static_cast<long long>(dt));
}

void test_signal_wakes_up() {
  drill::Completion c;
  std::thread t([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    c.complete();
  });
  const auto t0 = std::chrono::steady_clock::now();
  const bool got = c.wait_for(std::chrono::milliseconds(2000));   // ★ 应提前醒，不用等满 2s
  const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
  t.join();
  assert(got);
  assert(dt < 500);                                               // 远小于 2s
  std::printf("  [ok] 收到信号提前唤醒（约 %lld ms，而非死等 2000ms）\n",
              static_cast<long long>(dt));
}

void test_no_lost_wakeup() {
  // 先 complete 再 wait —— 用"谓词"写法不会丢；用裸 wait 就会永久挂住
  drill::Completion c;
  c.complete();                                                   // 信号先到
  const bool got = c.wait_for(std::chrono::milliseconds(100));     // 后等
  assert(got);                                                    // ★ 没丢信号
  std::printf("  [ok] 先 complete 后 wait 不会丢信号（谓词写法的价值）\n");
}

void test_concurrent_many() {
  drill::Completion c;
  std::atomic<int> got{0};
  std::thread waiter([&] {
    for (int i = 0; i < 1000; ++i) {
      if (c.wait_for(std::chrono::milliseconds(2000))) got.fetch_add(1);
    }
  });
  for (int i = 0; i < 1000; ++i) {
    c.complete();
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  waiter.join();
  assert(got.load() == 1000);                                     // ★ 1000 个信号一个不丢
  std::printf("  [ok] 1000 次完成通知，一次都没丢\n");
}

void test_hybrid_spin_then_block() {
  drill::HybridWaiter hw;
  // ① 信号很快到 → 自旋段就能拿到，不需要挂起
  std::thread quick([&] {
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    hw.signal();
  });
  bool ok = hw.wait(std::chrono::milliseconds(200), std::chrono::microseconds(200));
  quick.join();
  assert(ok);
  std::printf("  [ok] 混合等待：快信号走自旋路径拿到\n");

  // ② 信号慢 → 自旋等不到，挂起等
  drill::HybridWaiter hw2;
  std::thread slow([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    hw2.signal();
  });
  ok = hw2.wait(std::chrono::milliseconds(2000), std::chrono::microseconds(100));
  slow.join();
  assert(ok);
  std::printf("  [ok] 混合等待：慢信号走挂起路径也能拿到\n");
}

}  // namespace

int main() {
  std::printf("=== 08 带超时的条件变量等待 ===\n");
  test_timeout_returns_false();
  test_signal_wakes_up();
  test_no_lost_wakeup();
  test_concurrent_many();
  test_hybrid_spin_then_block();
  std::printf("[PASS] 08 全部通过\n");
  return 0;
}
