// 03_thread_pool.cpp —— 线程池
//
// 【为什么这个岗会考】
//   通信库到处是"投递任务 → 异步等完成"：你的 nccl-net plugin 里
//   mc_isend/mc_irecv 就需要一个后台执行队列；NCCL 自己也有 proxy 线程。
//
// 【考点】
//   ① 为什么要 packaged_task + future？—— 让调用方能拿到返回值/异常
//   ② 条件变量为什么必须配 mutex，且判断用 while 不是 if？—— 防虚假唤醒
//   ③ 析构时怎么优雅退出？—— 置 stop → notify_all → join 全部线程
//   ④ 为什么 worker 里要先 move 出任务再解锁执行？—— 否则任务执行期间还占着锁，
//      其它线程无法提交/取任务，退化成串行
//
// 【易错点】
//   - 用 if 判断谓词 → 虚假唤醒时取到空任务崩溃
//   - 析构不 join → terminate
//   - 持锁执行任务 → 无并发
//   - std::function 拷贝开销大 → 用 move
//
// 编译： g++ -std=c++17 -Wall -Wextra -pthread -O2 -o 03 03_thread_pool.cpp

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace drill {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t n) {
    workers_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ~ThreadPool() { shutdown(); }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // 提交任务，返回 future 以便取结果 / 捕获异常
  template <typename F, typename... Args>
  auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using R = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<R()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<R> fut = task->get_future();
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (stop_) throw std::runtime_error("submit on stopped pool");
      tasks_.emplace([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (stop_) return;
      stop_ = true;
    }
    cv_.notify_all();                    // ★ 唤醒所有等待线程，让它们看到 stop_
    for (auto& t : workers_) {
      if (t.joinable()) t.join();        // ★ 必须 join，否则析构时 terminate
    }
    workers_.clear();
  }

  std::size_t pending() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return tasks_.size();
  }

 private:
  void worker_loop() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });  // ★ while 语义，防虚假唤醒
        if (stop_ && tasks_.empty()) return;                        // 退出前把活干完
        job = std::move(tasks_.front());                            // ★ move 出来
        tasks_.pop();
      }                                                             // ★ 先解锁
      job();                                                        // 再执行
    }
  }

  mutable std::mutex mtx_;
  std::condition_variable cv_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  bool stop_{false};
};

}  // namespace drill

// ============================ 测试 ============================

namespace {

void test_parallel_sum() {
  drill::ThreadPool pool(4);
  constexpr int kN = 1000;
  std::vector<std::future<long>> futs;
  for (int i = 0; i < kN; ++i) {
    futs.push_back(pool.submit([i] { return static_cast<long>(i) * i; }));
  }
  long sum = 0;
  for (auto& f : futs) sum += f.get();
  long expect = 0;
  for (int i = 0; i < kN; ++i) expect += static_cast<long>(i) * i;
  assert(sum == expect);
  std::printf("  [ok] %d 个任务并发执行，结果正确（sum=%ld）\n", kN, sum);
}

void test_args_and_return() {
  drill::ThreadPool pool(2);
  auto f1 = pool.submit([](int a, int b) { return a + b; }, 3, 4);
  auto f2 = pool.submit([] { return std::string("hello"); });
  assert(f1.get() == 7);
  assert(f2.get() == "hello");
  std::printf("  [ok] 带参数 / 带返回值\n");
}

void test_exception_propagates() {
  drill::ThreadPool pool(1);
  auto f = pool.submit([]() -> int { throw std::runtime_error("boom"); });
  bool caught = false;
  try {
    (void)f.get();                       // ★ 异常通过 future 传回调用方
  } catch (const std::runtime_error& e) {
    caught = (std::string(e.what()) == "boom");
  }
  assert(caught);
  std::printf("  [ok] 任务内异常能传回调用方\n");
}

void test_graceful_shutdown() {
  std::atomic<int> done{0};
  {
    drill::ThreadPool pool(4);
    for (int i = 0; i < 50; ++i) pool.submit([&done] { done.fetch_add(1); });
    // 出作用域 → 析构 → 应等所有任务跑完才返回（不丢任务、不 terminate）
  }
  assert(done.load() == 50);
  std::printf("  [ok] 析构时优雅退出，50 个任务全部执行完\n");
}

}  // namespace

int main() {
  std::printf("=== 03 线程池 ===\n");
  test_parallel_sum();
  test_args_and_return();
  test_exception_propagates();
  test_graceful_shutdown();
  std::printf("[PASS] 03 全部通过\n");
  return 0;
}
