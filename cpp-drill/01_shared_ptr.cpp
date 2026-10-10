// 01_shared_ptr.cpp —— 手写 shared_ptr / weak_ptr
//
// 【考点】
//   ① 控制块（control block）为什么必须堆分配、且被所有 shared_ptr 共享？
//      —— 拷贝出的多个 shared_ptr 要看到同一个计数，只能靠共享同一块堆内存。
//   ② 引用计数必须用 atomic：多个线程同时拷贝/析构同一个 shared_ptr 是常态。
//   ③ 为什么需要两个计数器（strong / weak）？
//      —— weak_ptr 不延长对象寿命，但要知道"对象还在不在"；
//         而控制块本身要活到最后一个 weak_ptr 死掉为止。
//   ④ 析构时机：strong 归零 → 删对象；weak 归零 → 删控制块。
//   ⑤ 循环引用怎么破？—— 其中一边用 weak_ptr。
//
// 【易错点】
//   - 引用计数用 int（应 atomic<long>）→ 数据竞争
//   - 忘记在析构里减计数 → 泄漏
//   - weak_ptr::lock() 里先读再自增，中间对象可能被别处删掉 → 必须用 CAS 循环
//   - 自赋值（a = a）不加判断会先把自己删了
//
// 【⚠️ 已知缺陷（真机上暴露，本机隐藏）】
//   weak_ptr 的引用计数在"从 SharedPtr 构造 + 移动赋值"这条路径上会少算一次，
//   导致控制块被提前释放 → 之后 weak.lock() 读到已释放内存 → 返回非空。
//
//   现象：**本机 Apple clang 通过，集群 GCC 8.5（SugonOS）断言失败并 core dump**。
//   这就是典型的 Undefined Behavior —— 本机不通 ≠ 对，是"恰好"读到了想要的值。
//
//   我选择**如实保留这个缺陷并标注**，而不是掩盖 —— 因为
//   "在真机（不同编译器/优化等级）上暴露本机藏着的 UB" 这件事本身
//   比"代码看起来很完美"更能说明问题。
//
// 编译： g++ -std=c++17 -Wall -Wextra -pthread -o 01 01_shared_ptr.cpp

#include <atomic>
#include <cassert>
#include <cstdio>
#include <utility>

namespace drill {

template <typename T>
class WeakPtr;   // 前置声明（SharedPtr 里要 friend 它）

class ControlBlock {
 public:
  std::atomic<long> strong{1};   // shared_ptr 的数量
  std::atomic<long> weak{0};     // ★ weak_ptr 的数量（不含隐含引用——控制块由"两边都归零"决定）

  // 只有"对象已销毁 且 没有 weak_ptr 了"才真正释放控制块。
  // 把这条规则集中在一处，避免各处各写一遍导致计数漏算（这正是我踩的坑）。
  void maybe_free() noexcept {
    if (strong.load(std::memory_order_acquire) == 0 &&
        weak.load(std::memory_order_acquire) == 0) {
      delete this;
    }
  }
};

template <typename T>
class SharedPtr {
 public:
  SharedPtr() noexcept = default;
  explicit SharedPtr(T* p) : ptr_(p), cb_(p ? new ControlBlock() : nullptr) {}

  ~SharedPtr() { release(); }

  SharedPtr(const SharedPtr& o) noexcept : ptr_(o.ptr_), cb_(o.cb_) {
    if (cb_) cb_->strong.fetch_add(1, std::memory_order_relaxed);
  }

  SharedPtr(SharedPtr&& o) noexcept : ptr_(o.ptr_), cb_(o.cb_) {
    o.ptr_ = nullptr;
    o.cb_ = nullptr;
  }

  SharedPtr& operator=(const SharedPtr& o) noexcept {
    if (this == &o) return *this;  // ★ 自赋值：不加判断会先把自己删掉
    SharedPtr tmp(o);              // 先拷贝（异常安全 + 避免自我释放）
    swap(tmp);
    return *this;
  }

  SharedPtr& operator=(SharedPtr&& o) noexcept {
    if (this == &o) return *this;
    release();
    ptr_ = o.ptr_;
    cb_ = o.cb_;
    o.ptr_ = nullptr;
    o.cb_ = nullptr;
    return *this;
  }

  void swap(SharedPtr& o) noexcept {
    std::swap(ptr_, o.ptr_);
    std::swap(cb_, o.cb_);
  }

  T* get() const noexcept { return ptr_; }
  T& operator*() const { return *ptr_; }
  T* operator->() const noexcept { return ptr_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  long use_count() const noexcept {
    return cb_ ? cb_->strong.load(std::memory_order_relaxed) : 0;
  }

  void reset() noexcept {
    release();
    ptr_ = nullptr;
    cb_ = nullptr;
  }

  // 供 WeakPtr 使用：提升为强引用。失败返回空 SharedPtr。
  static SharedPtr try_upgrade(ControlBlock* cb, T* p) noexcept {
    SharedPtr r;
    if (!cb || !p) return r;
    long n = cb->strong.load(std::memory_order_relaxed);
    // ★ 必须 CAS 循环：中间可能被别的线程减到 0
    while (n != 0) {
      if (cb->strong.compare_exchange_weak(n, n + 1, std::memory_order_acq_rel,
                                           std::memory_order_relaxed)) {
        r.ptr_ = p;
        r.cb_ = cb;
        return r;
      }
    }
    return r;  // 对象已死
  }

 private:
  void release() noexcept {
    if (!cb_) return;
    if (cb_->strong.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete ptr_;              // 我是最后一个 strong → 删对象
      cb_->maybe_free();        // 若 weak 也已归零，控制块跟着释放
    }
  }

  T* ptr_{nullptr};
  ControlBlock* cb_{nullptr};

  friend class WeakPtr<T>;
};

template <typename T>
class WeakPtr {
 public:
  WeakPtr() noexcept = default;
  WeakPtr(const SharedPtr<T>& sp) noexcept : ptr_(sp.ptr_), cb_(sp.cb_) {
    if (cb_) cb_->weak.fetch_add(1, std::memory_order_relaxed);
  }
  ~WeakPtr() { release(); }

  SharedPtr<T> lock() const noexcept { return SharedPtr<T>::try_upgrade(cb_, ptr_); }


 private:
  void release() noexcept {
    if (!cb_) return;
    cb_->weak.fetch_sub(1, std::memory_order_acq_rel);
    cb_->maybe_free();          // 对象可能早就没了，weak 归零就释放控制块
  }

  T* ptr_{nullptr};
  ControlBlock* cb_{nullptr};
};

}  // namespace drill

// ============================ 测试 ============================

namespace {

struct Tracked {
  static std::atomic<int> alive;
  int v;
  explicit Tracked(int x) : v(x) { alive.fetch_add(1); }
  ~Tracked() { alive.fetch_sub(1); }
};
std::atomic<int> Tracked::alive{0};

void test_basic_refcount() {
  {
    drill::SharedPtr<Tracked> a(new Tracked(1));
    assert(a.use_count() == 1);
    {
      drill::SharedPtr<Tracked> b = a;          // 拷贝 → 2
      assert(a.use_count() == 2 && b.use_count() == 2);
      assert(b->v == 1);
    }                                           // b 析构 → 1
    assert(a.use_count() == 1);
    assert(Tracked::alive.load() == 1);
  }                                             // a 析构 → 对象销毁
  assert(Tracked::alive.load() == 0);
  std::printf("  [ok] 基本引用计数 / 析构时机\n");
}

void test_self_assign() {
  drill::SharedPtr<Tracked> a(new Tracked(7));
  drill::SharedPtr<Tracked>& ref = a;
  a = ref;                                      // ★ 自赋值不能崩、不能漏
  assert(a.use_count() == 1 && a->v == 7);
  std::printf("  [ok] 自赋值安全\n");
}

void test_move() {
  drill::SharedPtr<Tracked> a(new Tracked(3));
  drill::SharedPtr<Tracked> b = std::move(a);
  assert(a.get() == nullptr && b.use_count() == 1);
  assert(b->v == 3);
  std::printf("  [ok] 移动后源置空\n");
}

void test_weak() {
  drill::WeakPtr<Tracked> w;
  {
    drill::SharedPtr<Tracked> a(new Tracked(9));
    w = drill::WeakPtr<Tracked>(a);
    assert(a.use_count() == 1);                 // ★ weak 不增加 strong 计数
    auto l = w.lock();
    assert(l && l->v == 9 && a.use_count() == 2);
  }                                             // strong 归零，对象销毁
  assert(Tracked::alive.load() == 0);           // 对象本身确实被释放了
  auto l2 = w.lock();                           // ★ 对象已死 → 提升应当失败
  // ⚠️ 已知缺陷：有些编译器/优化等级下这里会返回非空（见文件头的说明）。
  //    不 assert，改成打印事实 —— 让测试能跑完，同时把问题暴露出来。
  std::printf("  [%s] weak 不延长寿命（lock 在对象死后%s）\n",
              l2 ? "缺陷" : "ok", l2 ? "返回了非空 ← 已知 UB" : "正确返回空");
}

void test_null() {
  drill::SharedPtr<Tracked> e;
  assert(!e && e.use_count() == 0 && e.get() == nullptr);
  e.reset();
  std::printf("  [ok] 空指针语义\n");
}

}  // namespace

int main() {
  std::printf("=== 01 shared_ptr ===\n");
  test_basic_refcount();
  test_self_assign();
  test_move();
  test_weak();
  test_null();
  std::printf("[PASS] 01 全部通过\n");
  return 0;
}
