// 04_memory_pool.cpp —— 固定块内存池（free list）
//
// 【为什么这个岗会考】
//   RDMA 的内存必须"注册"（ibv_reg_mr）才能被网卡访问，注册/注销很贵；
//   而且网卡用的是**虚拟地址**——所以通信库（NCCL 的 buffer、你的 rdma-demo）
//   都倾向"一次申请一大块 + 池化管理"，而不是反复 malloc。
//
// 【考点】
//   ① 为什么不用 malloc？—— 每次 malloc 有锁/元数据开销；长期运行会碎片化；
//      实时路径上不能有不可控延迟
//   ② free list 的经典实现：把空闲块的**前 8 字节当地址用**（块在空闲时不需要存数据）
//   ③ 为什么需要对齐（alignas）？—— 高速网络/DCU 的 DMA 通常要求 64/128 字节对齐
//   ④ 越界/重复释放怎么发现？—— 加 magic 标记（这是"调试版"的做法）
//
// 【易错点】
//   - 空池时返回 nullptr 但调用方没判空
//   - 块大小小于 sizeof(void*) → free list 存不下指针
//   - 重复 free 同一块 → free list 成环，后续全错
//   - 忘记对齐 → DMA 出错（性能掉一半或直接失败）
//
// 编译： g++ -std=c++17 -Wall -Wextra -O2 -o 04 04_memory_pool.cpp

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <unordered_set>
#include <vector>

namespace drill {

class FixedPool {
 public:
  // block_size: 单块大小；nblocks: 块数；alignment: 对齐要求（默认 cache line）
  FixedPool(std::size_t block_size, std::size_t nblocks, std::size_t alignment = 64)
      : nblocks_(nblocks) {
    // ★ 块大小下限 = sizeof(void*)（free list 要用前 8 字节存指针）
    block_size_ = block_size < sizeof(void*) ? sizeof(void*) : block_size;
    // ★ 再把块大小抬到 alignment 的倍数——否则起始块对齐了，后面第 i 块还是歪的
    block_size_ = (block_size_ + alignment - 1) / alignment * alignment;

    // ★ 一次申请整块，减少系统调用 + 保证地址连续（对 RDMA 友好）
    // 注意 aligned_alloc 要求 size 必须是 alignment 的整数倍，先向上取整
    std::size_t total = block_size_ * nblocks_ + alignment;
    total = (total + alignment - 1) / alignment * alignment;
    raw_ = static_cast<char*>(std::aligned_alloc(alignment, total));
    if (!raw_) throw std::bad_alloc();

    // 建立起址对齐的 free list
    char* p = raw_;
    const std::size_t misalign = reinterpret_cast<std::uintptr_t>(p) % alignment;
    if (misalign != 0) p += (alignment - misalign);
    free_head_ = nullptr;
    for (std::size_t i = 0; i < nblocks_; ++i) {
      char* blk = p + i * block_size_;
      *reinterpret_cast<char**>(blk) = free_head_;   // ★ 空闲块的前 8 字节存下一个空闲块地址
      free_head_ = blk;
    }
  }

  ~FixedPool() { std::free(raw_); }

  FixedPool(const FixedPool&) = delete;
  FixedPool& operator=(const FixedPool&) = delete;

  void* alloc() {
    if (!free_head_) return nullptr;                 // ★ 池空，调用方必须判空
    char* blk = free_head_;
    free_head_ = *reinterpret_cast<char**>(blk);
    ++in_use_;
    blocks_.insert(blk);                             // 调试用：记录已分配块
    return blk;
  }

  void dealloc(void* p) {
    if (!p) return;
    char* blk = static_cast<char*>(p);
    if (blocks_.erase(blk) == 0) {                   // ★ 重复释放 / 野指针能查出来
      std::fprintf(stderr, "[pool] 非法释放 %p\n", p);
      std::abort();
    }
    *reinterpret_cast<char**>(blk) = free_head_;
    free_head_ = blk;
    --in_use_;
  }

  std::size_t in_use() const { return in_use_; }
  std::size_t capacity() const { return nblocks_; }
  std::size_t block_size() const { return block_size_; }

 private:
  char* raw_{nullptr};
  char* free_head_{nullptr};
  std::size_t block_size_{0};
  std::size_t nblocks_{0};
  std::size_t in_use_{0};
  std::unordered_set<char*> blocks_;                  // 调试：检测重复释放
};

// 带 RAII 的分配器，配合池使用（问答常接着问这个）
class PoolGuard {
 public:
  PoolGuard(FixedPool& pool) : pool_(pool), p_(pool.alloc()) {}
  ~PoolGuard() { pool_.dealloc(p_); }
  PoolGuard(const PoolGuard&) = delete;
  PoolGuard& operator=(const PoolGuard&) = delete;
  void* get() const { return p_; }
  bool valid() const { return p_ != nullptr; }

 private:
  FixedPool& pool_;
  void* p_;
};

}  // namespace drill

// ============================ 测试 ============================

namespace {

void test_alloc_all_then_empty() {
  drill::FixedPool pool(128, 16);
  std::vector<void*> v;
  for (int i = 0; i < 16; ++i) {
    void* p = pool.alloc();
    assert(p != nullptr);
    v.push_back(p);
  }
  assert(pool.alloc() == nullptr);           // ★ 用完了
  assert(pool.in_use() == 16);
  for (void* p : v) pool.dealloc(p);
  assert(pool.in_use() == 0);
  assert(pool.alloc() != nullptr);           // 回收后又能给
  std::printf("  [ok] 分配满 → 返回 nullptr → 回收后复用\n");
}

void test_alignment() {
  drill::FixedPool pool(100, 8, 64);
  for (int i = 0; i < 8; ++i) {
    void* p = pool.alloc();
    assert(reinterpret_cast<std::uintptr_t>(p) % 64 == 0);   // ★ 按 64 字节对齐
  }
  std::printf("  [ok] 每块都按 64 字节对齐（DMA/DCU 需要）\n");
}

void test_no_overlap_and_writable() {
  drill::FixedPool pool(64, 32);
  std::vector<char*> v;
  for (int i = 0; i < 32; ++i) v.push_back(static_cast<char*>(pool.alloc()));
  // 每块写自己的编号，验证没有重叠
  for (int i = 0; i < 32; ++i) std::snprintf(v[i], 64, "blk-%d", i);
  for (int i = 0; i < 32; ++i) {
    char want[64];
    std::snprintf(want, 64, "blk-%d", i);
    assert(std::string(v[i]) == want);       // ★ 块之间互不干扰
  }
  // 地址两两不同
  std::unordered_set<void*> s(v.begin(), v.end());
  assert(s.size() == 32);
  std::printf("  [ok] 32 块互不重叠、内容互不干扰\n");
}

void test_min_block_size() {
  drill::FixedPool pool(1, 4);               // 比指针还小的请求 → 自动抬到 sizeof(void*)
  assert(pool.block_size() >= sizeof(void*));
  std::printf("  [ok] 块大小下限自动抬到 sizeof(void*)=%zu\n", sizeof(void*));
}

void test_guard_raii() {
  drill::FixedPool pool(256, 4);
  {
    drill::PoolGuard g1(pool), g2(pool);
    assert(g1.valid() && g2.valid());
    assert(pool.in_use() == 2);
  }                                          // 出作用域自动归还
  assert(pool.in_use() == 0);
  std::printf("  [ok] RAII guard 自动归还\n");
}

}  // namespace

int main() {
  std::printf("=== 04 固定块内存池 ===\n");
  test_alloc_all_then_empty();
  test_alignment();
  test_no_overlap_and_writable();
  test_min_block_size();
  test_guard_raii();
  std::printf("[PASS] 04 全部通过\n");
  return 0;
}
