// 07_lru_cache.cpp —— LRU 缓存（O(1) get/put）
//
// 【考点】
//   ① 为什么必须是"哈希表 + 双向链表"？
//      —— 哈希表给 O(1) 定位；双向链表给 O(1) 删除/移动到头部
//        （单链表删一个节点需要前驱，做不到 O(1)）
//   ② 为什么用 std::list 而不是手写链表？—— 问答写 std::list 版本足够；
//      但要知道 std::list 每个节点一次堆分配（可作为加分点展开）
//   ③ list 的迭代器为什么能长期保存？—— std::list 插入/删除其它节点不会使迭代器失效
//   ④ 怎么变成 LFU / 带 TTL 的缓存？—— 加分追问方向
//
// 【易错点】
//   - 用 vector 做链表 → 删除是 O(n)
//   - 忘了把访问过的节点移到头部（那就不叫 LRU 了）
//   - put 已存在的 key 时忘了更新值 + 提升
//   - map 里存迭代器，但 list 删了节点后迭代器没同步清
//
// 编译： g++ -std=c++17 -Wall -Wextra -O2 -o 07 07_lru_cache.cpp

#include <cassert>
#include <cstdio>
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace drill {

template <typename K, typename V>
class LruCache {
 public:
  explicit LruCache(std::size_t capacity) : cap_(capacity) {}

  std::optional<V> get(const K& key) {
    auto it = map_.find(key);
    if (it == map_.end()) return std::nullopt;
    // ★ 命中：把该节点移到链表头（最近使用）
    items_.splice(items_.begin(), items_, it->second);
    return it->second->second;
  }

  void put(const K& key, const V& value) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      it->second->second = value;                   // ★ 已存在：更新值
      items_.splice(items_.begin(), items_, it->second);  // 并提升
      return;
    }
    if (items_.size() >= cap_) {
      // 淘汰链表尾（最久未使用）
      auto& victim = items_.back();
      map_.erase(victim.first);
      items_.pop_back();
    }
    items_.emplace_front(key, value);
    map_[key] = items_.begin();
  }

  std::size_t size() const { return items_.size(); }
  std::size_t capacity() const { return cap_; }

  // 调试：按"最近→最久"顺序打印 key
  void dump_order(std::vector<K>& out) const {
    out.clear();
    for (const auto& kv : items_) out.push_back(kv.first);
  }

 private:
  std::size_t cap_;
  // 头部 = 最近使用，尾部 = 最久未使用
  std::list<std::pair<K, V>> items_;
  std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> map_;
};

}  // namespace drill

// ============================ 测试 ============================

namespace {

void test_basic_eviction() {
  drill::LruCache<int, std::string> c(2);
  c.put(1, "one");
  c.put(2, "two");
  assert(c.get(1).value() == "one");
  c.put(3, "three");                 // ★ 应该淘汰 2（因为 1 刚被访问过）
  assert(!c.get(2).has_value());
  assert(c.get(1).has_value() && c.get(3).has_value());
  std::printf("  [ok] 容量 2 淘汰最久未用的（访问过的被保留）\n");
}

void test_update_existing() {
  drill::LruCache<int, int> c(2);
  c.put(1, 10);
  c.put(2, 20);
  c.put(1, 100);                     // ★ 更新 + 提升
  assert(c.get(1).value() == 100);
  c.put(3, 30);                      // 淘汰 2
  assert(!c.get(2).has_value());
  assert(c.get(1).value() == 100 && c.get(3).value() == 30);
  std::printf("  [ok] 更新已有 key：值被覆盖且被提升\n");
}

void test_lru_order_precise() {
  drill::LruCache<int, int> c(3);
  c.put(1, 1);
  c.put(2, 2);
  c.put(3, 3);                       // 顺序(新→旧): 3,2,1
  (void)c.get(1);                    // 访问 1 → 顺序: 1,3,2
  std::vector<int> order;
  c.dump_order(order);
  assert((order == std::vector<int>{1, 3, 2}));

  c.put(4, 4);                       // 淘汰 2（最旧）→ 顺序: 4,1,3
  c.dump_order(order);
  assert((order == std::vector<int>{4, 1, 3}));
  std::printf("  [ok] 精确 LRU 顺序：1,3,2 → 插入4 → 4,1,3\n");
}

void test_capacity_one() {
  drill::LruCache<int, int> c(1);
  c.put(1, 1);
  c.put(2, 2);
  assert(!c.get(1).has_value());
  assert(c.get(2).value() == 2);
  assert(c.size() == 1);
  std::printf("  [ok] 容量 1 的边界\n");
}

void test_miss_and_size() {
  drill::LruCache<int, int> c(4);
  assert(!c.get(999).has_value());   // 未命中
  assert(c.size() == 0);
  for (int i = 0; i < 10; ++i) c.put(i, i);   // 超过容量
  assert(c.size() == 4);             // ★ 不会超过容量
  std::printf("  [ok] 未命中返回 nullopt；size 不超过 capacity\n");
}

}  // namespace

int main() {
  std::printf("=== 07 LRU 缓存 ===\n");
  test_basic_eviction();
  test_update_existing();
  test_lru_order_precise();
  test_capacity_one();
  test_miss_and_size();
  std::printf("[PASS] 07 全部通过\n");
  return 0;
}
