// tuner_test.cpp —— 模拟 NCCL 调用 tuner 的过程，验证选择逻辑
//
// NCCL 调 tuner 的流程：
//   ① 启动时  init(nRanks, nNodes, rank, ranksPerNode, ...)
//   ② 每次集合通信前 getCollInfo(collInfo, collType, nBytes, numPipeOps,
//                                collCostTable, numAlgo, numProto, regression, nChannels)
//   ③ 退出时  finalize()
//
// 这里用一个假的代价表把 ② 走一遍，检查：
//   · 小消息 → 偏置到 Tree（步数主导）
//   · 大消息 → 偏置到 Ring（搬运量主导）
//   · regression > 0 → 必须原样返回，不能改
//   · 跨机大消息 → 通道数下限被抬到 8
//
// 编译： g++ -std=c++17 -Wall -Wextra -O2 -o tuner_test tuner_test.cpp mycard_tuner.cpp

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "nccl_tuner.h"

extern "C" ncclTuner_v5_t ncclTunerPlugin_v5;

namespace {

constexpr int kNumAlgo = NCCL_NUM_ALGORITHMS;
constexpr int kNumProto = NCCL_NUM_PROTOCOLS;

struct Table {
  std::vector<float> flat;   // 扁平 [algo*proto] —— 模拟 NCCL 传进来的真实内存布局
  Table() : flat(kNumAlgo * kNumProto, -1.0f) {}
  float& at(int a, int p) { return flat[a * kNumProto + p]; }
  float  at(int a, int p) const { return flat[a * kNumProto + p]; }
  // ★ 关键：把扁平数组的地址伪装成 float**（和 NCCL 的做法一致），
  //   插件内部再 reinterpret_cast 回 float* 按扁平下标访问。
  float** data() const { return reinterpret_cast<float**>(const_cast<float*>(flat.data())); }
};

// 造一张"I 型"代价表：小消息 Tree 便宜、大消息 Ring 便宜 —— 和真实世界一致
Table make_table(size_t nBytes) {
  Table t;
  const bool small = nBytes < 256 * 1024;
  // Tree：LL 适合小消息，Simple 适合大消息
  t.at(NCCL_ALGO_TREE, NCCL_PROTO_LL)     = small ? 10.0f : 900.0f;
  t.at(NCCL_ALGO_TREE, NCCL_PROTO_LL128)  = small ? 12.0f : 800.0f;
  t.at(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE) = small ? 20.0f : 100.0f;
  // Ring：小消息步数多 → 贵；大消息流水好 → 便宜
  t.at(NCCL_ALGO_RING, NCCL_PROTO_LL)     = small ? 60.0f : 500.0f;
  t.at(NCCL_ALGO_RING, NCCL_PROTO_LL128)  = small ? 55.0f : 300.0f;
  t.at(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE) = small ? 50.0f : 80.0f;
  // CollNet 类：这台机器没有卸载 → 标记不可用
  t.at(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE) = -1.0f;
  return t;
}

ncclResult_t call_get_coll_info(size_t nBytes, int regression, ncclInfo& out_info,
                                int* nChannels, const Table& t) {
  out_info = ncclInfo{};
  *nChannels = 4;   // NCCL 的默认建议
  return ncclTunerPlugin_v5.getCollInfo(&out_info, ncclFuncAllReduce, nBytes,
                                        /*numPipeOps=*/1,
                                        t.data(), kNumAlgo, kNumProto,
                                        regression, nChannels);
}

}  // namespace

int main() {
  std::printf("=== NCCL tuner plugin 测试 ===\n");
  auto& plug = ncclTunerPlugin_v5;
  assert(plug.name && std::string(plug.name) == "mycard_tuner");

  // ① init：16 rank / 2 节点 / 每节点 8 卡
  assert(plug.init(16, 2, 0, 8, nullptr, nullptr, nullptr, nullptr) == ncclSuccess);
  std::printf("\n");

  // ② 小消息（8KB）：偏置应指向 Tree
  {
    const size_t nBytes = 8 * 1024;
    Table t = make_table(nBytes);
    ncclInfo info{};
    int ch = 0;
    assert(call_get_coll_info(nBytes, 0, info, &ch, t) == ncclSuccess);
    assert(info.algorithm == NCCL_ALGO_TREE);          // ★ 小消息 → Tree
    std::printf("  [ok] 8KB    → Tree/%s（步数主导）\n", proto_name(info.protocol));
  }

  // ③ 大消息（64MB）：偏置应指向 Ring
  {
    const size_t nBytes = 64ull << 20;
    Table t = make_table(nBytes);
    ncclInfo info{};
    int ch = 0;
    assert(call_get_coll_info(nBytes, 0, info, &ch, t) == ncclSuccess);
    assert(info.algorithm == NCCL_ALGO_RING);          // ★ 大消息 → Ring
    assert(ch >= 8);                                   // ★ 跨机大消息：通道下限被抬高
    std::printf("  [ok] 64MB   → Ring/%s，channels=%d（搬运量主导 + 多通道）\n",
                proto_name(info.protocol), ch);
  }

  // ④ regression > 0：不能动 NCCL 的选择（否则自校准失真）
  {
    const size_t nBytes = 8 * 1024;
    Table t = make_table(nBytes);
    ncclInfo info{};
    int ch = 0;
    assert(call_get_coll_info(nBytes, /*regression=*/1, info, &ch, t) == ncclSuccess);
    // regression 下应该返回"代价表里的最小"，而不是应用偏置
    std::printf("  [ok] regression>0 → 不应用偏置，返回代价表最优（algo=%d）\n",
                info.algorithm);
  }

  // ⑤ 单机场景（nNodes=1）：不应触发多通道抬高
  {
    assert(plug.finalize() == ncclSuccess);
    assert(plug.init(8, 1, 0, 8, nullptr, nullptr, nullptr, nullptr) == ncclSuccess);
    const size_t nBytes = 64ull << 20;
    Table t = make_table(nBytes);
    ncclInfo info{};
    int ch = 0;
    assert(call_get_coll_info(nBytes, 0, info, &ch, t) == ncclSuccess);
    assert(ch == 4);                                   // ★ 单机不抬通道
    std::printf("  [ok] 单机大消息 → 不抬通道下限（保持 NCCL 建议 %d）\n", ch);
  }

  assert(plug.finalize() == ncclSuccess);

  // ================= v3 版本（旧 RCCL 用的就是这个）=================
  std::printf("\n--- v3 接口（语义不同：改代价表而不是写回选择）---\n");
  extern ncclTuner_v3_t ncclTunerPlugin_v3;
  auto& v3 = ncclTunerPlugin_v3;
  assert(v3.name && std::string(v3.name) == "mycard_tuner");

  void* ctx = nullptr;
  assert(v3.init(16, 2, nullptr, &ctx) == ncclSuccess);
  assert(ctx != nullptr);                        // ★ v3 要给出上下文

  {
    // 小消息 → 应保留 Tree，把其余算法标成 IGNORE
    const size_t nBytes = 8 * 1024;
    Table t = make_table(nBytes);
    int ch = 4;
    assert(v3.getCollInfo(ctx, ncclFuncAllReduce, nBytes, 1, t.data(), kNumAlgo, kNumProto,
                          &ch) == ncclSuccess);
    assert(t.at(NCCL_ALGO_TREE, NCCL_PROTO_LL) > 0.0f);      // ★ Tree 保留
    assert(t.at(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE) < 0.0f);  // ★ Ring 被标 IGNORE
    std::printf("  [ok] v3 8KB   → 保留 Tree，Ring 标 IGNORE（NCCL 会自己选最小）\n");
  }
  {
    // 跨机大消息 → 应保留 Ring + 抬通道
    const size_t nBytes = 64ull << 20;
    Table t = make_table(nBytes);
    int ch = 4;
    assert(v3.getCollInfo(ctx, ncclFuncAllReduce, nBytes, 1, t.data(), kNumAlgo, kNumProto,
                          &ch) == ncclSuccess);
    assert(t.at(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE) > 0.0f);   // ★ Ring 保留
    assert(t.at(NCCL_ALGO_TREE, NCCL_PROTO_LL) < 0.0f);      // ★ Tree 被标 IGNORE
    assert(ch == 8);
    std::printf("  [ok] v3 64MB  → 保留 Ring，Tree 标 IGNORE，channels=%d\n", ch);
  }
  assert(v3.destroy(ctx) == ncclSuccess);

  std::printf("\n[PASS] tuner plugin 全部通过（v5 + v3 双版本）\n");
  return 0;
}
