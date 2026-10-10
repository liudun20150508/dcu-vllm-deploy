// mycard_tuner.cpp —— NCCL tuner plugin 骨架（接口版本 v5）
//
// 【这个文件回答的问题】
//   JD 职责 6 写的是"建立通信性能测试与分析体系，开发性能测试、数据分析、
//   问题定位及**自动化调优工具**"。NCCL 给"自动化调优"留的**官方挂载点**
//   就是 **tuner plugin** ——
//
//       集合通信每次启动时，NCCL 会自己算一张"代价表"：
//            collCostTable[算法][协议] = 预估耗时
//       然后 **tuner 有机会改写这个选择**。
//
//   所以"自动调优工具"不是在外面挂个脚本改环境变量，而是**在库内部按规则决策**。
//
// 【和 NET plugin 的分工】
//   NET plugin：解决"通信库认不认识这张网卡"         → 能不能跑
//   tuner plugin：解决"认得了之后走哪条最合适"        → 跑得多快
//
// 【怎么挂上去】
//   NCCL 用 dlopen + dlsym 找符号 ncclTunerPlugin_v5；
//   也可以用环境变量 NCCL_TUNER_PLUGIN 指定 .so 路径
//   （见 NCCL 源码 src/plugin/tuner/tuner_v5.cc 的 getNcclTuner_v5）
//
// 【这份代码的定位】
//   教学骨架：接口填满、每一步的**判断依据**写清楚，并用"量级模型"给出可运行的
//   选择逻辑。它不是生产实现 —— 每处简化都标了 [简化]。
//
// 依赖：nccl_tuner.h（NCCL 源码 src/include/plugin/ 下）
//      本目录自带一个 stub 版本，便于在没装 NCCL 的机器上验证编译。

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "nccl_tuner.h"

namespace {

// ---------------------------------------------------------------- 状态
struct TunerState {
  size_t nRanks{0};
  size_t nNodes{0};
  int rank{0};
  int ranksPerNode{0};
  int compCap{0};        // 计算能力（DCU 上是 gfx 代际的映射，NVIDIA 上是 SM）
  int collNetSupport{0}; // 有没有 SHARP/集合卸载
  int nvlsSupport{0};    // 有没有 NVLink SHARP
  bool inited{false};
};

TunerState g_state;

// 代价表里的"忽略"标记：这个 (algo, proto) 组合不可用
constexpr float kIgnore = -1.0f;

// -------------------------------------------------- ⚠️ 代价表的真实内存布局（关键）
// 接口签名里 collCostTable 是 float**，但 NCCL/RCCL 实际传进来的是**连续二维数组**
// float[numAlgo][numProto] 强转成 float**（见 src/plugin/tuner/tuner_v2.cc 里
// `float (*table)[NCCL_NUM_PROTOCOLS] = (float(*)[NCCL_NUM_PROTOCOLS])collCostTable`）。
// 所以必须按**扁平下标** a*numProto+p 访问；若当成"指针数组"写 table[a][p]，
// 会把两个 float 的位模式当成指针 → 空指针/野指针 → 段错误。
// （这正是本插件此前在真 RCCL 上 getCollInfo 崩溃的根因。）
static inline float* cost_flat(float** table) { return reinterpret_cast<float*>(table); }

// -------------------------------------------------- ① 按代价表选（基线）
// NCCL 已经算好了每种 (算法, 协议) 的预估成本，先取最小的 —— 这是"不动偏置"的行为。
bool pick_min_cost(float** table, int numAlgo, int numProto, int* outAlgo, int* outProto,
                   float* outCost) {
  float* flat = cost_flat(table);
  bool found = false;
  float best = 0.0f;
  int bestA = -1, bestP = -1;
  for (int a = 0; a < numAlgo; ++a) {
    for (int p = 0; p < numProto; ++p) {
      const float c = flat[a * numProto + p];
      if (c <= kIgnore) continue;          // 该组合不可用
      if (!found || c < best) {
        found = true;
        best = c;
        bestA = a;
        bestP = p;
      }
    }
  }
  if (found) {
    *outAlgo = bestA;
    *outProto = bestP;
    *outCost = best;
  }
  return found;
}

// -------------------------------------------------- ② 自己的修正规则
// 这里才是"调优"发生的地方。规则来自我在自己写的集合通信原型里量化出的结论：
//   · 小消息：**步数**主导  → 倾向步数少的算法（Tree：log N 步；Ring：2(N-1) 步）
//   · 大消息：**每步搬运量**主导 → 倾向搬运效率高的算法（Ring 有双向流水）
//   · 交叉点**随规模右移**：rank 越多，Tree 的优势区间越宽
//
// 判断规模用 nBytes（单次通信量），阈值随 rank 数缩放。
struct Bias {
  bool force_tree{false};
  bool force_ring{false};
  int  minChannels{0};
};

Bias compute_bias(const TunerState& st, size_t nBytes) {
  Bias b;
  // 交叉点的粗略模型：消息越小、rank 越少 → 越适合 Tree
  // [简化] 真实系统应该用实测定标出来的表，而不是这个公式。
  const double ranks = static_cast<double>(st.nRanks);
  const double cross = 256.0 * 1024.0 * (1.0 + ranks / 64.0);   // 256KB 起，随 rank 数上移

  if (static_cast<double>(nBytes) < cross) {
    b.force_tree = true;    // 小消息：步数主导
  } else {
    b.force_ring = true;    // 大消息：搬运量主导
  }

  // 跨机占比高的时候，多开几条通道更划算（把大象流摊开、避免单链路撞车）
  if (st.nNodes > 1 && nBytes >= 1u << 20) {
    b.minChannels = 8;
  }
  return b;
}

// -------------------------------------------------- ③ 在候选里找算法
int find_algo_by_name(int numAlgo, const char* want) {
  // NCCL 的 algo 枚举顺序是稳定的（TREE=0, RING=1, ...），
  // 这里按名字匹配，避免硬编码数字 —— 换 NCCL 版本也不会错。
  for (int a = 0; a < numAlgo; ++a) {
    if (algo_name(a) && std::strcmp(algo_name(a), want) == 0) return a;
  }
  return -1;
}

// 在某个算法下，挑代价最小的协议
bool pick_proto_of_algo(float** table, int numProto, int algo, int* outProto, float* outCost) {
  float* flat = cost_flat(table);
  bool found = false;
  float best = 0.0f;
  int bestP = -1;
  for (int p = 0; p < numProto; ++p) {
    const float c = flat[algo * numProto + p];
    if (c <= kIgnore) continue;
    if (!found || c < best) {
      found = true;
      best = c;
      bestP = p;
    }
  }
  if (found) {
    *outProto = bestP;
    *outCost = best;
  }
  return found;
}

}  // namespace

// ============================================================ 接口实现

// ---- 共用逻辑：两个版本都调用这几个 ----

static ncclResult_t tuner_core_init(size_t nRanks, size_t nNodes, int rank,
                                    int nRanksPerNode, int collNetSupport, int nvlsSupport) {
  g_state.nRanks = nRanks;
  g_state.nNodes = nNodes;
  g_state.rank = rank;
  g_state.ranksPerNode = nRanksPerNode;
  g_state.collNetSupport = collNetSupport;
  g_state.nvlsSupport = nvlsSupport;
  g_state.inited = true;
  std::printf("[tuner] init: nRanks=%zu nNodes=%zu rank=%d ranksPerNode=%d "
              "collNet=%d nvls=%d\n",
              nRanks, nNodes, rank, nRanksPerNode, collNetSupport, nvlsSupport);
  return ncclSuccess;
}

// 核心决策：从 NCCL 的代价表出发，再叠加自己的偏置
static ncclResult_t tuner_core_decide(void* collInfo, size_t nBytes,
                                      float** collCostTable, int numAlgo, int numProto,
                                      int regression, int* nChannels) {
  if (!collInfo || !collCostTable) return ncclInvalidArgument;
  auto* info = static_cast<ncclInfo*>(collInfo);

  int algo = 0, proto = 0;
  float cost = 0.0f;
  if (!pick_min_cost(collCostTable, numAlgo, numProto, &algo, &proto, &cost)) {
    return ncclSuccess;   // 表全不可用 → 交给 NCCL 自己定
  }

  // regression 期间绝不能改（否则 NCCL 自校准失真）
  if (regression > 0) {
    info->algorithm = algo;
    info->protocol = proto;
    return ncclSuccess;
  }

  const Bias bias = compute_bias(g_state, nBytes);
  int biased = algo;
  if (bias.force_tree) {
    const int t = find_algo_by_name(numAlgo, "Tree");
    if (t >= 0) biased = t;
  } else if (bias.force_ring) {
    const int r = find_algo_by_name(numAlgo, "Ring");
    if (r >= 0) biased = r;
  }
  if (biased != algo) {
    int p2 = 0;
    float c2 = 0.0f;
    if (pick_proto_of_algo(collCostTable, numProto, biased, &p2, &c2)) {
      algo = biased;
      proto = p2;
      cost = c2;
    }
  }

  info->algorithm = algo;
  info->protocol = proto;
  if (nChannels && bias.minChannels > 0 && *nChannels < bias.minChannels) {
    *nChannels = bias.minChannels;
  }

  std::printf("[tuner] nBytes=%zu nRanks=%zu -> algo=%d proto=%d cost=%.2f channels=%d%s\n",
              nBytes, g_state.nRanks, algo, proto, cost, nChannels ? *nChannels : -1,
              (bias.force_tree ? " (bias=Tree)"
                               : (bias.force_ring ? " (bias=Ring)" : "")));
  return ncclSuccess;
}

// ============================================ v5 版本（新 NCCL：2.23+）

static ncclResult_t v5_init(size_t nRanks, size_t nNodes, int rank, int nRanksPerNode,
                            size_t* compCap, int* collNetSupport, int* nvlsSupport,
                            int* /*numAlgos*/) {
  (void)compCap;
  return tuner_core_init(nRanks, nNodes, rank, nRanksPerNode,
                         collNetSupport ? *collNetSupport : 0,
                         nvlsSupport ? *nvlsSupport : 0);
}

static ncclResult_t v5_get_coll_info(void* collInfo, ncclFunc_t /*collType*/, size_t nBytes,
                                     int /*numPipeOps*/, float** collCostTable, int numAlgo,
                                     int numProto, int regression, int* nChannels) {
  return tuner_core_decide(collInfo, nBytes, collCostTable, numAlgo, numProto, regression,
                           nChannels);
}

static ncclResult_t v5_finalize(void) {
  std::printf("[tuner] finalize (v5)\n");
  g_state = TunerState{};
  return ncclSuccess;
}

// ★ 用 extern "C" { } 块而不是「extern "C" 变量 = ...」：
//   GCC 8.5 会把后者报成 "已初始化却又声明为 extern"（clang 不报）。
//   又是一个「本机过、真机不过」的例子。
extern "C" {
ncclTuner_v5_t ncclTunerPlugin_v5 = {
    "mycard_tuner", v5_init, v5_get_coll_info, v5_finalize,
};
}

// ============================================ v3 版本（旧 NCCL/RCCL：曙光这套就是）
// ★ 这就是"版本对齐"的实操：同一个决策逻辑，包一层适配旧接口。
//   曙光那套 RCCL 只认到 ncclTunerPlugin_v3（用 strings librccl.so 查到的）。
//
// 【结构出处】RCCL 源码 src/include/plugin/tuner/tuner_v3.h
//   该头文件注释原文：
//     "collCostTable: ... NCCL core sets ignored algo/proto cost table entries
//      to -1.0 (NCCL_ALGO_PROTO_IGNORE)."
//   ⇒ 这就是 v3 的约定：**不是写回选择，而是改代价表**。下面的实现照此办理。

static constexpr int kContextMagic = 0x4d594344;   // 'MYCD'

static ncclResult_t v3_init(size_t nRanks, size_t nNodes, ncclDebugLogger_t /*logfn*/,
                            void** context) {
  const ncclResult_t rc = tuner_core_init(nRanks, nNodes, /*rank=*/0,
                                          /*nRanksPerNode=*/0, 0, 0);
  if (context) {
    // v3 要我们自己挂一个上下文；这里就把状态地址递出去。
    static int context_storage = kContextMagic;
    *context = &context_storage;
  }
  return rc;
}

// ★ v3 与 v5 的**语义差异**（这是关键，不是简单改签名）：
//   v5：回调带 collInfo，**直接写回 algorithm / protocol**
//   v3：回调**没有 collInfo**，约定是**修改 collCostTable** ——
//       把不想选的 (算法,协议) 标成 NCCL_ALGO_PROTO_IGNORE，让 NCCL 自己取最小。
static ncclResult_t v3_get_coll_info(void* /*context*/, ncclFunc_t /*collType*/, size_t nBytes,
                                     int /*numPipeOps*/, float** collCostTable, int numAlgo,
                                     int numProto, int* nChannels) {
  if (!collCostTable) return ncclInvalidArgument;

  const Bias bias = compute_bias(g_state, nBytes);
  const bool want_tree = bias.force_tree;
  const bool want_ring = bias.force_ring;

  if (want_tree || want_ring) {
    const char* keep = want_tree ? "Tree" : "Ring";
    float* flat = cost_flat(collCostTable);
    for (int a = 0; a < numAlgo; ++a) {
      const char* nm = algo_name(a);
      if (nm && std::strcmp(nm, keep) == 0) continue;   // 保留想用的那个算法
      for (int p = 0; p < numProto; ++p) {
        flat[a * numProto + p] = -1.0f;                 // ★ 其余算法全部标为不可用
      }
    }
  }

  if (nChannels && bias.minChannels > 0 && *nChannels < bias.minChannels) {
    *nChannels = bias.minChannels;
  }

  std::printf("[tuner/v3] nBytes=%zu nRanks=%zu -> 保留 %s，其余标记 IGNORE，channels=%d\n",
              nBytes, g_state.nRanks, want_tree ? "Tree" : (want_ring ? "Ring" : "全部"),
              nChannels ? *nChannels : -1);
  return ncclSuccess;
}

static ncclResult_t v3_destroy(void* /*context*/) {
  std::printf("[tuner] destroy (v3)\n");
  g_state = TunerState{};
  return ncclSuccess;
}

extern "C" {
ncclTuner_v3_t ncclTunerPlugin_v3 = {
    "mycard_tuner", v3_init, v3_get_coll_info, v3_destroy,
};
}
