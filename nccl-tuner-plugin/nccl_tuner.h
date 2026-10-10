// nccl_tuner.h —— NCCL tuner plugin 接口的最小 stub（仅用于编译验证）
//
// 【为什么要这个文件】
//   真实环境里这个头文件在 NCCL 源码的 src/include/plugin/nccl_tuner.h。
//   本机（Mac）没装 NCCL，为了能**验证插件本身能编过**，这里给出一个
//   字段/签名一致的 stub。
//
//   ⚠️ 真要挂到 NCCL 上时，**必须改用 NCCL 源码里的真头文件**，
//      否则结构体布局不一致会直接段错误。
//
// 对应 NCCL 版本：2.2x（tuner 接口 v5）

#pragma once

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------- 基础类型
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;

typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncSendRecv = 5,
  ncclFuncSend = 6,
  ncclFuncRecv = 7,
  ncclNumFuncs = 8
} ncclFunc_t;

typedef enum {
  NCCL_ALGO_TREE = 0,
  NCCL_ALGO_RING = 1,
  NCCL_ALGO_COLLNET_DIRECT = 2,
  NCCL_ALGO_COLLNET_CHAIN = 3,
  NCCL_ALGO_NVLS = 4,
  NCCL_ALGO_NVLS_TREE = 5,
  NCCL_NUM_ALGORITHMS = 6,
  NCCL_ALGO_PROTO_IGNORE = -1
} ncclAlgo_t;

typedef enum {
  NCCL_PROTO_LL = 0,
  NCCL_PROTO_LL128 = 1,
  NCCL_PROTO_SIMPLE = 2,
  NCCL_NUM_PROTOCOLS = 3
} ncclProto_t;

// ---------------------------------------------------------------- ncclInfo
// tuner 通过 void* 拿到它，需要写回 algorithm / protocol（nChannels 走单独参数）。
// [简化] 真实 ncclInfo 字段更多，但 tuner 只碰这几个。
typedef struct {
  ncclFunc_t coll;
  const void* sendbuff;
  void* recvbuff;
  size_t sendcount;
  size_t recvcount;
  int root;
  int nChannels;
  int algorithm;
  int protocol;
} ncclInfo;

// ---------------------------------------------------------------- 日志回调
typedef void (*ncclDebugLogger_t)(int level, unsigned long flags, const char* file,
                                  int line, const char* fmt, ...);

// ---------------------------------------------------------------- v3 结构（旧版 NCCL/RCCL 用）
// 依据：NCCL 源码 src/plugin/tuner/tuner_v3.cc 里对 ncclTuner_v3_t 的调用方式反推。
//   · init(nRanks, nNodes, logfn, context)  —— 有 context 出参
//   · getCollInfo(context, collType, nBytes, numPipeOps, table, nAlgo, nProto, nChannels)
//     —— ★ 没有 regression、没有 numAlgo 入参（v5 才有）
//   · destroy(context)                       —— 叫 destroy 不叫 finalize
//
// ⚠️ collCostTable 的签名是 float**，但实为**连续二维数组 float[numAlgo][numProto]**
//    强转成 float**（见 NCCL src/plugin/tuner/tuner_v2.cc：
//    `float (*table)[NCCL_NUM_PROTOCOLS] = (float(*)[NCCL_NUM_PROTOCOLS])collCostTable`）。
//    实现里必须按扁平下标 a*numProto+p 访问，不能当"指针数组"写 table[a][p]。
typedef struct {
  const char* name;
  ncclResult_t (*init)(size_t nRanks, size_t nNodes, ncclDebugLogger_t logfn, void** context);
  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo,
                              int numProto, int* nChannels);
  ncclResult_t (*destroy)(void* context);
} ncclTuner_v3_t;

// ---------------------------------------------------------------- tuner 回调签名
typedef ncclResult_t (*ncclTunerInit_t)(size_t nRanks, size_t nNodes, int rank,
                                        int nRanksPerNode, size_t* compCap,
                                        int* collNetSupport, int* nvlsSupport,
                                        int* numAlgos);

typedef ncclResult_t (*ncclTunerGetCollInfo_t)(void* collInfo, ncclFunc_t collType,
                                               size_t nBytes, int numPipeOps,
                                               float** collCostTable, int numAlgo,
                                               int numProto, int regression,
                                               int* nChannels);

typedef ncclResult_t (*ncclTunerFinalize_t)(void);

typedef struct {
  const char* name;
  ncclTunerInit_t init;
  ncclTunerGetCollInfo_t getCollInfo;
  ncclTunerFinalize_t finalize;
} ncclTuner_v5_t;

// ---------------------------------------------------------------- 辅助
// 把算法枚举翻译成名字。真实 NCCL 里没有这个函数（枚举顺序是约定），
// 放在 stub 里是为了让插件能"按名字"找算法，而不是硬编码数字。
inline const char* algo_name(int a) {
  switch (a) {
    case NCCL_ALGO_TREE: return "Tree";
    case NCCL_ALGO_RING: return "Ring";
    case NCCL_ALGO_COLLNET_DIRECT: return "CollNetDirect";
    case NCCL_ALGO_COLLNET_CHAIN: return "CollNetChain";
    case NCCL_ALGO_NVLS: return "NVLS";
    case NCCL_ALGO_NVLS_TREE: return "NVLSTree";
    default: return nullptr;
  }
}

inline const char* proto_name(int p) {
  switch (p) {
    case NCCL_PROTO_LL: return "LL";
    case NCCL_PROTO_LL128: return "LL128";
    case NCCL_PROTO_SIMPLE: return "Simple";
    default: return "?";
  }
}
