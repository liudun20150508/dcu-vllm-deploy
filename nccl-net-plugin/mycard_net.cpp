// mycard_net.cpp —— NCCL NET plugin 骨架（教学版，接口版本 v12）
//
// 【这是干什么的】
//   NCCL 通过一个叫 NET plugin 的接口把"网卡"抽象出来。要接入一张 NCCL
//   还不支持的网卡（自研/国产），标准做法**不是改 NCCL 核心**，而是：
//       ① 实现下面这张函数表 ncclNet_v12_t
//       ② 编成 libnccl-net.so
//       ③ NCCL 侧 export NCCL_NET_PLUGIN=libnccl-net.so（或放进默认搜索路径）
//   核心一行不用动 —— 这就是"适配"这两个字的技术含义。
//
// 【NCCL 怎么找到它】
//   NCCL 用 dlopen + dlsym 找两个符号：
//       ncclNetPlugin_v12       ← 本文件导出的就是这个
//       ncclCollNetPlugin_v12   ← 集合通信卸载（SHARP 那类），可选
//   （见 src/plugin/net/net_v12.cc 里的 getNcclNet_v12）
//
// 【这份代码的定位】
//   教学骨架：函数表填满、每个函数的**职责**写清楚，并用 TCP 给出可运行的
//   最小实现。它不是生产实现 —— 每处简化都标了 [简化]。
//
// 依赖：nccl_net.h（在 NCCL 源码的 src/include/plugin/ 下）

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nccl_net.h"

#define MYCARD_NAME "MyCard"
#define MYCARD_CTRL_PORT 18520

// ============================================================ 小工具

static ncclDebugLogger_t g_log;  // 由 init() 注入，插件靠它往 NCCL 日志里写

static void log_info(const char* fmt, ...) {
  if (!g_log) return;
  va_list args;
  va_start(args, fmt);
  g_log(NCCL_LOG_INFO, NCCL_NET, __FILE__, __LINE__, fmt, args);
  va_end(args);
}

template <typename T>
static ncclResult_t nccl_ok(T** p, const char* what) {
  if (*p) return ncclSuccess;
  log_info("MyCard: %s failed", what);
  return ncclSystemError;
}

// 一次完整的 TCP 写/读（控制面和数据面都用它）
static bool tcp_write_all(int fd, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n) {
    ssize_t k = ::send(fd, c, n, 0);
    if (k <= 0) return false;
    c += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}
static bool tcp_read_all(int fd, void* p, size_t n) {
  char* c = static_cast<char*>(p);
  while (n) {
    ssize_t k = ::recv(fd, c, n, 0);
    if (k <= 0) return false;
    c += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

// ============================================================ 内部对象
// [简化] 生产实现里 sendComm/recvComm 会带上 QP、CQ、MR 这些 RDMA 资源；
//        这里只有 TCP 连接，因为 TCP 不需要注册内存。

struct SendComm {
  int fd{-1};
  int dev{-1};
};
struct RecvComm {
  int fd{-1};
  int dev{-1};
};
struct ListenComm {
  int fd{-1};       // 监听 socket
  int dev{-1};
  uint16_t port{0};
};
// 请求：NCCL 用 test() 轮询它，done=1 表示完成
struct Request {
  std::atomic<bool> done{false};
  std::thread worker;
  int bytes{0};

  ~Request() { if (worker.joinable()) worker.join(); }
};
struct MrHandle {
  void* addr{nullptr};
  size_t size{0};
  int type{0};  // NCCL_PTR_HOST / NCCL_PTR_CUDA / NCCL_PTR_DMABUF
};

// ============================================================ ① 生命周期

static ncclResult_t mc_init(void** ctx, uint64_t commId,
                            ncclNetCommConfig_v12_t* config,
                            ncclDebugLogger_t logFunction,
                            ncclProfilerCallback_t /*profFunction*/) {
  g_log = logFunction;  // 拿到日志句柄，后面所有日志都走它
  *ctx = nullptr;       // 没有全局状态要保存时给 nullptr 就行
  log_info("MyCard: init (commId=%lu, tc=%d)", commId,
           config ? config->trafficClass : -1);
  // 生产实现在这里做：打开设备、建线程池、设置 flow steering / DSCP 等
  return ncclSuccess;
}

static ncclResult_t mc_devices(int* ndev) {
  // 告诉 NCCL "我有几张网卡"。NCCL 会按 railId/dev 给每张卡分配 rank。
  // [简化] 这里报 1 张。真实实现要枚举 PCI 设备并与 GPU 做亲和性匹配。
  *ndev = 1;
  log_info("MyCard: devices -> %d", *ndev);
  return ncclSuccess;
}

// ★ 这个函数是整个插件里最关键的一个 ——
//   它是插件**向 NCCL 声明自己能力**的地方，NCCL 后面所有决策都基于它。
static ncclResult_t mc_getProperties(int dev, ncclNetProperties_v12_t* props) {
  std::memset(props, 0, sizeof(*props));
  props->name = const_cast<char*>(MYCARD_NAME);

  // ptrSupport 决定"这块网卡能不能直接访问哪些内存"：
  //   NCCL_PTR_HOST   —— 能访问主机内存（基础能力）
  //   NCCL_PTR_CUDA   —— 能直接访问 GPU 显存  ← **GPUDirect RDMA 就在这一行**
  //   NCCL_PTR_DMABUF —— 能访问 DMA-BUF（容器/异构场景）
  // 如果这里不报 CUDA，NCCL 就只能走"显存→主机内存→网卡"的绕路，
  // 性能会差一个量级。所以适配一张新卡，第一件事就是确认它能报什么。
  props->ptrSupport = NCCL_PTR_HOST;
  // [简化] 真卡上这里应该是 NCCL_PTR_HOST | NCCL_PTR_CUDA

  props->speed = 100000;    // 端口速率 (Mbps) —— NCCL 用它算带宽预算、选算法
  props->port = 1;
  props->latency = 1.0;     // 网络延迟 (us) —— 同上
  props->maxComms = 1024;   // 最多能建多少条连接
  props->maxRecvs = 1;      // 一次能提交几个分组接收
  props->regIsGlobal = 0;   // regMr 是否与 comm 无关
  props->forceFlush = 0;    // 接收后是否强制 flush
  props->netDeviceType = NCCL_NET_DEVICE_HOST;  // 无硬件卸载
  props->netDeviceVersion = 0;
  props->maxP2pBytes = SIZE_MAX;   // 单次点对点传输上限
  props->maxCollBytes = SIZE_MAX;
  props->maxMultiRequestSize = 1;
  props->railId = static_cast<int16_t>(dev);  // 多 rail 网络里的轨道号
  props->planeId = 0;                          // 多平面网络里的平面号
  return ncclSuccess;
}

// ============================================================ ② 建链
// RDMA 里这三步交换的是 QPN/GID/rkey；TCP 里交换的是端口号。
// 但调用流程完全一样，NCCL 不关心你底下是什么。

static ncclResult_t mc_listen(void* /*ctx*/, int dev, void* handle,
                              void** listenComm) {
  auto* lc = new ListenComm();
  lc->dev = dev;
  lc->fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (lc->fd < 0) { delete lc; return ncclSystemError; }
  int one = 1;
  ::setsockopt(lc->fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = INADDR_ANY;
  a.sin_port = 0;  // 让内核挑一个空闲端口
  if (::bind(lc->fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 ||
      ::listen(lc->fd, 4) != 0) {
    ::close(lc->fd);
    delete lc;
    return ncclSystemError;
  }
  socklen_t len = sizeof(a);
  ::getsockname(lc->fd, reinterpret_cast<sockaddr*>(&a), &len);
  lc->port = ntohs(a.sin_port);

  // handle 是给对端用的"联系方式"。RDMA 下塞 QPN+GID+rkey，
  // 这里塞 (dev, port)。
  auto* h = static_cast<uint64_t*>(handle);
  h[0] = static_cast<uint64_t>(dev);
  h[1] = lc->port;
  *listenComm = lc;
  return ncclSuccess;
}

static ncclResult_t mc_connect(void* /*ctx*/, int dev, void* handle,
                               void** sendComm,
                               ncclNetDeviceHandle_v12_t** sendDevComm) {
  auto* h = static_cast<uint64_t*>(handle);
  // [简化] 真实实现要按 GID/IP 连过去；这里只用端口，连本地。
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<uint16_t>(h[1]));
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);

  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  for (int i = 0; i < 200; ++i) {
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) break;
    ::usleep(20000);
  }
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  auto* sc = new SendComm{fd, dev};
  *sendComm = sc;
  *sendDevComm = nullptr;  // 没有网卡侧卸载就返回 nullptr
  return ncclSuccess;
}

static ncclResult_t mc_accept(void* listenComm, void** recvComm,
                              ncclNetDeviceHandle_v12_t** recvDevComm) {
  auto* lc = static_cast<ListenComm*>(listenComm);
  int fd = ::accept(lc->fd, nullptr, nullptr);
  if (fd < 0) return ncclSystemError;
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  *recvComm = new RecvComm{fd, lc->dev};
  *recvDevComm = nullptr;
  return ncclSuccess;
}

// ============================================================ ③ 内存注册
// 这一步是 RDMA 的地基（ibv_reg_mr：pin 页 + 生成 lkey/rkey）。
// TCP 不需要注册，但**接口必须实现** —— 而且这是暴露"能力边界"的地方：
//   NCCL 会根据 getProperties 里报的 ptrSupport 决定传什么指针进来。

static ncclResult_t mc_regMr(void* /*comm*/, void* data, size_t size, int type,
                             void** mhandle) {
  auto* mh = new MrHandle{data, size, type};
  *mhandle = mh;
  // [简化] 真卡上这里调 ibv_reg_mr。
  // 如果 type == NCCL_PTR_CUDA，说明 NCCL 在要求"直接注册显存"——
  // 只有 getProperties 报了 NCCL_PTR_CUDA 它才会这么传。
  return ncclSuccess;
}

static ncclResult_t mc_regMrDmaBuf(void* /*comm*/, void* /*data*/, size_t /*size*/,
                                   int /*type*/, uint64_t /*offset*/, int /*fd*/,
                                   void** /*mhandle*/) {
  // 容器场景（显存以 DMA-BUF 形式暴露）才走这条路。
  // [简化] 报"不支持"，NCCL 会回退到 regMr。
  return ncclInternalError;
}

static ncclResult_t mc_deregMr(void* /*comm*/, void* mhandle) {
  delete static_cast<MrHandle*>(mhandle);
  return ncclSuccess;
}

// ============================================================ ④ 数据传输
// 注意：isend / irecv **都是异步的** —— 提交后立刻返回 request，
// NCCL 用 test() 轮询完成。这个异步语义是通信库能和计算重叠的前提。

static ncclResult_t mc_isend(void* sendComm, void* data, size_t size, int /*tag*/,
                             void* /*mhandle*/, void* /*phandle*/, void** request) {
  auto* sc = static_cast<SendComm*>(sendComm);
  auto* req = new Request();
  req->bytes = static_cast<int>(size);
  int fd = sc->fd;
  // [简化] 每次开一个线程发。生产实现会用线程池 + 环形缓冲，
  //        并且把 request 和 WR（ibv_post_send 的返回）对应起来。
  req->worker = std::thread([fd, data, size, req]() {
    uint64_t hdr = size;
    if (!tcp_write_all(fd, &hdr, sizeof(hdr)) ||
        !tcp_write_all(fd, data, size)) {
      log_info("MyCard: isend failed");
    }
    req->done = true;   // ← NCCL 的 test() 就看这个
  });
  *request = req;
  return ncclSuccess;
}

static ncclResult_t mc_irecv(void* recvComm, int n, void** data, size_t* sizes,
                             int* /*tags*/, void** /*mhandles*/, void** /*phandles*/,
                             void** request) {
  auto* rc = static_cast<RecvComm*>(recvComm);
  if (n != 1) return ncclInternalError;  // [简化] 只支持单段接收
  auto* req = new Request();
  req->bytes = static_cast<int>(sizes[0]);
  int fd = rc->fd;
  auto* dst = static_cast<char*>(data[0]);
  size_t want = sizes[0];
  req->worker = std::thread([fd, dst, want, req]() {
    // 协议：先来 8 字节长度，再是数据。接收方按**自己声明的长度**收并校验 ——
    // 不能盲信对端给的长度（第一版就是这么写的，等于把缓冲区大小交给对端决定）。
    uint64_t hdr = 0;
    if (!tcp_read_all(fd, &hdr, sizeof(hdr)) || hdr != want ||
        !tcp_read_all(fd, dst, want)) {
      log_info("MyCard: irecv failed (expected %zu, peer says %lu)", want, hdr);
    }
    req->done = true;
  });
  *request = req;
  return ncclSuccess;
}

static ncclResult_t mc_iflush(void* /*recvComm*/, int /*n*/, void** /*data*/, int* /*sizes*/,
                              void** /*mhandles*/, void** request) {
  // "flush" 是给**有网卡侧卸载**的设备用的（数据已经在卡上，要保证可见）。
  // 纯软件路径不需要 —— 返回一个已完成的 request 即可。
  auto* req = new Request();
  req->done = true;
  *request = req;
  return ncclSuccess;
}

static ncclResult_t mc_test(void* request, int* done, int* sizes) {
  auto* req = static_cast<Request*>(request);
  if (req->done.load()) {
    *done = 1;
    if (sizes) sizes[0] = req->bytes;
    // [简化] 这里没释放 request —— 真实实现要等 NCCL 不再引用后再 delete，
    //        或者由 closeXxx 统一回收。
  } else {
    *done = 0;
  }
  return ncclSuccess;
}

// ============================================================ ⑤ 收尾
// close 的配对关系：connect ↔ closeSend，accept ↔ closeRecv，listen ↔ closeListen

static ncclResult_t mc_closeSend(void* sendComm) {
  auto* sc = static_cast<SendComm*>(sendComm);
  if (sc) { if (sc->fd >= 0) ::close(sc->fd); delete sc; }
  return ncclSuccess;
}
static ncclResult_t mc_closeRecv(void* recvComm) {
  auto* rc = static_cast<RecvComm*>(recvComm);
  if (rc) { if (rc->fd >= 0) ::close(rc->fd); delete rc; }
  return ncclSuccess;
}
static ncclResult_t mc_closeListen(void* listenComm) {
  auto* lc = static_cast<ListenComm*>(listenComm);
  if (lc) { if (lc->fd >= 0) ::close(lc->fd); delete lc; }
  return ncclSuccess;
}

// ============================================================ ⑥ 可选能力
// 这几个对应"网卡侧卸载"，纯软件实现可以一律报不支持。

static ncclResult_t mc_getDeviceMr(void* /*comm*/, void* /*mhandle*/,
                                   void** dptr_mhandle) {
  *dptr_mhandle = nullptr;   // 没有设备侧 MR
  return ncclSuccess;
}
static ncclResult_t mc_irecvConsumed(void* /*recvComm*/, int /*n*/, void* /*request*/) {
  return ncclSuccess;        // 没有网卡内存时无需通知
}
static ncclResult_t mc_makeVDevice(int* /*d*/, ncclNetVDeviceProps_v12_t* /*props*/) {
  return ncclInternalError;  // 虚拟设备（多卡聚合）暂不支持
}
static ncclResult_t mc_finalize(void* /*ctx*/) {
  log_info("MyCard: finalize");
  return ncclSuccess;
}
static ncclResult_t mc_setNetAttr(void* /*ctx*/, ncclNetAttr_v12_t* /*attr*/) {
  // 让 NCCL 告诉插件"这次用的是什么 op/algo/proto"，
  // 插件可以据此调整队列数、flow 等。不需要就留空。
  return ncclSuccess;
}

// ============================================================ ⑦ 导出
// NCCL 用 dlsym 找这个符号（见 src/plugin/net/net_v12.cc 的 getNcclNet_v12）
// ★ 用 extern "C" { } 块而不是「extern "C" 变量 = ...」：
//   GCC 会把后者报成 "已初始化却又声明为 extern"（与 tuner 插件同一处坑）。
extern "C" {
ncclNet_v12_t ncclNetPlugin_v12 = {
    MYCARD_NAME,
    mc_init,
    mc_devices,
    mc_getProperties,
    mc_listen,
    mc_connect,
    mc_accept,
    mc_regMr,
    mc_regMrDmaBuf,
    mc_deregMr,
    mc_isend,
    mc_irecv,
    mc_iflush,
    mc_test,
    mc_closeSend,
    mc_closeRecv,
    mc_closeListen,
    mc_getDeviceMr,
    mc_irecvConsumed,
    mc_makeVDevice,
    mc_finalize,
    mc_setNetAttr,
};
}  // extern "C"
