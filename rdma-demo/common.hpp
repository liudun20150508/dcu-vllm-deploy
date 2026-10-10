// common.hpp —— RDMA Verbs 最小可用封装（C++17）
//
// 这份代码的目的不是"能跑就行"，而是把 RDMA 的四个操作
// （SEND / RECV / WRITE / READ）从头到尾走一遍，看清每一样东西
// 到底在哪里起作用。
//
// 软件栈位置：
//     应用
//       ↓ 本文件用到的 verbs API（ibv_*）
//     libibverbs（用户态驱动库）
//       ↓
//     内核驱动（mlx5_ib / rdma_rxe …）
//       ↓
//     网卡（真卡，或 SoftRoCE 的软件实现）
//
// 一句话记住这套 API 的骨架：
//     open_device → alloc_pd → reg_mr → create_cq → create_qp
//       → modify_qp(RESET → INIT → RTR → RTS) → post_send/post_recv → poll_cq
//
// 其中只有两步需要**对端**的信息，所以数据面之前必须先有控制面：
//     RTR 需要 (目的 QPN, 目的 GID)
//     单边 WRITE/READ 需要 (目的虚拟地址、rkey)
// 这就是为什么所有 RDMA 程序开头都有一段 TCP 握手。

#pragma once

#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// ── LID 宽度适配 ────────────────────────────────────────────────────────
// 标准 IB 的 LID 是 16 位（uint16_t）。曙光自研网卡 shca 用的是 17 位 LID，
// 其定制版 verbs.h 把 ibv_port_attr::lid 和 ibv_ah_attr::dlid 从 uint16_t
// 换成了 struct u17 { u16 low; u16 high; }（见 infiniband/shca_17b_types.h）。
// 下面用条件编译，让同一份代码在标准 IB 与 shca 两种 ABI 上都能编过。
#if defined(__has_include)
#  if __has_include(<infiniband/shca_17b_types.h>)
#    define RDMA_LID_IS_U17 1
#  endif
#endif
#ifndef RDMA_LID_IS_U17
#  define RDMA_LID_IS_U17 0
#endif

#if RDMA_LID_IS_U17
   // 读 LID（取低 16 位；自研卡实际 17 位，本程序够用）
#  define RDMA_LID_READ(field) static_cast<std::uint32_t>((field).low)
   // 写 LID：把 17 位拆进 low/high 两段
#  define RDMA_LID_WRITE(field, v)                                    \
      do {                                                            \
        (field).low = static_cast<decltype((field).low)>((v) & 0xffffu);      \
        (field).high = static_cast<decltype((field).high)>(((v) >> 16) & 0x1u); \
      } while (0)
   // 清 LID：走 GID 时 dlid 不参与寻址
#  define RDMA_LID_ZERO(field) do { (field).low = 0; (field).high = 0; } while (0)
#else
#  define RDMA_LID_READ(field) static_cast<std::uint32_t>(field)
#  define RDMA_LID_WRITE(field, v) do { (field) = static_cast<decltype(field)>(v); } while (0)
#  define RDMA_LID_ZERO(field) do { (field) = 0; } while (0)
#endif

namespace rdma {

// 控制面要交换的全部信息 —— 正好对应"QP 建链要三样，单边操作要两样"
struct ConnInfo {
  std::uint32_t qpn{};        // 对端 QP 号        → RTR 时填 ah_attr.dest_qp_num
  std::uint32_t rkey{};       // 对端内存注册键    → 单边 WRITE/READ 放进 RETH
  std::uint64_t addr{};       // 对端缓冲区地址    → 单边 WRITE/READ 的目标
  std::uint8_t gid[16]{};     // 对端 GID          → RTR 时填 ah_attr.grh.dgid
  std::uint32_t lid{};        // 对端 LID          → 纯 IB 链路层寻址用 ah_attr.dlid
                              //   （RoCEv2 走 GID，这个字段不参与）
} __attribute__((packed));

void fatal(const std::string& what);
void check(int rc, const std::string& what);

// ---------------------------------------------------------------- 控制面
// 建 QP 之前要先跟对端换 ConnInfo。用最朴素的 TCP 完成这件事 ——
// 真实系统里这一步是 RDMA_CM（rdma_cm）或者 NCCL 自己的 bootstrap。
class ControlChannel {
 public:
  ControlChannel(bool is_server, const std::string& host, std::uint16_t port);
  ~ControlChannel();
  ControlChannel(const ControlChannel&) = delete;
  ControlChannel& operator=(const ControlChannel&) = delete;

  // 双向交换：我发我的，收对端的
  ConnInfo exchange(const ConnInfo& mine);

 private:
  int fd_{-1};
};

// ---------------------------------------------------------------- 数据面
class RdmaEndpoint {
 public:
  // buf_size: 本地注册的工作缓冲区大小
  // gid_index: RoCEv2 通常用 1；纯 IB 用 0。可用 `show_gids` 确认
  // external_buf: 非空时**不自己分配**，直接把这块外部内存注册成 MR。
  //               这是 GPUDirect RDMA 的入口 —— 传一个 hipMalloc 的 DCU 显存指针进来，
  //               网卡就能对显存做单边 RDMA，数据不过 CPU/主机内存。
  RdmaEndpoint(const std::string& dev_name, std::size_t buf_size,
               int gid_index = 1, int tx_depth = 16, int rx_depth = 16,
               void* external_buf = nullptr);
  ~RdmaEndpoint();
  RdmaEndpoint(const RdmaEndpoint&) = delete;
  RdmaEndpoint& operator=(const RdmaEndpoint&) = delete;

  void* buf() const { return buf_; }
  std::size_t buf_size() const { return buf_size_; }

  ConnInfo local_info() const;

  // 把 QP 从 RESET 一路推到 RTS —— 走完这四步才能收发
  // （path MTU 会自动取本端 port 的 active_mtu）
  void connect_to(const ConnInfo& peer, int port_num = 1);

  // --- 双边：对端必须**预先** post 好接收缓冲，否则 RNR 重试/失败 ---
  void post_send(const void* local, std::size_t len);
  void post_recv(void* local, std::size_t len);

  // --- 单边：对端 CPU 完全不参与，也不需要预先 post ---
  void post_write(const void* local, std::size_t len, const ConnInfo& peer);
  void post_read(void* local, std::size_t len, const ConnInfo& peer);

  // 等一个完成事件，返回它属于哪个 WR（wr_id）
  std::uint64_t wait_one();

 private:
  void to_init();
  void to_rtr(const ConnInfo& peer, int port_num, int mtu_log);
  void to_rts();

  ibv_context* ctx_{nullptr};
  ibv_pd* pd_{nullptr};
  ibv_mr* mr_{nullptr};
  ibv_cq* cq_{nullptr};
  ibv_qp* qp_{nullptr};

  void* buf_{nullptr};
  bool owns_buf_{true};   // false = 用的是外部内存（显存），析构时不能 free
  std::size_t buf_size_{0};
  int gid_index_{0};
  ibv_gid local_gid_{};
  std::uint32_t lid_{0};
  bool use_gid_{false};   // true=RoCEv2(GID 寻址) / false=纯 IB(LID 寻址)，构造时按 link_layer 自适应
  int tx_depth_{0};
  int rx_depth_{0};
  std::uint32_t psn_{0};
  std::uint64_t next_wr_id_{1};
};

}  // namespace rdma
