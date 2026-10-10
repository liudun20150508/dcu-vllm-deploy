// common.cpp —— 上面那些声明的实现
//
// 读这份代码时建议按两个问题走：
//   Q1「这一步需要谁的参与？」  → 区分双边 / 单边
//   Q2「这一步的信息从哪来？」  → 区分控制面 / 数据面

#include "common.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rdma {

void fatal(const std::string& what) {
  std::fprintf(stderr, "[fatal] %s\n", what.c_str());
  std::exit(EXIT_FAILURE);
}

void check(int rc, const std::string& what) {
  if (rc != 0) {
    std::fprintf(stderr, "[error] %s (rc=%d, errno=%d: %s)\n", what.c_str(), rc,
                 errno, std::strerror(errno));
    std::exit(EXIT_FAILURE);
  }
}

/* ============================================================ 控制面 */

ControlChannel::ControlChannel(bool is_server, const std::string& host,
                               std::uint16_t port) {
  int listen_fd = -1;
  if (is_server) {
    listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) fatal("socket() failed");
    int one = 1;
    ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    check(::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)),
          "bind() failed");
    check(::listen(listen_fd, 1), "listen() failed");

    std::printf("[ctrl] listening on :%u, waiting for peer...\n", port);
    sockaddr_in peer{};
    socklen_t plen = sizeof(peer);
    fd_ = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&peer), &plen);
    if (fd_ < 0) fatal("accept() failed");
    std::printf("[ctrl] peer connected\n");
    ::close(listen_fd);
  } else {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) fatal("socket() failed");

    /* 用 getaddrinfo 而不是 inet_pton：前者既能吃 "10.6.11.66"，
       也能吃主机名 "f11r4n06" —— 集群里用主机名更常见。 */
    addrinfo hints{};
    hints.ai_family = AF_INET;        // 控制面统一走 IPv4
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const int gai = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res);
    if (gai != 0 || !res) {
      std::fprintf(stderr, "[error] getaddrinfo(\"%s\") failed: %s\n",
                   host.c_str(), ::gai_strerror(gai));
      fatal("getaddrinfo() failed");
    }
    sockaddr_in addr = *reinterpret_cast<sockaddr_in*>(res->ai_addr);
    const socklen_t addrlen = res->ai_addrlen;
    ::freeaddrinfo(res);

    int rc = -1;
    for (int attempt = 0; attempt < 200; ++attempt) {
      rc = ::connect(fd_, reinterpret_cast<sockaddr*>(&addr), addrlen);
      if (rc == 0) break;
      ::usleep(50'000);  // 等服务端起监听
    }
    if (rc != 0) fatal("connect() failed after retries");
    std::printf("[ctrl] connected to %s:%u\n", host.c_str(), port);
  }
  int one = 1;
  ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

ControlChannel::~ControlChannel() {
  if (fd_ >= 0) ::close(fd_);
}

namespace {
void write_all(int fd, const void* p, std::size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n > 0) {
    ssize_t k = ::send(fd, c, n, 0);
    if (k <= 0) fatal("send() failed");
    c += k;
    n -= static_cast<std::size_t>(k);
  }
}

void read_all(int fd, void* p, std::size_t n) {
  char* c = static_cast<char*>(p);
  while (n > 0) {
    ssize_t k = ::recv(fd, c, n, 0);
    if (k <= 0) fatal("recv() failed");
    c += k;
    n -= static_cast<std::size_t>(k);
  }
}
}  // namespace

ConnInfo ControlChannel::exchange(const ConnInfo& mine) {
  // 先发后收：两边都这样，就不会互相等
  write_all(fd_, &mine, sizeof(mine));
  ConnInfo theirs{};
  read_all(fd_, &theirs, sizeof(theirs));
  return theirs;
}

/* ============================================================ 数据面 */

RdmaEndpoint::RdmaEndpoint(const std::string& dev_name, std::size_t buf_size,
                           int gid_index, int tx_depth, int rx_depth,
                           void* external_buf)
    : buf_size_(buf_size),
      gid_index_(gid_index),
      tx_depth_(tx_depth),
      rx_depth_(rx_depth) {
  /* --- 1. 打开设备：拿到 ibv_context --- */
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list || n == 0) fatal("no RDMA device found（SoftRoCE 没配好？）");

  ibv_device* dev = nullptr;
  for (int i = 0; i < n; ++i) {
    if (dev_name.empty() || dev_name == ibv_get_device_name(list[i])) {
      dev = list[i];
      break;
    }
  }
  if (!dev) fatal("device '" + dev_name + "' not found");

  std::printf("[dev ] using %s\n", ibv_get_device_name(dev));
  ctx_ = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (!ctx_) fatal("ibv_open_device() failed");

  /* --- 2. 保护域 PD：所有资源的容器 --- */
  pd_ = ibv_alloc_pd(ctx_);
  if (!pd_) fatal("ibv_alloc_pd() failed");

  /* --- 3. 注册内存 MR：pin 住物理页 + 生成 lkey/rkey  ---
     这一步是 RDMA 的地基：没有 MR，网卡就没法直接访问这块内存，
     也就没有"内核旁路"。rkey 给对端用（单边操作），lkey 给自己用。 */
  if (external_buf != nullptr) {
    /* GPUDirect 路径：直接注册调用方给的**设备内存**（如 hipMalloc 的 DCU 显存）。
       注意：不能 memset（那是显存），且 buffer 由调用方持有，析构时不能 free。 */
    buf_ = external_buf;
    owns_buf_ = false;
    std::printf("[mr  ] 外部缓冲区（设备显存）addr=%p\n", buf_);
  } else {
    buf_ = std::aligned_alloc(4096, ((buf_size_ + 4095) / 4096) * 4096);
    if (!buf_) fatal("std::aligned_alloc() failed");
    std::memset(buf_, 0, buf_size_);
  }

  mr_ = ibv_reg_mr(pd_, buf_, buf_size_,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ);
  if (!mr_) fatal("ibv_reg_mr() failed（显存注册失败 = 这块卡/驱动不支持 GPUDirect）");
  std::printf("[mr  ] buf=%p len=%zu lkey=0x%x rkey=0x%x\n", buf_, buf_size_,
              mr_->lkey, mr_->rkey);

  /* --- 4. 完成队列 CQ：硬件干完活往这里写 CQE，软件 poll --- */
  cq_ = ibv_create_cq(ctx_, tx_depth_ + rx_depth_, nullptr, nullptr, 0);
  if (!cq_) fatal("ibv_create_cq() failed");

  /* --- 5. 创建 QP：一次 RC 连接的基本句柄 --- */
  ibv_qp_init_attr qp_attr{};
  qp_attr.send_cq = cq_;
  qp_attr.recv_cq = cq_;
  qp_attr.qp_type = IBV_QPT_RC;  // 可靠连接：按序、有 ACK、有重传
  qp_attr.cap.max_send_wr = tx_depth_;
  qp_attr.cap.max_recv_wr = rx_depth_;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  qp_ = ibv_create_qp(pd_, &qp_attr);
  if (!qp_) fatal("ibv_create_qp() failed");
  std::printf("[qp  ] qpn=%u\n", qp_->qp_num);

  /* --- 6. 取本端 GID（对端 RTR 时要用它） --- */
  check(ibv_query_gid(ctx_, 1, gid_index_, &local_gid_), "ibv_query_gid() failed");
  ibv_port_attr pa{};
  check(ibv_query_port(ctx_, 1, &pa), "ibv_query_port() failed");
  lid_ = RDMA_LID_READ(pa.lid);
  /* 链路层自适应：InfiniBand 用 LID 寻址（16/17 位），Ethernet/RoCEv2 用 GID 寻址。
     同一 QP 的 AV 只能选一种，选错了就是 transport retry counter exceeded。 */
  use_gid_ = (pa.link_layer == IBV_LINK_LAYER_ETHERNET);
  std::printf("[link] %s -> %s 寻址 (lid=0x%x)\n",
              pa.link_layer == IBV_LINK_LAYER_ETHERNET     ? "Ethernet(RoCE)"
              : pa.link_layer == IBV_LINK_LAYER_INFINIBAND ? "InfiniBand"
                                                           : "unknown",
              use_gid_ ? "GID" : "LID", lid_);
  std::printf("[gid ] index=%d %02x%02x:%02x%02x:...:%02x%02x\n", gid_index_,
              local_gid_.raw[0], local_gid_.raw[1], local_gid_.raw[2],
              local_gid_.raw[3], local_gid_.raw[14], local_gid_.raw[15]);
}

RdmaEndpoint::~RdmaEndpoint() {
  if (qp_) ibv_destroy_qp(qp_);
  if (cq_) ibv_destroy_cq(cq_);
  if (mr_) ibv_dereg_mr(mr_);
  if (pd_) ibv_dealloc_pd(pd_);
  if (ctx_) ibv_close_device(ctx_);
  if (buf_ && owns_buf_) std::free(buf_);
}

ConnInfo RdmaEndpoint::local_info() const {
  ConnInfo me{};
  me.qpn = qp_->qp_num;
  me.rkey = mr_->rkey;
  me.addr = reinterpret_cast<std::uint64_t>(buf_);
  std::memcpy(me.gid, local_gid_.raw, 16);
  me.lid = lid_;
  return me;
}

/* ---- QP 状态机：RESET → INIT → RTR → RTS，少一步都收不了数据 -
   RESET 是创建后的初始态，只能本地操作；
   INIT  开始能收（但还不知道对端在哪）；
   RTR   知道了对端 QPN/GID，能收不能发；
   RTS   最后一步，可以发了。                                        */

void RdmaEndpoint::to_init() {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  check(ibv_modify_qp(qp_, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS),
        "modify_qp(RESET->INIT) failed");
}

void RdmaEndpoint::to_rtr(const ConnInfo& peer, int port_num, int mtu_log) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = static_cast<ibv_mtu>(mtu_log);
  attr.dest_qp_num = peer.qpn;          // ← 对端信息 ①
  attr.rq_psn = psn_;
  attr.max_dest_rd_atomic = 1;          // 允许对端对我发起 1 个 READ
  attr.min_rnr_timer = 12;
  /* --- 寻址方式按链路层二选一（构造时已探测）--- */
  if (use_gid_) {
    /* RoCEv2（Ethernet）：AH 带上 GRH，用对端 GID 寻址 */
    attr.ah_attr.is_global = 1;
    std::memcpy(attr.ah_attr.grh.dgid.raw, peer.gid, 16);   // ← 对端信息 ②
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.sgid_index = static_cast<std::uint8_t>(gid_index_);
    attr.ah_attr.grh.traffic_class = 0;
    RDMA_LID_ZERO(attr.ah_attr.dlid);
  } else {
    /* 纯 InfiniBand：不用 GRH，直接用对端 LID（16/17 位）寻址 */
    attr.ah_attr.is_global = 0;
    RDMA_LID_WRITE(attr.ah_attr.dlid, peer.lid);
  }
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = static_cast<std::uint8_t>(port_num);
  check(ibv_modify_qp(qp_, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
        "modify_qp(INIT->RTR) failed");
}

void RdmaEndpoint::to_rts() {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;                    // 对端还没 post 接收缓冲时不报死，重试
  attr.sq_psn = psn_;
  attr.max_rd_atomic = 1;
  check(ibv_modify_qp(qp_, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC),
        "modify_qp(RTR->RTS) failed");
}

void RdmaEndpoint::connect_to(const ConnInfo& peer, int port_num) {
  ibv_port_attr pa{};
  check(ibv_query_port(ctx_, port_num, &pa), "ibv_query_port() failed");
  to_init();
  to_rtr(peer, port_num, pa.active_mtu);
  to_rts();
  std::printf("[qp  ] RESET -> INIT -> RTR -> RTS done (peer qpn=%u)\n", peer.qpn);
}

/* ---- 一收一发：四个操作的区别全在这里 ----------------------------
   SEND / RECV  —— 双边：双方都要 post。对端必须先 post_recv，
                   否则触发 RNR 重试（rnr_retry=7 就是为这个留的余地）。
   WRITE / READ —— 单边：**对端什么都不用做**，远端 CPU 全程不知情。
                   代价是发起端必须先知道 (远端地址, rkey)。        */

static void post_common(ibv_send_wr& wr, ibv_sge& sge, void* local,
                        std::size_t len, ibv_mr* mr, std::uint64_t wr_id) {
  std::memset(&wr, 0, sizeof(wr));
  sge.addr = reinterpret_cast<std::uint64_t>(local);
  sge.length = static_cast<std::uint32_t>(len);
  sge.lkey = mr->lkey;
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;  // 要 CQE 就得带这个；不要 CQE 就不带
}

void RdmaEndpoint::post_send(const void* local, std::size_t len) {
  ibv_send_wr wr{};
  ibv_sge sge{};
  post_common(wr, sge, const_cast<void*>(local), len, mr_, next_wr_id_++);
  wr.opcode = IBV_WR_SEND;
  ibv_send_wr* bad = nullptr;
  check(ibv_post_send(qp_, &wr, &bad), "ibv_post_send(SEND) failed");
}

void RdmaEndpoint::post_recv(void* local, std::size_t len) {
  ibv_recv_wr wr{};
  ibv_sge sge{};
  std::memset(&wr, 0, sizeof(wr));
  sge.addr = reinterpret_cast<std::uint64_t>(local);
  sge.length = static_cast<std::uint32_t>(len);
  sge.lkey = mr_->lkey;
  wr.wr_id = next_wr_id_++;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  ibv_recv_wr* bad = nullptr;
  check(ibv_post_recv(qp_, &wr, &bad), "ibv_post_recv() failed");
}

void RdmaEndpoint::post_write(const void* local, std::size_t len,
                              const ConnInfo& peer) {
  ibv_send_wr wr{};
  ibv_sge sge{};
  post_common(wr, sge, const_cast<void*>(local), len, mr_, next_wr_id_++);
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.wr.rdma.remote_addr = peer.addr;   // 推：我从本端推到对端
  wr.wr.rdma.rkey = peer.rkey;
  ibv_send_wr* bad = nullptr;
  check(ibv_post_send(qp_, &wr, &bad), "ibv_post_send(RDMA_WRITE) failed");
}

void RdmaEndpoint::post_read(void* local, std::size_t len,
                             const ConnInfo& peer) {
  ibv_send_wr wr{};
  ibv_sge sge{};
  post_common(wr, sge, local, len, mr_, next_wr_id_++);
  wr.opcode = IBV_WR_RDMA_READ;
  wr.wr.rdma.remote_addr = peer.addr;   // 拉：请求发出去，数据自己回来
  wr.wr.rdma.rkey = peer.rkey;
  ibv_send_wr* bad = nullptr;
  // READ 的首字节延迟比 WRITE 多一个 RTT —— 请求得先到对端，数据才能回。
  check(ibv_post_send(qp_, &wr, &bad), "ibv_post_send(RDMA_READ) failed");
}

std::uint64_t RdmaEndpoint::wait_one() {
  ibv_wc wc{};
  while (ibv_poll_cq(cq_, 1, &wc) == 0) {
    // 真实系统不会这样裸轮询：可以改用事件通道（ibv_create_comp_channel），
    // 或者像 NCCL 那样把 poll 混进计算流里，避免空转吃满一个核。
  }
  if (wc.status != IBV_WC_SUCCESS) {
    std::fprintf(stderr, "[wc  ] status=%s (%d) opcode=%d vendor_err=%u\n",
                 ibv_wc_status_str(wc.status), wc.status, wc.opcode, wc.vendor_err);
    fatal("work completion failed");
  }
  std::printf("[wc  ] wr_id=%llu opcode=%d byte_len=%u qpn=%u\n",
              static_cast<unsigned long long>(wc.wr_id), wc.opcode,
              wc.byte_len, wc.qp_num);
  return wc.wr_id;
}

}  // namespace rdma
