// send_recv.cpp —— 双边操作：SEND / RECV
//
// 核心特征：**双方都要动**。发送方 post_send，接收方必须**提前** post_recv。
// 如果对端还没准备好接收缓冲，会触发 RNR（Receiver Not Ready）重试 ——
// 这就是"双边"的代价：对端 CPU 被惊动了。
//
// 用法（两个终端）：
//     ./send_recv server
//     ./send_recv client 127.0.0.1
//
// 在 SoftRoCE 上跑：
//     sudo rdma link add rxe0 type rxe netdev eth0
//     ./send_recv server & ./send_recv client 127.0.0.1

#include <cstdio>
#include <cstring>
#include <string>

#include "common.hpp"

namespace {

constexpr std::uint16_t kCtrlPort = 18515;
constexpr std::size_t kMsgLen = 64;

void run_server(const std::string& dev) {
  rdma::RdmaEndpoint ep(dev, 4096);
  rdma::ControlChannel ctrl(true, "", kCtrlPort);

  // 1) 交换 ConnInfo —— RTR 需要对端 QPN/GID
  auto peer = ctrl.exchange(ep.local_info());
  ep.connect_to(peer);

  // 2) 关键：**先** post_recv，否则对端 SEND 过来会 RNR
  char* buf = static_cast<char*>(ep.buf());
  ep.post_recv(buf, kMsgLen);
  std::printf("[server] posted RECV, waiting for SEND...\n");

  ep.wait_one();
  std::printf("[server] received %zu bytes: \"%.*s\"\n", kMsgLen,
              static_cast<int>(kMsgLen), buf);

  // 3) 回一条，演示同一条 QP 上双向都能走
  std::strncpy(buf, "ACK-FROM-SERVER", kMsgLen);
  ep.post_send(buf, kMsgLen);
  ep.wait_one();
  std::printf("[server] sent reply\n");
}

void run_client(const std::string& dev, const std::string& host) {
  rdma::RdmaEndpoint ep(dev, 4096);
  rdma::ControlChannel ctrl(false, host, kCtrlPort);

  auto peer = ctrl.exchange(ep.local_info());
  ep.connect_to(peer);

  char* buf = static_cast<char*>(ep.buf());
  std::strncpy(buf, "HELLO-FROM-CLIENT", kMsgLen);

  ep.post_send(buf, kMsgLen);
  ep.wait_one();
  std::printf("[client] sent %zu bytes\n", kMsgLen);

  // 收对端的回复 —— 同样要**先** post_recv
  std::memset(buf, 0, kMsgLen);
  ep.post_recv(buf, kMsgLen);
  ep.wait_one();
  std::printf("[client] got reply: \"%.*s\"\n", static_cast<int>(kMsgLen), buf);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s {server|client} [host] [device]\n", argv[0]);
    return 1;
  }
  const std::string role = argv[1];
  const std::string host = (argc > 2) ? argv[2] : "127.0.0.1";
  const std::string dev = (argc > 3) ? argv[3] : "rxe0";  // SoftRoCE 默认设备名

  if (role == "server") {
    run_server(dev);
  } else if (role == "client") {
    run_client(dev, host);
  } else {
    std::fprintf(stderr, "unknown role: %s\n", role.c_str());
    return 1;
  }
  return 0;
}
