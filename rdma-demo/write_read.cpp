// write_read.cpp —— 单边操作：RDMA WRITE / RDMA READ
//
// 核心特征：**对端 CPU 全程不知情**。
//   写：我 post_send(RDMA_WRITE) → 我的网卡直接从我的内存取数 → 推到对端内存。
//   读：我 post_send(RDMA_READ)  → 我的网卡只发一个很小的读请求出去 →
//        **对端的网卡**按 rkey 查页表、把数据回发给我。
//
// 三件事这份代码能让你亲眼看到：
//   ① READ 能读到远端内存，而远端进程一行代码都没执行；
//   ② WRITE 改了远端的内存，而远端进程同样什么都没做；
//   ③ READ 比 WRITE 多一个 RTT —— 请求得先到对端，数据才回得来。
//      （在 SoftRoCE 上配合 `tc netem` 加延迟，这个差别会放大到肉眼可见）
//
// 用法：
//     ./write_read server
//     ./write_read client 127.0.0.1

#include <cstdio>
#include <cstring>
#include <string>

#include "common.hpp"

namespace {

constexpr std::uint16_t kCtrlPort = 18516;
constexpr std::size_t kMsgLen = 64;

void run_server(const std::string& dev) {
  rdma::RdmaEndpoint ep(dev, 4096);
  rdma::ControlChannel ctrl(true, "", kCtrlPort);

  auto peer = ctrl.exchange(ep.local_info());
  ep.connect_to(peer);

  char* buf = static_cast<char*>(ep.buf());
  const auto me = ep.local_info();
  std::printf("[dbg ] server 本地 buf=%p rkey=0x%x | 对端 addr=0x%llx rkey=0x%x qpn=%u\n",
              static_cast<void*>(buf), me.rkey,
              static_cast<unsigned long long>(peer.addr), peer.rkey, peer.qpn);
  std::strncpy(buf, "SERVER-BUFFER", kMsgLen);
  std::printf("[server] 我的 buf 初始内容: \"%.*s\"\n", static_cast<int>(kMsgLen), buf);

  // ★ 就绪握手：初值写好了再通知对端，否则对端可能在我说"初值是 SERVER-BUFFER"
  //   之前就 READ 了，读到的是空 buffer（竞态）。
  char* ready = buf + 3072;
  *ready = 'R';
  ep.post_send(ready, 1);
  ep.wait_one();

  // 对端接下来会：READ 我这段内存 → 再 WRITE 覆盖它。
  // **我这个进程这期间什么都不做** —— 唯一做的一件事是等一个信令，
  // 好知道对方干完了（真实系统用完成事件/心跳通知，这里用一个字节的 SEND 代替）。
  char* sig = buf + 2048;
  ep.post_recv(sig, 1);
  std::printf("[server] 我不参与任何数据搬运，等对端完工信令...\n");
  ep.wait_one();

  std::printf("[server] 对方干完了。我的 buf 现在是: \"%.*s\"\n",
              static_cast<int>(kMsgLen), buf);
  std::printf("[server] ↑ 注意：这个变化是对方 WRITE 进来的，"
              "我的 CPU 全程没有碰过这段内存\n");
}

void run_client(const std::string& dev, const std::string& host) {
  rdma::RdmaEndpoint ep(dev, 4096);
  rdma::ControlChannel ctrl(false, host, kCtrlPort);

  auto peer = ctrl.exchange(ep.local_info());
  ep.connect_to(peer);

  char* buf = static_cast<char*>(ep.buf());
  const auto me = ep.local_info();
  std::printf("[dbg ] client 本地 buf=%p rkey=0x%x | 对端 addr=0x%llx rkey=0x%x qpn=%u lid=0x%x\n",
              static_cast<void*>(buf), me.rkey,
              static_cast<unsigned long long>(peer.addr), peer.rkey, peer.qpn, peer.lid);

  // ★ 等对端把初值写好（对应 server 端的就绪握手），消除竞态
  char* ready = buf + 3072;
  ep.post_recv(ready, 1);
  ep.wait_one();

  // ---- ① RDMA READ：单边拉远端内存 ----
  std::memset(buf, 0, kMsgLen);
  ep.post_read(buf, kMsgLen, peer);
  ep.wait_one();
  std::printf("[client] READ 到对端内存: \"%.*s\"\n", static_cast<int>(kMsgLen), buf);

  // ---- ② RDMA WRITE：单边推数据到远端内存 ----
  std::strncpy(buf, "CLIENT-WROTE-HERE", kMsgLen);
  ep.post_write(buf, kMsgLen, peer);
  ep.wait_one();
  std::printf("[client] WRITE 完成：对端那段内存已被我覆盖\n");

  // ---- ③ 再 READ 一次确认：单边写确实生效了 ----
  std::memset(buf, 0, kMsgLen);
  ep.post_read(buf, kMsgLen, peer);
  ep.wait_one();
  std::printf("[client] 再 READ 一次确认: \"%.*s\"\n", static_cast<int>(kMsgLen), buf);

  // ---- 发个信令给对面，让它打印自己那块内存 ----
  buf[2048] = 'x';   // 写 1 字节信令（strncpy 会触发 -Wstringop-truncation，直接赋值）
  ep.post_send(buf + 2048, 1);
  ep.wait_one();
  std::printf("[client] 已通知对端\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s {server|client} [host] [device]\n", argv[0]);
    return 1;
  }
  const std::string role = argv[1];
  const std::string host = (argc > 2) ? argv[2] : "127.0.0.1";
  const std::string dev = (argc > 3) ? argv[3] : "rxe0";

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
