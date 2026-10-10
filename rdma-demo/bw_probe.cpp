// bw_probe.cpp —— 自写 RDMA Verbs 程序的**带宽**测试（与 perftest 同口径对照）
//
// 【为什么加这个】
//   rdma-demo 的 send_recv / write_read 是"功能演示"（64 字节、一发一收），
//   证明得了"能跑通"，证明不了"跑多快"。
//   而 perftest 的 ib_write_bw 是**流水线**发的（连续 post 多个 WR，不等完成）。
//
//   所以这里用同一套 common（QP 建立、LID/GID 自适应）实现一个**流水线版 WRITE**：
//       status: post 多个 WR 再统一收割 CQ  → 这才是带宽测试该有的样子
//       status: 对照 perftest 实测的 45.07 GB/s，看自己写的能到百分之多少
//
// 编译：g++ -std=c++17 -O2 -Wall -Wextra -o bw_probe bw_probe.cpp common.cpp -libverbs
// 运行：./bw_probe server 0.0.0.0 shca_0 8388608 200
//       ./bw_probe client <host>     shca_0 8388608 200

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common.hpp"

namespace {

double now_sec() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s {server|client} {host} [device] [msg_bytes] [iters]\n",
                 argv[0]);
    return 1;
  }
  const std::string role = argv[1];
  const std::string host = argv[2];
  const std::string dev = (argc > 3) ? argv[3] : "shca_0";
  const std::size_t msg = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 8u << 20;
  const int iters = (argc > 5) ? std::atoi(argv[5]) : 200;
  const int depth = 16;  // 流水线深度：连续 post 的 WR 数
  const bool is_server = (role == "server");

  // buffer 要能装下 msg，且对齐（MR 注册要求）
  const std::size_t buf_size = ((msg + 4095) / 4096) * 4096;
  rdma::RdmaEndpoint ep(dev, buf_size, /*gid_index=*/1, depth, depth);
  rdma::ControlChannel ctl(is_server, host, 18540);
  const rdma::ConnInfo peer = ctl.exchange(ep.local_info());
  ep.connect_to(peer);

  // client 的准备数据写进自己的 buffer（server 侧不用动，它只是被写的目标）
  if (!is_server) std::memset(ep.buf(), 0xAB, buf_size);

  const double gb = static_cast<double>(msg) * iters / 1e9;

  if (is_server) {
    std::printf("[srv ] 等 client 写完 %d 次 × %.1f MB（我的 CPU 全程不参与）\n", iters,
                msg / 1024.0 / 1024.0);
    // 只收一个 1 字节完工信令
    ep.post_recv(ep.buf(), 1);
    ep.wait_one();
    std::printf("[srv ] 收到完工信令，buffer 首字节=0x%02x（被 WRITE 覆盖过）\n",
                *static_cast<unsigned char*>(ep.buf()));
  } else {
    // 预热
    for (int i = 0; i < depth; ++i) ep.post_write(ep.buf(), msg, peer);
    for (int i = 0; i < depth; ++i) ep.wait_one();

    const double t0 = now_sec();
    int done = 0;
    while (done < iters) {
      const int batch = std::min(depth, iters - done);
      for (int i = 0; i < batch; ++i) ep.post_write(ep.buf(), msg, peer);  // 连续 post
      for (int i = 0; i < batch; ++i) ep.wait_one();                       // 再统一收割
      done += batch;
    }
    const double dt = now_sec() - t0;

    std::printf("[cli ] 自写 verbs WRITE  %.1f MB × %d 次  用时 %.4f s\n",
                msg / 1024.0 / 1024.0, iters, dt);
    std::printf("       带宽 = %.2f GB/s  (%.0f MB/s)   流水线深度=%d\n", gb / dt, gb * 1000 / dt,
                depth);

    ep.post_send(ep.buf(), 1);
    ep.wait_one();
  }
  return 0;
}
