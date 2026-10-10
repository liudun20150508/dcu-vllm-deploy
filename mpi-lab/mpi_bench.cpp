// mpi_bench.cpp —— MPI 点对点 + 集合通信基准
//
// 目的：把 MPI 这一层也亲手跑一遍，并且和 RCCL 的同一指标对照，
//       这样"通信库选型"就不是纸上谈兵。
//
// 测三样：
//   ① Ping-Pong 延迟（点对点，小消息 → 看延迟）
//   ② Ping-Pong 带宽（点对点，大消息 → 看带宽）
//   ③ AllReduce 带宽（集合操作，和 all_reduce_perf 对照）
//
// 编译： mpicxx -O2 -o mpi_bench mpi_bench.cpp
// 运行： mpirun -np <N> ./mpi_bench

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

double now_sec() {
  return MPI_Wtime();
}

// ① + ② 点对点 Ping-Pong：rank0 发、rank1 回，测 RTT
void pingpong(int rank, int nranks, std::size_t bytes, int iters) {
  if (nranks < 2) return;
  std::vector<char> sbuf(bytes, 'a'), rbuf(bytes, 'b');
  MPI_Status st;

  // 预热
  for (int i = 0; i < 3; ++i) {
    if (rank == 0) {
      MPI_Send(sbuf.data(), static_cast<int>(bytes), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(rbuf.data(), static_cast<int>(bytes), MPI_CHAR, 1, 0, MPI_COMM_WORLD, &st);
    } else if (rank == 1) {
      MPI_Recv(rbuf.data(), static_cast<int>(bytes), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &st);
      MPI_Send(sbuf.data(), static_cast<int>(bytes), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  const double t0 = now_sec();
  for (int i = 0; i < iters; ++i) {
    if (rank == 0) {
      MPI_Send(sbuf.data(), static_cast<int>(bytes), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(rbuf.data(), static_cast<int>(bytes), MPI_CHAR, 1, 0, MPI_COMM_WORLD, &st);
    } else if (rank == 1) {
      MPI_Recv(rbuf.data(), static_cast<int>(bytes), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &st);
      MPI_Send(sbuf.data(), static_cast<int>(bytes), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
  }
  const double dt = (now_sec() - t0) / iters;

  if (rank == 0) {
    const double lat_us = dt * 1e6 / 2.0;                       // 单程
    const double bw_gbs = 2.0 * bytes / dt / 1e9;               // 双向
    std::printf("  pingpong  %10zu B  RTT %9.2f us  单向延迟 %8.2f us  带宽 %8.2f GB/s\n",
                bytes, dt * 1e6, lat_us, bw_gbs);
  }
}

// ③ AllReduce：每个 rank 先填自己的值，规约后校验 + 计时
void allreduce_bw(MPI_Comm comm, int rank, int nranks, std::size_t bytes,
                  std::size_t count, MPI_Datatype dtype, int iters) {
  std::vector<double> buf(count, static_cast<double>(rank + 1));
  std::vector<double> ref(count);

  // 正确性：AllReduce(sum) 结果应该是 1+2+...+nranks
  MPI_Allreduce(buf.data(), ref.data(), static_cast<int>(count), dtype, MPI_SUM, comm);
  const double expect = static_cast<double>(nranks) * (nranks + 1) / 2.0;
  bool ok = true;
  for (std::size_t i = 0; i < count; ++i) {
    if (ref[i] != expect) { ok = false; break; }
  }

  for (int i = 0; i < 3; ++i)
    MPI_Allreduce(buf.data(), ref.data(), static_cast<int>(count), dtype, MPI_SUM, comm);
  MPI_Barrier(comm);

  const double t0 = now_sec();
  for (int i = 0; i < iters; ++i)
    MPI_Allreduce(buf.data(), ref.data(), static_cast<int>(count), dtype, MPI_SUM, comm);
  const double dt = (now_sec() - t0) / iters;

  if (rank == 0) {
    // Ring-AllReduce 的 busbw 系数（和 nccl-tests 同口径）：2(n-1)/n
    const double factor = 2.0 * (nranks - 1) / nranks;
    const double algbw = bytes / dt / 1e9;
    std::printf("  allreduce %10zu B  时间 %10.2f us  algbw %7.2f GB/s  busbw %7.2f GB/s  %s\n",
                bytes, dt * 1e6, algbw, algbw * factor,
                ok ? "OK" : "!! 结果错误");
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  char host[MPI_MAX_PROCESSOR_NAME];
  int hlen = 0;
  MPI_Get_processor_name(host, &hlen);

  if (rank == 0) {
    std::printf("=== MPI 通信基准 ===\n");
    std::printf("rank 数: %d\n", nranks);
    std::printf("MPI 版本: %d.%d\n", MPI_VERSION, MPI_SUBVERSION);
  }
  // 打印 rank→节点映射（确认真的跨节点了）
  if (nranks <= 8) {
    std::printf("  rank %d -> %s\n", rank, host);
  } else if (rank < 8 && rank == 0) {
    std::printf("  rank 0 -> %s\n", host);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) std::printf("\n--- ① 点对点 Ping-Pong ---\n");
  for (std::size_t b : {8ul, 1024ul, 65536ul, 1048576ul, 8388608ul})
    pingpong(rank, nranks, b, b < 65536 ? 1000 : 50);

  if (rank == 0) std::printf("\n--- ② MPI AllReduce（double）---\n");
  for (std::size_t cnt : {1024ul, 65536ul, 1048576ul, 8388608ul}) {
    allreduce_bw(MPI_COMM_WORLD, rank, nranks, cnt * sizeof(double), cnt,
                 MPI_DOUBLE, cnt < 65536 ? 200 : 20);
  }

  MPI_Finalize();
  return 0;
}
