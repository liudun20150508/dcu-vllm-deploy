// mpi_rocm_allreduce.cpp —— GPU-aware MPI：让 MPI 直接对 DCU 显存做集合通信
//
// 【为什么这个实验关键】
//   前面测出来 MPI（CPU 侧）的 AllReduce 只有 1.19 GB/s @256MB，
//   而 RCCL 在 DCU 上同尺寸是 55~68 GB/s —— 差 50 倍左右。
//   差在哪？**数据要过 CPU 内存**：D2H 拷下来 → 主机做归约 → H2D 拷回去。
//   所以问题的核心是：**这个 MPI 能不能直接认 DCU 显存指针**（GPU-aware）。
//
//   集群里的 OpenMPI 5.0.3/shca 编译期是带 `rocm` 扩展的（见 ompi_info），
//   这个程序就是要把它"运行期到底行不行"验出来，并量出两种路径的差距：
//     ① GPU-aware：MPI_Allreduce 的 buffer 直接给显存指针
//     ② 手工中转：D2H → MPI_Allreduce(host) → H2D
//
// 编译： mpicxx -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -o mpi_rocm_allreduce mpi_rocm_allreduce.cpp -lamdhip64 -L$DTKROOT/hip/lib
// 运行： mpirun -np 2 ./mpi_rocm_allreduce

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void hip_ok(hipError_t rc, const char* what) {
  if (rc != hipSuccess) {
    std::fprintf(stderr, "[hip] %s -> %s\n", what, hipGetErrorString(rc));
    std::exit(1);
  }
}

double bw(double bytes, double sec) { return bytes / sec / 1e9; }

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  hipDeviceProp_t prop{};
  hip_ok(hipGetDeviceProperties(&prop, 0), "get prop");
  if (rank == 0) {
    std::printf("=== GPU-aware MPI：MPI 直接对 DCU 显存做 AllReduce ===\n");
    std::printf("进程数=%d  设备=%s\n\n", size, prop.name);
  }

  // 编译期是否声明支持
#ifdef MPIX_CUDA_AWARE_SUPPORT
  if (rank == 0) std::printf("MPIX_CUDA_AWARE_SUPPORT = %d\n", MPIX_CUDA_AWARE_SUPPORT);
#endif

  const std::size_t count = 64u * 1024 * 1024 / sizeof(float);  // 64 MB
  const std::size_t bytes = count * sizeof(float);
  const int iters = 5;

  float* d_buf = nullptr;
  hip_ok(hipMalloc(&d_buf, bytes), "hipMalloc");
  std::vector<float> h_buf(count, 0.0f), h_ref(count, 0.0f);
  for (std::size_t i = 0; i < count; ++i) h_buf[i] = static_cast<float>(rank + 1);
  hip_ok(hipMemcpy(d_buf, h_buf.data(), bytes, hipMemcpyHostToDevice), "H2D");

  // ---------------- ① GPU-aware：直接把显存指针交给 MPI ----------------
  int gpu_ok = 1;
  double t_gpu = 0;
  if (rank == 0) std::printf("[①] GPU-aware（buffer 直接是 DCU 显存指针）\n");
  MPI_Barrier(MPI_COMM_WORLD);
  {
    const double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
      // MPI_IN_PLACE 让 MPI 直接读写显存，不经主机中转
      const int rc = MPI_Allreduce(MPI_IN_PLACE, d_buf, static_cast<int>(count), MPI_FLOAT,
                                   MPI_SUM, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) { gpu_ok = 0; break; }
    }
    const double dt = (MPI_Wtime() - t0) / iters;
    t_gpu = dt;
    // 校验：每个元素都应等于 1+2+...+size
    hip_ok(hipMemcpy(h_ref.data(), d_buf, bytes, hipMemcpyDeviceToHost), "D2H");
    const float expect = static_cast<float>(size * (size + 1) / 2);
    int bad = 0;
    for (std::size_t i = 0; i < count; i += count / 16) {
      if (h_ref[i] != expect) ++bad;
    }
    if (rank == 0) {
      if (gpu_ok && bad == 0)
        std::printf("    ✓ MPI 直接对显存做 AllReduce 成功：%.3f ms  busbw=%.2f GB/s  "
                    "（校验值=%.0f ✓）\n", t_gpu * 1e3, bw(bytes, t_gpu) * 2 * (size - 1) / size, expect);
      else
        std::printf("    ✗ 失败：gpu_ok=%d 校验不符=%d 项（说明这个 MPI 运行期不支持 ROCm 指针）\n",
                    gpu_ok, bad);
    }
  }

  // ---------------- ② 手工中转：D2H → MPI → H2D ----------------
  if (rank == 0) std::printf("\n[②] 手工中转（D2H → MPI_Allreduce(host) → H2D）\n");
  MPI_Barrier(MPI_COMM_WORLD);
  {
    const double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
      hip_ok(hipMemcpy(h_buf.data(), d_buf, bytes, hipMemcpyDeviceToHost), "D2H");
      MPI_Allreduce(MPI_IN_PLACE, h_buf.data(), static_cast<int>(count), MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      hip_ok(hipMemcpy(d_buf, h_buf.data(), bytes, hipMemcpyHostToDevice), "H2D");
    }
    const double dt = (MPI_Wtime() - t0) / iters;
    if (rank == 0)
      std::printf("    %.3f ms  busbw=%.2f GB/s（含两次 PCIe 拷贝）\n", dt * 1e3,
                  bw(bytes, dt) * 2 * (size - 1) / size);
  }

  // ---------------- ③ 对照：纯显存内拷贝，看 PCIe 的天花板 ----------------
  MPI_Barrier(MPI_COMM_WORLD);
  {
    const double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
      hip_ok(hipMemcpy(h_buf.data(), d_buf, bytes, hipMemcpyDeviceToHost), "D2H");
    }
    const double dt = (MPI_Wtime() - t0) / iters;
    if (rank == 0) std::printf("\n[③] 参考：单次 D2H 拷贝 %.3f ms → %.2f GB/s（PCIe 带宽上限）\n",
                               dt * 1e3, bw(bytes, dt));
  }

  hip_ok(hipFree(d_buf), "hipFree");
  if (rank == 0) std::printf("\n=== 完 ===\n");
  MPI_Finalize();
  return 0;
}
