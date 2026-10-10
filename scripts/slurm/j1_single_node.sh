#!/bin/bash
#SBATCH -J acc-j1-single
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:8
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j1-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j1-%j.err

# J1：单节点 8 卡 —— RCCL 五种集合操作 + HIP kernel + RDMA verbs + MPI 单机 + UCX
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
EX="--allow-run-as-root --map-by ppr:8:node"
XOPT="-x LD_LIBRARY_PATH -x PATH -x HSA_FORCE_FINE_GRAIN_PCIE -x NCCL_NET_PLUGIN -x NCCL_IB_HCA -x NCCL_TOPO_FILE -x RCCL_NET_PLANE -x RCCL_PXN_GPU_BALANCE"
T=/public/software/app/rccl/tests
echo "########## J1 on $(hostname) $(date) ##########"

echo "===== [1] DCU / GPU 信息 ====="
command -v hy-smi >/dev/null && hy-smi 2>&1 | head -20
echo "===== [2] RDMA 设备 ====="
ibv_devices 2>&1
echo "===== [3] UCX（JD 任职 6）====="
ucx_info -v 2>&1 | head -3
ucx_info -d 2>&1 | grep -B1 -A2 "Transport: rc" | head -20

echo "===== [4] RCCL 单节点 8 卡：五种集合操作 ====="
for t in all_reduce all_gather reduce_scatter alltoall; do
  echo "----- $t -----"
  mpirun $EX -np 8 $XOPT $T/${t}_perf -b 8 -e 64M -f 4 -g 1 -w 2 -n 3 2>&1 | tail -20
done
echo "----- sendrecv (点对点) -----"
mpirun $EX -np 8 $XOPT $T/sendrecv_perf -b 8 -e 64M -f 4 -g 1 -w 2 -n 3 2>&1 | tail -15

echo "===== [5] HIP kernel（JD 优先 3）====="
hipcc -O2 -o /tmp/vector_add $WS/experiments/vector_add.hip 2>&1 | tail -5
if [ -x /tmp/vector_add ]; then /tmp/vector_add 2>&1 | tail -15; else echo "[FAIL] hipcc 编译失败"; fi

echo "===== [6] RDMA verbs 单边/双边（本节点 loopback，自研卡 shca_0）====="
cd $WS/rdma-demo
( ./write_read server 127.0.0.1 shca_0 > /tmp/j1_wr_srv.log 2>&1 & ) ; sleep 3
./write_read client 127.0.0.1 shca_0 2>&1 | tail -12
sleep 2; echo "--- server 侧 ---"; tail -6 /tmp/j1_wr_srv.log
( ./send_recv server 127.0.0.1 shca_0 > /tmp/j1_sr_srv.log 2>&1 & ) ; sleep 3
./send_recv client 127.0.0.1 shca_0 2>&1 | tail -8
sleep 2; echo "--- server 侧 ---"; tail -5 /tmp/j1_sr_srv.log

echo "===== [7] MPI 基准（8 进程单节点）====="
mpicxx -O2 -o /tmp/mpi_bench $WS/mpi-lab/mpi_bench.cpp 2>&1 | tail -3
mpirun --allow-run-as-root -np 8 /tmp/mpi_bench 2>&1 | tail -20

echo "===== [8] torch four_ops（若环境有 torch+DCU）====="
cd $WS/experiments
if command -v torchrun >/dev/null 2>&1 && python3 -c "import torch" >/dev/null 2>&1; then
  torchrun --nproc_per_node=8 four_ops_test.py 2>&1 | tail -25
else
  echo "[skip] 该节点无 python3.12 解释器 / 无 torch —— four_ops_test.py 无法在本机运行"
  python3 -c "import torch" 2>&1 | tail -2
fi

echo "########## J1 done $(date) ##########"
