#!/bin/bash
#SBATCH -J ucx-ucc
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/ucx-ucc-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/ucx-ucc-%j.err
# ============================================================
# JD 职责 2 / 任职 6：UCX + UCC 实测（补上「UCC 未跑」、升级「UCX 概念级」为实测）
# ============================================================
set -u
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load mpi/openmpi/5.0.3/gcc-8.5.0/shca

UCC=/public/software/mpi/ucc-dtk_2604/1.5.0
export PATH=$UCC/bin:$PATH
export LD_LIBRARY_PATH=$UCC/lib:$LD_LIBRARY_PATH

echo "===== [1] UCX：认不认 shca 自研网卡（传输层 + 设备）====="
ucx_info -d 2>/dev/null | grep -iE "Transport|Device" | head -12

echo
echo "===== [2] UCC：版本 + 可用组件 ====="
ucc_info -v 2>&1 | head -6
echo "--- UCC 组件/传输层 ---"
ucc_info -c 2>&1 | grep -iE "component|transport|shca|verbs|ucx|available" | head -15

echo
echo "===== [3] UCC perftest 用法探测 ====="
ucc_perftest -h 2>&1 | head -20

echo
echo "===== [4] UCX 点对点带宽（tag_bw，跨节点）====="
timeout 60 ucx_perftest -c 0 -t tag_bw 2>&1 | tail -8 || echo "(ucx_perftest 参数需按 -h 调整)"

echo "===== done $(date) ====="
