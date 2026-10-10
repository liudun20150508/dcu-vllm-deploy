#!/bin/bash
#SBATCH -J acc-j31-bwprobe
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j31-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j31-%j.err

# J31（优先 2）：自写 verbs 程序的带宽 —— 与 perftest 同口径对照
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
HOSTS=($(scontrol show hostname $SLURM_NODELIST))
N1=${HOSTS[0]}; N2=${HOSTS[1]}
echo "########## J31 on $N1 <-> $N2 ($(date)) ##########"

echo "=== 编译（g++ + libibverbs）==="
cd $WS/rdma-demo && make clean >/dev/null 2>&1
g++ -std=c++17 -O2 -Wall -Wextra -o bw_probe bw_probe.cpp common.cpp -libverbs 2>&1 | head -8
ls -la bw_probe

run() {  # $1=msg bytes  $2=iters
  local msg=$1 iters=$2
  ssh -o StrictHostKeyChecking=no $N1 "cd $WS/rdma-demo && ./bw_probe server 0.0.0.0 shca_0 $msg $iters" > /tmp/j31_srv.log 2>&1 &
  local pid=$!
  sleep 4
  ssh -o StrictHostKeyChecking=no $N2 "cd $WS/rdma-demo && ./bw_probe client $N1 shca_0 $msg $iters" > /tmp/j31_cli.log 2>&1
  wait $pid 2>/dev/null
  echo "--- 消息 $(( msg / 1024 / 1024 )) MB × $iters 次 ---"
  grep -E "带宽|用时" /tmp/j31_cli.log | sed 's/^/    /'
  grep -E "完工信令" /tmp/j31_srv.log | sed 's/^/    /'
}

echo
echo "===== 自写 verbs WRITE 带宽（流水线深度 16）====="
run 1048576   500
run 4194304   500
run 8388608   500

echo
echo "===== 对照 perftest（同一张卡、同一天）====="
echo "    ib_write_bw 8MB = 46150 MB/s = 45.07 GB/s"
echo "    ib_write_bw 4MB = 42300 MB/s = 42.30 GB/s"
echo "    ⇒ 自写/官方 的百分比见上面输出"
echo "########## J31 done $(date) ##########"
