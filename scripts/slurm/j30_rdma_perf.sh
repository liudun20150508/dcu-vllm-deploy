#!/bin/bash
#SBATCH -J acc-j30-rdmaperf
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:1
#SBATCH -t 00:30:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j30-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j30-%j.err

# J30（优先 2 / 职责 4）：RDMA Verbs 性能调试 —— perftest 单变量调优
#
# 原则：**每次只改一个参数**，其余全固定，这样才知道到底是哪个因素在起作用。
#   A. 三种操作对照（WRITE 推 / READ 拉 / SEND 双边）
#   B. MTU 扫描（512 → 4096）
#   C. QP 数扫描（多 QP 并行能不能吃满）
#   D. tx depth 扫描
#   E. 延迟（小消息）
#
# 这样做的意义：JD 职责 4 要"优化通信路径、数据传输及**网络资源利用效率**" ——
# 得先知道这块自研卡的单卡上限在哪、哪个参数是瓶颈。
set -u
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
HOSTS=($(scontrol show hostname $SLURM_NODELIST))
N1=${HOSTS[0]}; N2=${HOSTS[1]}
DEV=shca_0
echo "########## J30 RDMA 性能调试 on $N1 <-> $N2 ($(date)) ##########"
echo "工具：$(ib_write_bw --version 2>&1 | head -1)"
echo "自研卡：$DEV  active_mtu=$(ibv_devinfo -d $DEV | grep active_mtu | awk '{print $2}')"

# 一对 server/client，只跑一次测试
pair() {  # $1=tool  $2=附加参数  $3=说明
  local tool="$1" args="$2" desc="$3"
  local srvlog=/tmp/j30_srv.log clilog=/tmp/j30_cli.log
  ssh -o StrictHostKeyChecking=no $N1 "$tool -d $DEV -F $args" > "$srvlog" 2>&1 &
  local pid=$!
  sleep 4
  ssh -o StrictHostKeyChecking=no $N2 "$tool -d $DEV -F $args $N1" > "$clilog" 2>&1
  wait $pid 2>/dev/null
  # 取最后一个（最大消息）结果行
  echo "  $desc"
  grep -E "^ *[0-9]+ +[0-9]+" "$clilog" | tail -2 | sed 's/^/      /'
  grep -E "BW average|MsgRate|t_avg|latency" "$clilog" | tail -2 | sed 's/^/      /'
}

echo
echo "===== A. 三种 RDMA 操作对照（8MB 消息）====="
for t in ib_write_bw ib_read_bw ib_send_bw; do
  pair "$t" "-s 8388608 -n 2000" "$t"
done

echo
echo "===== B. MTU 扫描（4MB 消息，单 QP）====="
for m in 512 1024 2048 4096; do
  pair ib_write_bw "-s 4194304 -n 2000 -m $m" "MTU=$m"
done

echo
echo "===== C. QP 数扫描（8MB，看多 QP 能否叠加）====="
for q in 1 2 4 8; do
  pair ib_write_bw "-s 8388608 -n 1000 -q $q" "QP=$q"
done

echo
echo "===== D. tx depth 扫描（4MB）====="
for tx in 16 64 128 256; do
  pair ib_write_bw "-s 4194304 -n 1000 -t $tx" "tx_depth=$tx"
done

echo
echo "===== E. 延迟（8 字节，小消息）====="
for t in ib_write_lat ib_send_lat; do
  pair "$t" "-s 8 -n 5000" "$t(8B)"
done

echo
echo "===== F. 消息大小全扫描（WRITE，看带宽拐点）====="
pair ib_write_bw "-a -n 500" "全尺寸扫描"

echo
echo "===== 参考：官方 perftest 峰值（grep 全部 client 输出）====="
echo "########## J30 done $(date) ##########"
