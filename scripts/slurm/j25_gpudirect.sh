#!/bin/bash
#SBATCH -J acc-j25-gpudirect
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j25-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j25-%j.err

# J25（优先 2）：GPUDirect RDMA —— 让远端网卡把数据直接写进本端 DCU 显存
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
HOSTS=($(scontrol show hostname $SLURM_NODELIST))
N1=${HOSTS[0]}; N2=${HOSTS[1]}
# ssh 是非登录 shell，必须显式把 DTK 环境带过去（否则 hipMalloc 会 invalid device ordinal）
DTKENV='source /etc/profile.d/modules.sh 2>/dev/null; module load compiler/dtk/26.04; export HSA_FORCE_FINE_GRAIN_PCIE=1'
echo "########## J25 on $N1 <-> $N2 $(date) ##########"

echo "===== [1] 编译（g++ + libibverbs + libamdhip64）====="
cd $WS/gpu-rdma-demo && make clean >/dev/null 2>&1; make 2>&1 | tail -4
ls -la gpu_rdma 2>&1

echo
echo "===== [2] 跨节点跑：server 在 $N1，client 在 $N2 ====="
ssh -o StrictHostKeyChecking=no $N1 "$DTKENV; cd $WS/gpu-rdma-demo && ./gpu_rdma server 0.0.0.0 shca_0" > /tmp/j25_srv.log 2>&1 &
SSHPID=$!
sleep 10
ssh -o StrictHostKeyChecking=no $N2 "$DTKENV; cd $WS/gpu-rdma-demo && ./gpu_rdma client $N1 shca_0" > /tmp/j25_cli.log 2>&1
wait $SSHPID 2>/dev/null

echo "--- server 侧（$N1）---"; cat /tmp/j25_srv.log
echo "--- client 侧（$N2）---"; cat /tmp/j25_cli.log

echo
echo "===== [3] 结论判据 ====="
if grep -q "GPUDIRECT-WROTE-DCU-MEM" /tmp/j25_srv.log 2>/dev/null; then
  echo "  ★★★ GPUDirect RDMA 成功：远端网卡直接把数据写进了本端 DCU 显存（CPU 未参与搬运）"
else
  echo "  ✗ 未观察到显存被远端写入"
fi
echo "########## J25 done $(date) ##########"
