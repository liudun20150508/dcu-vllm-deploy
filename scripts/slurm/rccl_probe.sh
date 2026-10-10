#!/bin/bash
#SBATCH -J rccl-probe
#SBATCH -p kshdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:25:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/rccl-probe-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/rccl-probe-%j.out

# L0 纯 RCCL 探测（摘掉 vllm）：复现 flash/v4pro 的 RCCL proxy Connect 失败
# 123534066 flash 4n / 123502569 v4pro 20n 都死在 ibv_modify_qp EINVAL(22) →
# proxy Connect retcode 3。本作业用 2 节点×1 卡最小 allreduce 夹逼根因：
#   A: 复刻 vllm 环境（NCCL_IB_HCA=mlx5）          → 预期失败
#   B: HCA 自动选择（去掉 NCCL_IB_HCA）            → 若过=过滤器选错口
#   C: HCA 钉死 mlx5_0                             → 若过=特定口的问题
#   D: NCCL_NET=Socket（完全绕开 IB verbs）        → 若过=IB 层问题，网络本身通
set -u

module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
export LD_PRELOAD=$HOME/glibc-shim/libglibc232shim.so
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_HCA=mlx5

PY=$HOME/miniconda3/envs/das185/bin/python
SCRIPT=$HOME/中科曙光/验收/slurm/rccl_allreduce.py

MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
MASTER_ADDR=$(scontrol show node "$MASTER_HOST" 2>/dev/null | grep -oP 'NodeAddr=\K\S+' | head -1)
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1; exit}')
BASEPORT=$((30000 + SLURM_JOB_ID % 2000))

echo "########## rccl probe 2n on $(hostname) $(date) ##########"
echo "nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "master=$MASTER_ADDR base_port=$BASEPORT"

echo "=== ① 各节点 IB 端口状态 ==="
srun -N2 --ntasks=2 --cpus-per-task=2 bash -c '
echo "--- $(hostname) ---"
ibstat 2>/dev/null | grep -E "^CA |^Port [0-9]+|State:|Active MTU|Link layer|Base lid|Rate:" | head -40
for d in /sys/class/infiniband/mlx5_*; do
  for p in "$d"/ports/*; do
    [ -d "$p" ] || continue
    printf "  %s port %s: state=%s link=%s mtu=%s lid=%s\n" "$(basename $d)" "$(basename $p)" \
      "$(cat $p/state 2>/dev/null)" "$(cat $p/link_layer 2>/dev/null)" \
      "$(cat $p/active_mtu 2>/dev/null)" "$(cat $p/lid 2>/dev/null)"
  done
done
' 2>&1

run_test() {
  local name=$1 port=$2; shift 2
  echo ""
  echo "===== test ${name} (port ${port}) start $(date +%H:%M:%S) ====="
  env "$@" timeout 300 srun --export=ALL -N2 --ntasks=2 --cpus-per-task=8 --gres=dcu:1 \
    bash -c "export ROCR_VISIBLE_DEVICES=0; export RANK=\$SLURM_PROCID WORLD_SIZE=2 MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${port}; ${PY} ${SCRIPT}" 2>&1
  echo "===== test ${name} exit=$? end $(date +%H:%M:%S) ====="
}

run_test A-replicate-mlx5 $((BASEPORT))
run_test B-auto-hca      $((BASEPORT+1)) -u NCCL_IB_HCA
run_test C-pin-mlx5_0    $((BASEPORT+2)) NCCL_IB_HCA=mlx5_0
run_test D-socket-net    $((BASEPORT+3)) NCCL_NET=Socket

echo ""
echo "########## 汇总 ##########"
grep -aE 'ALLREDUCE|===== test .* exit=|ibv_modify_qp failed' /public/home/scndfy1yuj/中科曙光/验收/logs/rccl-probe-${SLURM_JOB_ID}.out 2>/dev/null
echo "########## rccl probe done $(date) ##########"
