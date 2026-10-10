#!/bin/bash
#SBATCH -J rccl-probe4
#SBATCH -p kshdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:20:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/rccl-probe4-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/rccl-probe4-%j.out

# L0 第二轮：复刻 vllm 几何（每节点 4 rank/4 卡）。第一轮(123535705)证明
# 1 rank/节点时 4 种配置全通，但 vllm 4 rank/节点失败时各 rank 被分到不同
# HCA(Dev 1/3 出现 ibv_modify_qp EINVAL)。本轮验证：
#   E: 8 rank(2 节点×4 rank)，NCCL_IB_HCA=mlx5   → 预期复现 vllm 失败
#   F: 8 rank，NCCL_IB_HCA=mlx5_0（全钉 Dev 0）   → 候选修复，预期通过
#   G: 1 rank/节点，NCCL_IB_HCA=mlx5_1            → Dev 1 单口健康检查
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
BASEPORT=$((32000 + SLURM_JOB_ID % 2000))

echo "########## rccl probe4 (4 rank/node) nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ') ##########"
echo "master=$MASTER_ADDR base_port=$BASEPORT $(date)"

run8() {
  local name=$1 port=$2; shift 2
  echo ""
  echo "===== test ${name} (8ranks, port ${port}) start $(date +%H:%M:%S) ====="
  env "$@" timeout 300 srun --export=ALL -N2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 \
    bash -c "export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID; export RANK=\$SLURM_PROCID WORLD_SIZE=8 MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${port}; ${PY} ${SCRIPT}" 2>&1
  echo "===== test ${name} exit=$? end $(date +%H:%M:%S) ====="
}

run8 E-replicate-8rank $((BASEPORT))
run8 F-pin-mlx5_0-8rank $((BASEPORT+1)) NCCL_IB_HCA=mlx5_0

echo ""
echo "===== test G-single-rank-mlx5_1 (port $((BASEPORT+2))) start $(date +%H:%M:%S) ====="
env NCCL_IB_HCA=mlx5_1 timeout 300 srun --export=ALL -N2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=8 \
  bash -c "export ROCR_VISIBLE_DEVICES=0; export RANK=\$SLURM_PROCID WORLD_SIZE=2 MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=$((BASEPORT+2)); ${PY} ${SCRIPT}" 2>&1
echo "===== test G exit=$? end $(date +%H:%M:%S) ====="

echo ""
echo "########## 汇总 ##########"
grep -aE 'ALLREDUCE|===== test .* exit=|ibv_modify_qp failed|Failed to execute operation Connect' /public/home/scndfy1yuj/中科曙光/验收/logs/rccl-probe4-${SLURM_JOB_ID}.out 2>/dev/null
echo "########## rccl probe4 done $(date) ##########"
