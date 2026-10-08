#!/bin/bash
# ============================================================================
# rccl_allreduce.sh — RCCL all_reduce 正确性验证启动器
#
# 验证「前置条件 5：RCCL 通信」。单机多卡用 torchrun，跨机用 Slurm srun。
#
# 用法：
#   单机 8 卡：  bash scripts/verify/rccl_allreduce.sh --np 8
#   跨机 Slurm： sbatch scripts/verify/rccl_allreduce.sh（内含 #SBATCH 模板）
# ============================================================================
set -u

# ---- 环境（复用 install/env.sh 的 DTK/HIP 设置）----
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$WS/scripts/install/env.sh" 2>/dev/null || {
  module purge 2>/dev/null; source /etc/profile.d/modules.sh 2>/dev/null
  module load compiler/dtk/26.04
  export HSA_FORCE_FINE_GRAIN_PCIE=1
}
PY="${PYTHON_BIN:-python3}"
SCRIPT="$WS/scripts/verify/rccl_allreduce.py"

# ---- 解析 --np ----
NP=8
while [ $# -gt 0 ]; do
  case "$1" in
    --np) NP="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

echo "########## RCCL all_reduce 验证 (np=$NP) $(date) ##########"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  # ---- 跨机 Slurm 路径：每节点 1 进程，共 SLURM_NTASKS 个 rank ----
  NNODES=${SLURM_NNODES:-1}
  MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
  MASTER_ADDR=$(scontrol show node "$MASTER_HOST" 2>/dev/null | grep -oP 'NodeAddr=\K\S+' | head -1)
  [ -z "$MASTER_ADDR" ] && MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1; exit}')
  PORT=$((30000 + SLURM_JOB_ID % 2000))
  echo "slurm: nnodes=$NNODES master=$MASTER_ADDR:$PORT"
  srun --export=ALL -N"$NNODES" --ntasks="$NNODES" --cpus-per-task=8 --gres=dcu:1 \
    bash -c "export ROCR_VISIBLE_DEVICES=0 RANK=\$SLURM_PROCID WORLD_SIZE=$NNODES \
             MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$PORT; $PY $SCRIPT"
else
  # ---- 单机路径：torchrun standalone ----
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NP" "$SCRIPT"
fi

echo "########## 验证结束 ##########"
echo "解读：每个 rank 输出 ALLREDUCE OK sum=<np> 即 RCCL 通信正常（✅ 前置条件 5）。"
