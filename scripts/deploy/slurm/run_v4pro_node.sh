#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
# 每节点 1 个进程：由 srun 逐节点拉起。
# args: <node_rank> <master_addr> <master_port> <dp_addr> <dp_rpc_port> <headless(0|1)>
set -u
ENVDIR=$WORK/miniconda3-py310/envs/das185/bin
VLLM=$ENVDIR/vllm
SHIM=$HOME/glibc-shim/libglibc232shim.so
MODEL=${MODEL_DIR:-$MODELS/DeepSeek-V4-Pro-0813}
LOG_DIR=$WS/logs

# 并行参数可通过环境变量覆盖（deploy.sh CLI 会传 TP/PP/DP/EP/MAXLEN）
TP=${TP:-8}
PP=${PP:-2}
DP=${DP:-2}
MAXLEN=${MAXLEN:-8192}

NR=$1
MASTER_ADDR=$2
MASTER_PORT=$3
DP_ADDR=$4
DP_RPC_PORT=$5
HEADLESS=$6
JOBID=${SLURM_JOB_ID:-0}
LOG="$LOG_DIR/v4pro-vllm-node${NR}-${JOBID}.log"

export LD_PRELOAD=$SHIM
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_LOGGING_LEVEL=INFO

# 坑6（871335）: 登录节点 TMPDIR 在计算节点不存在 → HIP/aiter JIT 写 HSACO
#   临时文件 ENOENT。强制用节点本地 /tmp，XDG 缓存指回稳定路径。
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
export XDG_CACHE_HOME=$HOME/.cache

# 坑7（871391）: triton HCU 后端需要 dtk 的 AI LLVM（aillvm）；集群 dtk-26.04 未装，
#   dtk/llvm/clang-17 不认识 triton 的 HCU -mllvm 参数 → HSACOError。
#   aillvm 已解包到 $WORK/dtk-shim；compiler.py 已 patch 跳过 2 个不兼容 flag。
export TRITON_HIP_CLANG_PATH=$WORK/dtk-shim/aillvm/bin/clang-18
# 坑8: aillvm buffer rsrc intrinsic 签名与 triton 不一致，关 buffer ops。
export AMDGCN_USE_BUFFER_OPS=0
# 坑9（871453）: fp8 权重下 fast-woa 分支用 fp8 wo_a 做 bf16 einsum 报 dtype 错误，
#   关掉，走 rocm_inv_rope_einsum（内部先反量化 wo_a）。
export VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0

COMMON=(
  --model "$MODEL"
  --served-model-name DeepSeek-V4-Pro-0813
  --tensor-parallel-size "$TP"
  --pipeline-parallel-size "$PP"
  --data-parallel-size "$DP"
  --data-parallel-size-local 1
  --enable-expert-parallel
  --nnodes "${NNODES:-4}"
  --node-rank "$NR"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --data-parallel-address "$DP_ADDR"
  --data-parallel-rpc-port "$DP_RPC_PORT"
  # 坑4（871128: "Unsupported mxfp4_backend for Mxfp4MoEMethod: EMULATION"）：
  #   emulation 被 Mxfp4MoEMethod 硬拒绝；用 auto → ROCm/DCU 上 DeepseekV4 会选
  #   AITER_MXFP4_BF16（TRITON_UNFUSED 在本机因 triton kernels 不可用被跳过，探针已验证）。
  --moe-backend auto
  --kv-cache-dtype fp8
  --max-model-len "$MAXLEN"
  --gpu-memory-utilization 0.85
  --enforce-eager
  --port 8000
)

if [ "$HEADLESS" = "1" ]; then
  echo "[node$NR] headless start pid=$$ host=$(hostname) master=$MASTER_ADDR:$MASTER_PORT dp=$DP_ADDR:$DP_RPC_PORT $(date)" | tee -a "$LOG"
  exec "$VLLM" serve --headless "${COMMON[@]}" >> "$LOG" 2>&1
else
  echo "[node$NR] api-server start pid=$$ host=$(hostname) master=$MASTER_ADDR:$MASTER_PORT dp=$DP_ADDR:$DP_RPC_PORT $(date)" | tee -a "$LOG"
  exec "$VLLM" serve "${COMMON[@]}" >> "$LOG" 2>&1
fi
