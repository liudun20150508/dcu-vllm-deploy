#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
# 8 卡 gfx936 Base（fp8）真实权重冒烟启动器（本机 crdnotebook，非 slurm）。
# 用法：bash 验收/slurm/flash_base_8card_smoke.sh
set -u
WS=$WS
LOG=/root/flash_base_8card_smoke.log
ulimit -c 0   # 禁用 core dump（避免崩溃时写满 NFS 持久盘）

source /opt/dtk-26.04/env.sh
unset CUDA_HOME CUDA_PATH                       # 坑4：防 tilelang 误判 cuda
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TRITON_HIP_CLANG_PATH=$DAS185/dtk-shim/aillvm/bin/clang-18
export AMDGCN_USE_BUFFER_OPS=0
export CC=$DTKROOT/bin/hipcc                     # 坑21：triton launcher 的 .c 实为 C++
export VLLM_HCU_USE_CUSTOM_OPS=0                 # 坑22：MHC 走 torch 参考实现
export VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0      # 坑9：fp8 fast-woa dtype 错
export VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK=1  # 坑27：decode 走参考实现
export PYTHONPATH="$WS/patches${PYTHONPATH:+:$PYTHONPATH}"  # 坑12/14
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

echo "########## flashbase 8card smoke start $(date) ##########" | tee -a "$LOG"
$DAS185/env/bin/python "$WS/scripts/deploy/slurm/flash_base_8card_smoke.py" >> "$LOG" 2>&1
echo "########## flashbase 8card smoke end $(date) rc=$? ##########" | tee -a "$LOG"
