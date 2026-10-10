#!/bin/bash
# 8 卡 gfx936 真实权重冒烟启动器（本机 crdnotebook，非 slurm）。
# 用法：bash 验收/slurm/flash0731_8card_smoke.sh
set -u
WS=/root/private_data/中科曙光
LOG=/root/flash0731_8card_smoke.log

source /opt/dtk-26.04/env.sh
unset CUDA_HOME CUDA_PATH                       # 坑4：防 tilelang 误判 cuda
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TRITON_HIP_CLANG_PATH=/root/das185/dtk-shim/aillvm/bin/clang-18
export AMDGCN_USE_BUFFER_OPS=0
export CC=$DTKROOT/bin/hipcc                     # 坑21：triton launcher 的 .c 实为 C++
export VLLM_HCU_USE_CUSTOM_OPS=0                 # 坑22：MHC 走 torch 参考实现
export VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0      # 坑9：fp8 fast-woa dtype 错
export VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK=1  # 坑27：decode 走参考实现
export PYTHONPATH="$WS/验收/slurm/dupfix${PYTHONPATH:+:$PYTHONPATH}"  # 坑12/14
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

echo "########## flash0731 8card smoke start $(date) ##########" | tee -a "$LOG"
/root/das185/env/bin/python "$WS/验收/slurm/flash0731_8card_smoke.py" >> "$LOG" 2>&1
echo "########## flash0731 8card smoke end $(date) rc=$? ##########" | tee -a "$LOG"
