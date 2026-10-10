#!/bin/bash
#SBATCH -J acc-j38-v2lite
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:2
#SBATCH -t 00:30:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j38-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j38-%j.err

# J38：vLLM + V2-Lite（依赖已在 j33 装进共享 home，这里只跑推理，跳过重装）
#   修复点：amdsmi 替身新增 amdsmi_topo_get_link_type（抛 AmdSmiException，
#   让 vLLM is_fully_connected 返回 False → 退回 RCCL 建 TP 组，不再 TypeError）
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
# 强制 Triton MLA：FlashMLA 需要 sm90a(CUDA) 的 _flashmla_C，gfx936 上没有。
# ★ 关键：本版 vLLM 的 VLLM_USE_FLASH_MLA 默认是 "1"（True），会无条件选中 FlashMLA，
#   必须显式关掉，否则上面 TRITON_MLA 也压不住它。
export VLLM_USE_FLASH_MLA=0
export VLLM_ATTENTION_BACKEND=TRITON_MLA
# triton 的 AMD backend 默认找 /opt/rocm/llvm/bin/ld.lld，DTK 装在别处。
# 用 TRITON_HIP_LLD_PATH 指到 DTK 自带的 lld。
export TRITON_HIP_LLD_PATH=/public/software/compiler/dtk-26.04/llvm/bin/ld.lld
PY=/usr/local/python3.12/bin/python3
# flashinfer 顶层 `import pynvml`，DCU 上没装 nvidia-ml-py。
# vLLM 自带一份完整拷贝（third_party/pynvml.py），直接复制到 user-site 供顶层 import。
cp -f ~/.local/lib/python3.12/site-packages/vllm/third_party/pynvml.py \
      ~/.local/lib/python3.12/site-packages/pynvml.py 2>/dev/null || true
echo "########## J38 on $(hostname) ($(date)) ##########"

echo "=== 环境自检 ==="
$PY - <<'PYEOF'
import amdsmi
print("amdsmi_topo_get_link_type 存在:", hasattr(amdsmi, "amdsmi_topo_get_link_type"))
import torch, vllm
print("torch", torch.__version__, "| vllm", vllm.__version__)
import numpy, transformers
print("numpy", numpy.__version__, "| transformers", transformers.__version__)
PYEOF

cd $WS/experiments
export VLLM_LOGGING_LEVEL=DEBUG
FULL=$WS/验收/logs/j38-full-${SLURM_JOB_ID}.log
VLLM_MODEL=$HOME/models/DeepSeek-V2-Lite VLLM_TP=2 \
  $PY v2lite_infer.py > "$FULL" 2>&1
rc=$?
echo "[v2lite_infer.py] exit=$rc"
echo "----- 尾部 40 行 -----"
grep -a -v -E "ProcessGroupNCCL|watchdog|heart beat|^\s*$" "$FULL" | tail -40
echo "----- 关键行（OK/tok/s/错误）-----"
grep -a -n -E "=== OK|tok/s|合计|加载完成|Traceback|Error|error:|Exception|RuntimeError|TypeError" "$FULL" | head -40
echo "########## J38 done $(date) ##########"
