# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
# 海光 DCU 运行环境（source 这个文件后再跑 PyTorch）
#   用法：  source ~/hcu/env.sh
#
# DCU 版 torch 已经装进 user-site（~/.local/lib/python3.12/site-packages），
# 所以【不需要设 PYTHONPATH】—— 但运行时仍需要 DTK 提供的 HIP/HSA 库。

module purge 2>/dev/null || true
source /etc/profile.d/modules.sh 2>/dev/null || true
module load compiler/dtk/26.04

export HSA_FORCE_FINE_GRAIN_PCIE=1        # 海光必需，否则慢/不稳/挂起

# ★ 曙光集群(CentOS7)适配：module load 只设 DTKROOT，下面两条要手动补，否则 import torch 报缺库
#   ① DTK 自带 .hyhal 运行时库（librocm_smi64.so 等，module load 不会带上）
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$LD_LIBRARY_PATH
#   ② 系统 libstdc++ 太旧(缺 CXXABI_1.3.8)，用 gcc11 的新版
export LD_LIBRARY_PATH=/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
#   ③ libgcvm.so.17git（triton/megatron 需要，在 DTK 的 dcc 编译器目录下）
export LD_LIBRARY_PATH=$DTKROOT/dcc/gcvm/lib:$LD_LIBRARY_PATH

# 想用 RCCL 走自研网卡（跨节点时）再放开这两行：
# module load app/rccl/dtk-25.04/26.01.1
# export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so

echo "[env] DTK=$DTKROOT  HSA_FORCE_FINE_GRAIN_PCIE=1"
echo "[env] python: /usr/local/python3.12/bin/python3  (torch 2.5.1 + das.opt1.dtk2604, 已装进 user-site)"
