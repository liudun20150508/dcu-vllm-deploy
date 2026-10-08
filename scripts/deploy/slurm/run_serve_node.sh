#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
# 每节点 1 个进程：由 srun 逐节点拉起（kshdnormal 版）。
# args: <node_rank> <master_addr> <master_port> <headless(0|1)>
# env:  MODEL_DIR SERVED_NAME TP PP MAXLEN LOG_TAG 由 sbatch 包装脚本导出。
set -u
ENVDIR=$HOME/miniconda3/envs/das185/bin
VLLM=$ENVDIR/vllm
LOG_DIR=$WS/logs

NR=$1
MASTER_ADDR=$2
MASTER_PORT=$3
HEADLESS=$4
JOBID=${SLURM_JOB_ID:-0}
LOG="$LOG_DIR/${LOG_TAG}-node${NR}-${JOBID}.log"

# 坑6（zksg-all 871335）：强制节点本地 /tmp，避免继承登录节点 TMPDIR 导致
#   HIP/aiter JIT 写临时文件 ENOENT。
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
# 坑7（871391）：triton HCU 需要 aillvm clang-18（本集群 dtk-26.04 未装，
#   已解包到 ~/dtk-shim；triton compiler.py 的两个不兼容 flag 已在环境内 patch）。
export TRITON_HIP_CLANG_PATH=$HOME/dtk-shim/aillvm/bin/clang-18
# 坑8：aillvm(dtk-llvm-6.3.3) buffer rsrc intrinsic 签名不一致，关 buffer ops。
export AMDGCN_USE_BUFFER_OPS=0
# 坑9（871453）：fp8 权重 fast-woa 分支 dtype 错误，关掉走 rocm_inv_rope_einsum。
export VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0
# 坑21（10-04，探针 123623914）：triton launcher 的 .c 实为 C++（HIP 头），
# CC 不设会 fallback 到 gcc 按 C 编译炸（template 语法错误）。
# 配合 dupfix 的 make_llir arch-spoof（gfx906→gfx938）才是完整药方。
export CC=$DTKROOT/bin/hipcc
[ -x "$CC" ] || export CC=g++
# 坑22（10-04，作业 123624277）：aiter 的 MHC tilelang 核（norm_fn_kernel.py:132 /
# pre_norm_fn_splitk_kernel.py:87）T.gemm(k_pack=2, use_tf32=True) → f32 k_dim=4×2=8，
# tilelang make_mfma_load_layout 只认 4/16/32/64 → ValueError: k_dim must be...。
# 关 custom ops：hcu 上 is_rocm()==True（_enum=ROCM），MHC 走 torch 参考实现、
# MLA 走 BF16 参考 wo_a 路径、multi-stream 关闭。先求通，性能后议。
export VLLM_HCU_USE_CUSTOM_OPS=0
# 坑27（10-04，作业 123637382）：flash_mla 包（CUDA 编译）在集群不存在，
# MLA prefill/decode 调它的 raise 桩 → worker 死、生成请求永久挂起。
# decode 用 hcu 自带 torch 参考实现（rocm_forward_decode_fallback）；
# prefill 已在 deepseek_v4_attention.py 接上同款参考实现（同一开关）。
export VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK=1
# 坑24（10-04，作业 123632044）：32 个 worker 共享 $HOME/.triton 缓存（PARASTOR
# 网络盘 flock/rename 语义坏），同一 kernel 并发写缓存互相踩（FileExistsError:
# File exists / FileNotFoundError）。改节点本地 /tmp：节点内 4 worker 用本地锁互斥，
# 跨节点各编各的（代价：每节点编译一份，约多花一次编译时间）。
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLURM_NODEID:-0}

# GLIBC shim：vllm _rocm_C/_moe_C/_C.abi3.so 链接了 glibc 2.32 的
# __libc_single_threaded 符号，本集群 glibc 2.31 节点没有 →
# LD_PRELOAD 自家编译的 shim（~/glibc-shim/libglibc232shim.so，导出版本化符号）。
export LD_PRELOAD=$HOME/glibc-shim/libglibc232shim.so${LD_PRELOAD:+:$LD_PRELOAD}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_LOGGING_LEVEL=INFO

# 本集群是 Mellanox mlx5_0..3（不是文档集群的自研 shca），RCCL 走内置 IB verbs，
# 不要设 NCCL_NET_PLUGIN=librccl-net-shca.so（那是 shca 插件，本集群没有）。
# 坑10（RCCL 墙，2026-10-03 rccl_probe/probe4 实测定案）：本集群 mlx5_1/2/3 三个口
# 对 RCCL QP 建立是坏的（ibv_modify_qp EINVAL errno 22 → proxy Connect retcode 3
# → unhandled system error），只有 mlx5_0 正常。单 rank/节点时 NCCL 自动选 Dev 0
# 所以能通；vllm 每节点 4 worker 时 rank 被分摊到 4 个口，分到坏口的必挂——
# 这就是 flash 4n(123534066)/v4pro 20n(123502569) 引擎初始化失败的根因。
# 修复：钉死 mlx5_0（probe4 F 测试 8 rank 全绿验证）。代价：四平面带宽降为单平面，
# 先跑通再优化。
export NCCL_IB_HCA=mlx5_0
# INFO 只限 INIT/NET 子系统：定位 123502569 RCCL proxy Connect retcode 3 的真实原因
# （选网口/IB/errno），又不至于全子系统刷屏。全链路跑通后可改回 WARN。
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

NNODES=${SLURM_NNODES:-1}

COMMON=(
  --model "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --tensor-parallel-size "$TP"
  --pipeline-parallel-size "$PP"
  --data-parallel-size 1
  --enable-expert-parallel
  --nnodes "$NNODES"
  --node-rank "$NR"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  # 坑4（871128）：EMULATION 被 Mxfp4MoEMethod 硬拒绝；auto 在 ROCm/DCU 上选
  #   AITER_MXFP4_BF16。
  --moe-backend auto
  --kv-cache-dtype fp8
  --max-model-len "$MAXLEN"
  # 坑25（10-04，作业 123633678）：V4 的 SWA 缓存 block 硬编码 64 令牌
  # （sparse_swa.py 两边注释都写明"与 C4A KV block [256//4] 共享同一物理张量"），
  # SWA 页 = 64×584B；默认 block-size 16 时 C4A 页 = (16//4)×584 = 2336B < SWA 页
  # → kv_cache_utils._get_kv_cache_groups_uniform_groups 断言
  # max(swa_page) <= max(full_mla_page) 炸。block-size 256 时 C4A 页 = 64×584 相等。
  --block-size 256
  # 坑26（10-04，作业 123635758）：16G 卡 profile 峰值后仅剩 1.1 GiB 给 KV cache，
  # 不够 8192 长度单请求的 ~1 GiB+ → "No available memory for the cache blocks"。
  # 0.85→0.95 提预算；MAXLEN 由包装脚本降到 4096 减 profile 峰值与 KV 需求。
  --gpu-memory-utilization 0.95
  --enforce-eager
  # 坑13（amdsmi 缺失，10-03 作业 123548768 实测）：das185 环境没有 amdsmi 模块，
  #   vllm_hcu/platforms/hcu.py:65 的 import 失败后 amdsmi_init 等名字根本不存在，
  #   worker 走到 init_device → CustomAllreduce → is_fully_connected 时
  #   wrapper 调 amdsmi_init() → NameError（那条 WARNING 其实是致命的，不是噪音）。
  #   禁掉 custom allreduce（走 RCCL）即可绕过；先跑通，性能优化时再考虑
  #   装/补 amdsmi 把 custom allreduce 拿回来。
  --disable-custom-all-reduce
  --port 8000
)

if [ "$HEADLESS" = "1" ]; then
  echo "[node$NR] headless start pid=$$ host=$(hostname) master=$MASTER_ADDR:$MASTER_PORT tp=$TP pp=$PP $(date)" | tee -a "$LOG"
  exec "$VLLM" serve --headless "${COMMON[@]}" >> "$LOG" 2>&1
else
  echo "[node$NR] api-server start pid=$$ host=$(hostname) master=$MASTER_ADDR:$MASTER_PORT tp=$TP pp=$PP $(date)" | tee -a "$LOG"
  exec "$VLLM" serve "${COMMON[@]}" >> "$LOG" 2>&1
fi
