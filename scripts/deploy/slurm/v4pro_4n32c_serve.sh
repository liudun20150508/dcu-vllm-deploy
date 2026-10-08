#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
#SBATCH -J v4pro-4n32c
#SBATCH -p hx1hdnormal01
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:8
#SBATCH -t 04:00:00
#SBATCH -o $WS/logs/v4pro-4n32c-%j.out
#SBATCH -e $WS/logs/v4pro-4n32c-%j.err

# DeepSeek-V4-Pro-0813 多机多卡启动（4 节点 × 8 DCU = 32 卡）
#   TP=8, PP=2, DP=2, EP=16（enable-expert-parallel，EP 组 = DP×TP 覆盖每个 PP stage）
#   MoE 后端 emulation（fp4 专家权重逐次反量化）；KV fp8；RCCL 走 shca RDMA 四平面
# 坑5（871212）: aiter 缺 shuffle_weight_a16w4/shuffle_scale_a16w4 → ImportError。
#   已在 aiter/ops/shuffle.py 末尾补函数（备份/移植说明见 验收/patches/）。
# 分区已从 hx1hdnormal 改名为 hx1hdnormal01（sbatch -p hx1hdnormal 会报 Invalid account/partition）。
set -u
WS=$WS
RUNNER=$WS/scripts/deploy/slurm/run_v4pro_node.sh
LOG_DIR=$WS/logs
PY=$WORK/miniconda3-py310/envs/das185/bin/python
SHIM=$HOME/glibc-shim/libglibc232shim.so

module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load app/rccl/dtk-25.04/26.01.1
module load app/rccl/shca_rdma_plugins/v8
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
export LD_PRELOAD=$SHIM
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# 坑6（871335）: sbatch 会继承登录节点环境的 TMPDIR（例如 Reasonix 会话的
#   /tmp/reasonix-session-tmp-*），计算节点上没有该目录 → HIP/aiter JIT 写
#   HSACO 临时文件时 ENOENT 崩（HSACOError: File operation failed）。
#   统一改用节点本地 /tmp，XDG 缓存也指回稳定路径。
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
export XDG_CACHE_HOME=$HOME/.cache

# 坑7（871391）: triton HCU 后端需要 dtk 的 AI LLVM（aillvm）；集群 dtk-26.04 未装，
#   而 dtk/llvm/clang-17 不认识 triton 传的 HCU -mllvm 参数 → HSACOError。
#   aillvm 已从 download.sourcefind.cn 的 dtk_llvm.run 解包到 $WORK/dtk-shim，
#   并同步 patch 了 triton/backends/hcu/compiler.py（跳过 aillvm 不认识的 2 个 flag）。
export TRITON_HIP_CLANG_PATH=$WORK/dtk-shim/aillvm/bin/clang-18
# 坑8: aillvm(dtk-llvm-6.3.3) 的 buffer rsrc intrinsic 签名与 triton 不一致，
#   关 buffer ops 走普通 load/store，避免 "Intrinsic has incorrect argument type"。
export AMDGCN_USE_BUFFER_OPS=0
# 坑9（871453）: fp8 权重下 deepseek_v4_attention 的 fast-woa 分支拿 fp8 wo_a 直接做
#   bf16 einsum → "expected scalar type BFloat16 but found Float8_e4m3fn"。
#   关掉该分支，走 rocm_inv_rope_einsum（内部会把 wo_a 反量化）。
export VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0

MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)

# 坑1（870374 zmq.error: Cannot assign requested address tcp://NODE:54161）：
#   多机 c10d/TCPStore 与 vllm 内部 ZMQ 都必须走 fabric 12.x（别名 <host>i），
#   不能走管理网 10.6.x——worker 节点把 ZMQ 绑到 master 的管理网地址会直接失败。
#   解析策略：优先 <host>i；失败则按本集群规律 10.6.x.x → 12.6.x.x 推导。
FABRIC_IP=$(getent ahostsv4 "${MASTER_HOST}i" 2>/dev/null | awk '{print $1}' | head -1)
if [ -z "$FABRIC_IP" ]; then
  MGMT_IP=$(getent ahostsv4 "$MASTER_HOST" 2>/dev/null | awk '{print $1}' | head -1)
  FABRIC_IP=$(printf '%s\n' "$MGMT_IP" | sed 's/^10\./12./')
fi
MASTER_ADDR=$FABRIC_IP
DP_ADDR=$FABRIC_IP

# 坑2（端口类冲突预防）：master/dp 端口先在本节点确认空闲再定。
pick_free_port() {
  local base=$1 p
  for ((p=base; p<base+200; p++)); do
    if ! ss -tln 2>/dev/null | grep -q ":$p "; then echo "$p"; return; fi
  done
  echo $(( base + SLURM_JOB_ID % 2000 ))
}
MASTER_PORT=$(pick_free_port $((28000 + SLURM_JOB_ID % 2000)))
DP_RPC_PORT=$(pick_free_port $((31000 + SLURM_JOB_ID % 2000)))
HOST=$(hostname)

echo "########## v4pro 4n32c launch on $HOST ($(date)) ##########"
echo "nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "master_host=$MASTER_HOST  fabric_ip=$MASTER_ADDR  (管理网 10.6.x 一律不用)"
echo "parallel: TP=8 PP=2 DP=2 EP=16 (world_across_dp=32)  master=$MASTER_ADDR:$MASTER_PORT  dp=$DP_ADDR:$DP_RPC_PORT"
echo "layout: node0=API+DP协调器, node1/3=headless worker, node2=headless engine(dp rank 1)"
echo "RDMA: NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN  RCCL_NET_PLANE=$RCCL_NET_PLANE  shca_0..3"
echo "model: $(du -sh $HOME/models/DeepSeek-V4-Pro-0813 2>/dev/null | cut -f1)"

# 4 个进程（每节点 1 个）：node0 起完整 vllm serve（API+协调器），node1-3 起 --headless
# headless=1 当 SLURM_NODEID>0
srun --export=ALL --ntasks-per-node=1 --cpus-per-task=32 \
  bash -c 'if [ "$SLURM_NODEID" = "0" ]; then H=0; else H=1; fi; exec "$0" "$SLURM_NODEID" "$1" "$2" "$3" "$4" "$H"' \
  "$RUNNER" "$MASTER_ADDR" "$MASTER_PORT" "$DP_ADDR" "$DP_RPC_PORT" &
SRV=$!
echo "srun pid=$SRV"

# 坑3（870437 TimeoutError: engine core 600s 超时）：VLLM_ENGINE_READY_TIMEOUT_S=3600 已在上面导出，
#   892GB 权重从 /work2 加载较慢，等 ready 放宽到 60 分钟；同时每 2 分钟扫一次各节点日志做 fail-fast，
#   避免某个 headless 节点早退后还傻等到超时。
echo "=== 等 ready（最多 60 分钟，模型 892GB 加载较慢）==="
ready=0
for i in $(seq 1 120); do
  sleep 30
  if curl -s --max-time 3 "http://${HOST}:8000/health" >/dev/null 2>&1; then
    ready=1; echo "  ready（第 $((i*30)) 秒）"; break
  fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo "  ✗ srun 已退出（第 $((i*30)) 秒）——看各节点日志"; break
  fi
  if [ $((i % 4)) -eq 0 ]; then
    if grep -alE "zmq.error|Cannot assign requested address|RuntimeError|FATAL|died with exit code|Timed out waiting for engine core" \
         "$LOG_DIR"/v4pro-vllm-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -1 | grep -q .; then
      echo "  ✗ 检测到节点日志出现致命错误（第 $((i*30)) 秒），提前收尾排查"; break
    fi
    echo "  …第 $((i*30)) 秒仍未就绪（日志: $(ls "$LOG_DIR"/v4pro-vllm-node*-${SLURM_JOB_ID}.log 2>/dev/null | wc -l)/4）"
  fi
done

if [ "$ready" = 1 ]; then
  echo "=== ① /health ==="
  curl -s "http://${HOST}:8000/health"; echo
  echo "=== ② /v1/completions 冒烟 ==="
  curl -s --max-time 900 "http://${HOST}:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"DeepSeek-V4-Pro-0813","prompt":"用一句话解释什么是RDMA。","max_tokens":64,"temperature":0.6}' \
    | "$PY" -c "import sys,json; d=json.load(sys.stdin); print('choices:', len(d.get('choices',[]))); print('text:', (d.get('choices') or [{}])[0].get('text','')[:300]); print('usage:', d.get('usage'))" 2>&1 \
    || echo "  响应解析失败（见原始 body）"
  echo "=== ③ 保持服务运行（剩余 walltime 内可访问各节点 :8000）==="
  while kill -0 $SRV 2>/dev/null; do sleep 60; done
  echo "服务已退出"
else
  echo "--- 各节点 vllm 日志尾部 ---"
  for n in 0 1 2 3; do
    L=$(ls -t "$LOG_DIR"/v4pro-vllm-node${n}-${SLURM_JOB_ID}.log 2>/dev/null | head -1)
    [ -n "$L" ] || continue
    echo "########## node$n ($L) ##########"
    tail -25 "$L"
  done
  echo "--- 关键错误行 ---"
  grep -ahE "Traceback|Error|error:|Exception|RuntimeError|ValueError|AssertionError|Unsupported|Unrecognized|Unknown|GLIBC|Failed|OOM|out of memory" \
    "$LOG_DIR"/v4pro-vllm-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -40
  kill $SRV 2>/dev/null
fi
echo "########## v4pro 4n32c done $(date) ##########"
