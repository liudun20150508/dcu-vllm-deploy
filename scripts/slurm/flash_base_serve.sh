#!/bin/bash
#SBATCH -J flashbase-fp8
#SBATCH -p kshdnormal
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 04:00:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/flashbase-fp8-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/flashbase-fp8-%j.err

# DeepSeek-V4-Flash-Base 全链路验证（8 节点 × 4 DCU = 32 卡，TP=4 PP=8 DP=1 EP=4）
# 模型 295GB（fp8 专家 e4m3+block128 + fp8 注意力）→ 约 9.2GB/卡 权重。
# 坑17：官方 fp8 唯一解（0731 fp4 已三轮证死）。gfx906 上两道软门槛，
# 见下方三组实验开关（勿同开）。其余坑位 1/4/6/7/8/9/10/12/13 修复见 run_serve_node.sh。
# 注意：Base 不是 instruct，冒烟 prompt 用续写式（"人工智能是"），别问指令。
set -u
export MODEL_DIR=$HOME/models/DeepSeek-V4-Flash-Base
export SERVED_NAME=DeepSeek-V4-Flash-Base
# 坑26：MAXLEN 8192→4096（冒烟够用；减 profile 峰值 + KV cache 需求，
# 16G 卡显存太紧，先跑通再调回 8192）
export TP=4 PP=8 MAXLEN=4096
export LOG_TAG=flashbase-vllm
RUNNER=$HOME/中科曙光/验收/slurm/run_serve_node.sh
HOOK_DIR=$HOME/中科曙光/验收/slurm/dupfix
LOG_DIR=$HOME/中科曙光/验收/logs

module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
# 坑12 修复：让 vllm_hcu 的 op 覆盖官方同名 op（见 dupfix/sitecustomize.py）
export PYTHONPATH="$HOOK_DIR${PYTHONPATH:+:$PYTHONPATH}"

# === 坑17 实验开关（勿同开）===
# 第0轮（基线，123611077 已跑）：都不开 → AITER 拒=设备门(hip)、TRITON 拒=量化方案门。
#   证据已落袋，且 [DUP-FIX] SUPPORT 日志确认两道门在全部 8 节点同款拒绝。
# 第A轮（已证死，10-04 二进制证据）：aiter 的 MoE .so（module_moe.so 等）只含
#   gfx936/gfx938 code object，无 gfx906 → 强开 FORCE_AITER 过门后 kernel 也加载不了。
# 第B轮（本轮）：开下面这条 → 赌 vllm 内置 TRITON fp8 MoE 核（反量化到 bf16 算，老卡可跑）。
#   10-04 修正：FORCE_FP8 之前只改 RocmPlatform.supports_fp8，但 DCU 上 current_platform
#   是 HCUPlatform（自带 supports_fp8=on_gfx938()），补丁落空 → dupfix 已加 hcu 补丁。
export DUPFIX_FORCE_FP8=1

MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
# 本集群节点间管理网可达（scontrol NodeAddr），ZMQ/TCPStore 直接用它。
MASTER_ADDR=$(scontrol show node "$MASTER_HOST" 2>/dev/null | grep -oP 'NodeAddr=\K\S+' | head -1)
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1}' | head -1)

pick_free_port() {
  local base=$1 p
  for ((p=base; p<base+200; p++)); do
    if ! ss -tln 2>/dev/null | grep -q ":$p "; then echo "$p"; return; fi
  done
  echo $(( base + SLURM_JOB_ID % 2000 ))
}
MASTER_PORT=$(pick_free_port $((28000 + SLURM_JOB_ID % 2000)))
HOST=$(hostname)

echo "########## flashbase serve 8n32c on $HOST ($(date)) ##########"
echo "nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "master=$MASTER_ADDR:$MASTER_PORT  parallel: TP=$TP PP=$PP DP=1 EP=$TP"
echo "model: $(du -sh $MODEL_DIR 2>/dev/null | cut -f1)"
echo "switches: FORCE_AITER=${DUPFIX_FORCE_AITER:-off} FORCE_FP8=${DUPFIX_FORCE_FP8:-off}"

srun --export=ALL --ntasks-per-node=1 --cpus-per-task=32 \
  bash -c 'if [ "$SLURM_NODEID" = "0" ]; then H=0; else H=1; fi; exec "$0" "$SLURM_NODEID" "$1" "$2" "$H"' \
  "$RUNNER" "$MASTER_ADDR" "$MASTER_PORT" &
SRV=$!
echo "srun pid=$SRV"

echo "=== 等 ready（最多 60 分钟）==="
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
    # 排除 [DUP-FIX] 噪音行（坑14 兜底日志里含 RuntimeError 字样但非致命，
    # 123619946 曾因此误杀正在正常加载权重的作业）
    if grep -ahE "zmq.error|Cannot assign requested address|RuntimeError|FATAL|died with exit code|Timed out waiting for engine core|HSACOError|ImportError" \
         "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | grep -avE "\[DUP-FIX\]" | head -1 | grep -q .; then
      echo "  ✗ 节点日志出现致命错误（第 $((i*30)) 秒），提前收尾"; break
    fi
    echo "  …第 $((i*30)) 秒仍未就绪（日志: $(ls "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | wc -l)/8）"
  fi
done

PY=$HOME/miniconda3/envs/das185/bin/python
if [ "$ready" = 1 ]; then
  echo "=== ① /health ==="
  curl -s "http://${HOST}:8000/health"; echo
  echo "=== ② /v1/completions 冒烟（Base 模型用续写式 prompt）==="
  curl -s --max-time 600 "http://${HOST}:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"prompt\":\"人工智能是\",\"max_tokens\":64,\"temperature\":0.6}" \
    | "$PY" -c "import sys,json; d=json.load(sys.stdin); print('choices:', len(d.get('choices',[]))); print('text:', (d.get('choices') or [{}])[0].get('text','')[:300]); print('usage:', d.get('usage'))" 2>&1 \
    || echo "  响应解析失败（见原始 body）"
  echo "=== ③ 保持服务运行（剩余 walltime 内可访问 node0:8000）==="
  while kill -0 $SRV 2>/dev/null; do sleep 60; done
  echo "服务已退出"
else
  echo "--- 各节点 vllm 日志尾部 ---"
  for n in $(seq 0 7); do
    L=$(ls -t "$LOG_DIR"/${LOG_TAG}-node${n}-${SLURM_JOB_ID}.log 2>/dev/null | head -1)
    [ -n "$L" ] || continue
    echo "########## node$n ($L) ##########"
    tail -25 "$L"
  done
  echo "--- 关键错误行 ---"
  grep -ahE "Traceback|Error|error:|Exception|RuntimeError|ValueError|AssertionError|Unsupported|Unrecognized|Unknown|GLIBC|Failed|OOM|out of memory" \
    "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -40
  echo "--- DUP-FIX 支持检查日志 ---"
  grep -ahE "DUP-FIX.*(SUPPORT|GATE|FP8-GATE)" \
    "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -30
  kill $SRV 2>/dev/null
fi
echo "########## flashbase serve done $(date) ##########"
