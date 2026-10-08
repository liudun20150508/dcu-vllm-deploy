#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
#SBATCH -J v4pro-20n80c
#SBATCH -p kshdnormal
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 08:00:00
#SBATCH -o $WS/logs/v4pro-20n80c-%j.out
#SBATCH -e $WS/logs/v4pro-20n80c-%j.err

# DeepSeek-V4-Pro-0813 多机推理（20 节点 × 4 DCU = 80 卡，TP=4 PP=20 DP=1 EP=4）
# 显存账：1.6T 参数，fp4 专家 + fp8 注意力 ≈ 830GB 权重 → 10.4GB/卡；
#   单卡 16GB × util0.85 = 13.6GB 预算，余 ~3GB 给 fp8 KV（max_model_len 8192）+ 激活。
#   若 OOM：把 -N 加到 24（PP=24，8.7GB/卡）或降 MAXLEN 到 4096。
# 注意：61 层 / PP=20 是不均匀切分（每 stage 3~4 层），vllm 0.21 支持；
#   若报 PP 切分错误，唯一的均匀切分是 PP=61，不要试其它数。
# 坑位 1/4/6/7/8/9 修复见 run_serve_node.sh 头部注释。
set -u
export MODEL_DIR=$HOME/models/DeepSeek-V4-Pro-0813
export SERVED_NAME=DeepSeek-V4-Pro-0813
export TP=4 PP=20 MAXLEN=8192
export LOG_TAG=v4pro-vllm
RUNNER=$WS/scripts/deploy/slurm/run_serve_node.sh
LOG_DIR=$WS/logs

module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache

MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
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

echo "########## v4pro 20n80c launch on $HOST ($(date)) ##########"
echo "nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "master=$MASTER_ADDR:$MASTER_PORT  parallel: TP=$TP PP=$PP DP=1 EP=$TP"
echo "RDMA: NCCL_IB_HCA=mlx5（本集群 Mellanox 200G×4，无 shca 插件）"
echo "model: $(du -sh $MODEL_DIR 2>/dev/null | cut -f1)"

# node0 起完整 vllm serve（API），node1-19 起 --headless
srun --export=ALL --ntasks-per-node=1 --cpus-per-task=32 \
  bash -c 'if [ "$SLURM_NODEID" = "0" ]; then H=0; else H=1; fi; exec "$0" "$SLURM_NODEID" "$1" "$2" "$H"' \
  "$RUNNER" "$MASTER_ADDR" "$MASTER_PORT" &
SRV=$!
echo "srun pid=$SRV"

# 832GB 权重经共享文件系统加载到 20 节点较慢，等 ready 放宽到 60 分钟，
# 每 2 分钟扫一次各节点日志做 fail-fast。
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
    if grep -alE "zmq.error|Cannot assign requested address|RuntimeError|FATAL|died with exit code|Timed out waiting for engine core|HSACOError|ImportError" \
         "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -1 | grep -q .; then
      echo "  ✗ 节点日志出现致命错误（第 $((i*30)) 秒），提前收尾"; break
    fi
    echo "  …第 $((i*30)) 秒仍未就绪（日志: $(ls "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | wc -l)/20）"
  fi
done

PY=$HOME/miniconda3/envs/das185/bin/python
if [ "$ready" = 1 ]; then
  echo "=== ① /health ==="
  curl -s "http://${HOST}:8000/health"; echo
  echo "=== ② /v1/completions 冒烟 ==="
  curl -s --max-time 900 "http://${HOST}:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"prompt\":\"用一句话解释什么是RDMA。\",\"max_tokens\":64,\"temperature\":0.6}" \
    | "$PY" -c "import sys,json; d=json.load(sys.stdin); print('choices:', len(d.get('choices',[]))); print('text:', (d.get('choices') or [{}])[0].get('text','')[:300]); print('usage:', d.get('usage'))" 2>&1 \
    || echo "  响应解析失败（见原始 body）"
  echo "=== ③ 保持服务运行（剩余 walltime 内可访问 node0:8000）==="
  while kill -0 $SRV 2>/dev/null; do sleep 60; done
  echo "服务已退出"
else
  echo "--- 各节点 vllm 日志尾部 ---"
  for n in $(seq 0 19); do
    L=$(ls -t "$LOG_DIR"/${LOG_TAG}-node${n}-${SLURM_JOB_ID}.log 2>/dev/null | head -1)
    [ -n "$L" ] || continue
    echo "########## node$n ($L) ##########"
    tail -15 "$L"
  done
  echo "--- 关键错误行 ---"
  grep -ahE "Traceback|Error|error:|Exception|RuntimeError|ValueError|AssertionError|Unsupported|Unrecognized|Unknown|GLIBC|Failed|OOM|out of memory" \
    "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -40
  kill $SRV 2>/dev/null
fi
echo "########## v4pro 20n80c done $(date) ##########"
