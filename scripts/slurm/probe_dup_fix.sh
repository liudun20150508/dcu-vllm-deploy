#!/bin/bash
#SBATCH -J probe-dup-fix
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:40:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/probe-dup-fix-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/probe-dup-fix-%j.out

# 坑12 修复验证（单节点 4 卡，TP=4 PP=1 EP=4）：注入 dupfix/sitecustomize.py
# 让 vllm_hcu 的 op 覆盖官方同名 op。
# 判据：① 日志出现 [DUP-FIX] OVERRIDE；② 不再出现 "Duplicate op name" /
#   "failed to be inspected"。
# 注意：单节点 4 卡装不下 156G 模型（39G/卡 > 16G），权重加载阶段必然 OOM——
# 本作业只验证 duplicate 这道坎是否越过，全链路要跑 4 节点。
set -u
export MODEL_DIR=$HOME/models/DeepSeek-V4-Flash-0731
export SERVED_NAME=DeepSeek-V4-Flash-0731
export TP=4 PP=1 MAXLEN=8192
export LOG_TAG=dup-fix
RUNNER=$HOME/中科曙光/验收/slurm/run_serve_node.sh
LOG_DIR=$HOME/中科曙光/验收/logs
HOOK_DIR=$HOME/中科曙光/验收/slurm/dupfix

module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
export PYTHONPATH="$HOOK_DIR${PYTHONPATH:+:$PYTHONPATH}"

HOST=$(hostname)
MASTER_ADDR=$HOST
MASTER_PORT=$((28000 + SLURM_JOB_ID % 2000))

echo "########## probe-dup-fix 1n4c on $HOST ($(date)) ##########"
echo "PYTHONPATH=$PYTHONPATH  TP=$TP PP=$PP"

srun --export=ALL --ntasks-per-node=1 --cpus-per-task=32 \
  bash -c 'exec "$0" "$SLURM_NODEID" "$1" "$2" 0' \
  "$RUNNER" "$MASTER_ADDR" "$MASTER_PORT" &
SRV=$!
echo "srun pid=$SRV"

for i in $(seq 1 30); do
  sleep 30
  if ! kill -0 $SRV 2>/dev/null; then echo "  srun 已退出（第 $((i*30)) 秒）"; break; fi
  if curl -s --max-time 3 "http://${HOST}:8000/health" >/dev/null 2>&1; then
    echo "  ready（第 $((i*30)) 秒）"; break
  fi
  if grep -ahq "Duplicate op name\|failed to be inspected" \
       "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null; then
    echo "  ✗ 仍撞 duplicate（第 $((i*30)) 秒）"; break
  fi
  if [ $((i % 4)) -eq 0 ]; then echo "  …第 $((i*30)) 秒"; fi
done

kill $SRV 2>/dev/null
echo "--- [DUP-FIX] 裁决记录 ---"
grep -ah "DUP-FIX" "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -20
echo "--- duplicate / inspect 相关 ---"
grep -ahE "Duplicate op name|failed to be inspected|OVERRIDE|SKIP" \
  "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -10
echo "--- 其他致命错误（预期 OOM，单节点装不下 156G）---"
grep -ahE "AssertionError|ValueError|ImportError|out of memory|OutOfMemory|RuntimeError" \
  "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | grep -v "DUP-FIX" | head -8
echo "########## probe-dup-fix done $(date) ##########"
