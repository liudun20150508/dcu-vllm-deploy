#!/bin/bash
#SBATCH -J probe-dup-serve
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:40:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/probe-dup-serve-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/probe-dup-serve-%j.out

# 坑12 复现探针（单节点 4 卡，TP=4 PP=1 EP=4）：验证 Duplicate op name 是否与
# 多节点/PP 无关，并用 duptrace/sitecustomize.py 抓第二次注册的完整调用栈。
# 期望：worker init 阶段就崩（不必等模型加载完），日志里出现 [DUP-TRACE] DUPLICATE。
set -u
export MODEL_DIR=$HOME/models/DeepSeek-V4-Flash-0731
export SERVED_NAME=DeepSeek-V4-Flash-0731
export TP=4 PP=1 MAXLEN=8192
export LOG_TAG=dup-serve
RUNNER=$HOME/中科曙光/验收/slurm/run_serve_node.sh
LOG_DIR=$HOME/中科曙光/验收/logs
HOOK_DIR=$HOME/中科曙光/验收/slurm/duptrace

module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
# 钩子注入：sitecustomize.py 会被 site.py 自动 import（含 _run_in_subprocess 子进程）
export PYTHONPATH="$HOOK_DIR${PYTHONPATH:+:$PYTHONPATH}"

HOST=$(hostname)
MASTER_ADDR=$HOST
MASTER_PORT=$((28000 + SLURM_JOB_ID % 2000))

echo "########## probe-dup-serve 1n4c on $HOST ($(date)) ##########"
echo "PYTHONPATH=$PYTHONPATH  TP=$TP PP=$PP"

srun --export=ALL --ntasks-per-node=1 --cpus-per-task=32 \
  bash -c 'exec "$0" "$SLURM_NODEID" "$1" "$2" 0' \
  "$RUNNER" "$MASTER_ADDR" "$MASTER_PORT" &
SRV=$!
echo "srun pid=$SRV"

# 只要崩了就收尾；最多等 8 分钟
for i in $(seq 1 16); do
  sleep 30
  if ! kill -0 $SRV 2>/dev/null; then echo "  srun 已退出（第 $((i*30)) 秒）"; break; fi
  if curl -s --max-time 3 "http://${HOST}:8000/health" >/dev/null 2>&1; then
    echo "  ready（第 $((i*30)) 秒）——未复现 duplicate"; break
  fi
  if [ $((i % 2)) -eq 0 ]; then echo "  …第 $((i*30)) 秒"; fi
done

kill $SRV 2>/dev/null
echo "--- [DUP-TRACE] 注册记录 ---"
grep -ah "DUP-TRACE" "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -60
echo "--- 关键错误行 ---"
grep -ahE "AssertionError|Duplicate op name|failed to be inspected|Traceback|ValueError|ImportError" \
  "$LOG_DIR"/${LOG_TAG}-node*-${SLURM_JOB_ID}.log 2>/dev/null | head -20
echo "########## probe-dup-serve done $(date) ##########"
