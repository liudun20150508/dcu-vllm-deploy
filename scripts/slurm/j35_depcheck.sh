#!/bin/bash
#SBATCH -J acc-j35-depcheck
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j35-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j35-%j.err

# J35：不再"逐个 import 试错" —— 直接让 pip check 把缺失依赖一次列全
set -u
W=$HOME/hcu/wheels
module purge; source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
PY=/usr/local/python3.12/bin/python3
echo "########## J35 依赖体检 ($(date)) ##########"

echo "=== 1) 补上确定需要的 ==="
$PY -m pip install --user --no-index --no-deps \
  $W/deps/sentencepiece-*.whl $W/deps/tiktoken-*.whl 2>&1 | tail -2

echo
echo "=== 2) pip check：列出所有缺失/不匹配 ==="
$PY -m pip check 2>&1 | head -40

echo
echo "=== 3) 逐个 import 关键包（每项独立，失败不互相影响）==="
for m in torch vllm transformers accelerate sentencepiece tiktoken numpy; do
  printf "  %-14s: " "$m"
  $PY -c "import $m; print(getattr($m,'__version__','OK'))" 2>&1 | tail -1
done
echo "########## J35 done $(date) ##########"
