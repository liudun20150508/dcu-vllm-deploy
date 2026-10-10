#!/bin/bash
#SBATCH -J acc-j26-mini
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH -t 00:25:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j26-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j26-%j.err

# J26（优先 5+6+7）：Megatron 张量并行真实训练步 + KV Cache 真机实测
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
L=$WS/验收/logs
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J26 on $(hostname) $(date) ##########"
cd $WS/experiments
$PY -m torch.distributed.run --standalone --nproc_per_node=2 megatron_mini_train.py \
    > $L/j26-full.log 2>&1
echo "exit=$?  完整日志 $L/j26-full.log"
echo "--- 真实报错（最后 25 行）---"
tail -25 $L/j26-full.log
echo "--- 关键结果行 ---"
grep -a -E "step[0-9]|loss|⇒|KV Cache|prefill|decode|TP 切分|===" $L/j26-full.log | head -20
echo "########## J26 done $(date) ##########"
