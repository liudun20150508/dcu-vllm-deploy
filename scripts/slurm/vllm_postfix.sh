#!/bin/bash
#SBATCH -J vllm-postfix
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:15:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/vllm-postfix-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/vllm-postfix-%j.out

# 坑11（vllm_hcu 补丁墙，2026-10-03 定位）：das185 环境装 vllm_hcu 时漏跑了
# post_install——整套 vllm__*.patch.py 磁盘补丁从未落盘（vllm 包内 0 个 PATCHED_
# 标记），导致 flash 作业 worker 崩在
#   ImportError: cannot import name 'set_deepep_auto_use_low_latency' from vllm.forward_context
# （vllm__forward_context.patch.py 第 0 段本来就是要注入这个函数的）。
# 修复：在计算节点（glibc 2.31 + shim）补跑 python -m vllm_hcu.post_install。
set -u

module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
export LD_PRELOAD=$HOME/glibc-shim/libglibc232shim.so

SP=$HOME/miniconda3/envs/das185/lib/python3.10/site-packages

echo "########## vllm_hcu post_install 补跑 $(date) on $(hostname) ##########"
echo "--- 修复前状态 ---"
echo "forward_context PATCHED 标记数: $(grep -c 'PATCHED_' $SP/vllm/forward_context.py)"
ls -la $SP/vllm/model_executor/layers/fused_moe/modular_kernel.py 2>/dev/null | sed 's/.*modular_kernel/modular_kernel/'

echo "--- 执行 post_install ---"
$HOME/miniconda3/envs/das185/bin/python -m vllm_hcu.post_install 2>&1 | tail -20

echo "--- 修复后验证 ---"
echo "forward_context PATCHED 标记数: $(grep -c 'PATCHED_' $SP/vllm/forward_context.py)"
grep -n 'def set_deepep_auto_use_low_latency\|def get_deepep_auto_use_low_latency' $SP/vllm/forward_context.py
ls -la $SP/vllm/model_executor/layers/fused_moe/modular_kernel.py 2>/dev/null | sed 's/.*modular_kernel/modular_kernel/'
echo "全 vllm 包被打补丁文件数: $(grep -rl 'PATCHED_' $SP/vllm/ 2>/dev/null | wc -l)"
echo "########## vllm-postfix done $(date) ##########"