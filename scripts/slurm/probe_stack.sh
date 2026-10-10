#!/bin/bash
#SBATCH -J probe-stack
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:20:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/probe-stack-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/probe-stack-%j.err

# kshdnormal 全链路探针：glibc2.31 节点 + DTK26.04 + das185 环境 + triton/aillvm
# 与 test_vllm.sh 同源的环境变量（10/1 已验证能 import），补齐 zksg-all 坑位修复。
set -u
PY=$HOME/miniconda3/envs/das185/bin/python

module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04

export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH

# 坑6（zksg-all 871335）：TMPDIR 指节点本地 /tmp，避免继承登录节点会话目录
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
# 坑7：triton HCU 后端需要 aillvm clang-18（本集群 dtk-26.04 未装，已解包到 ~/dtk-shim）
export TRITON_HIP_CLANG_PATH=$HOME/dtk-shim/aillvm/bin/clang-18
# 坑8：aillvm buffer rsrc intrinsic 签名不一致，关 buffer ops
export AMDGCN_USE_BUFFER_OPS=0

echo "########## probe-stack on $(hostname) ($(date)) ##########"
echo "glibc: $(ldd --version | head -1)"
echo "clang-18: $($TRITON_HIP_CLANG_PATH --version 2>/dev/null | head -1)"

# 注意：sbatch 会把脚本复制到 spool 执行，BASH_SOURCE 不可靠，必须用绝对路径
$PY /public/home/scndfy1yuj/中科曙光/验收/slurm/probe_stack.py
RC=$?
echo "########## probe-stack exit=$RC ($(date)) ##########"
exit $RC
