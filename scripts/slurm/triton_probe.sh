#!/bin/bash
#SBATCH -J triton-probe
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:30:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/triton-probe-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/triton-probe-%j.err

# 坑21 探针（123620800）：inductor 编译 vocab 掩码逐元素核时
# triton/backends/amd/compiler.py make_llir → PassManager::run failed。
# 最小化复现：4 类核逐个试（float 逐元素 / int64 位运算 / inductor int64 / inductor float）。
set -u
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
export TRITON_HIP_CLANG_PATH=$HOME/dtk-shim/aillvm/bin/clang-18
export AMDGCN_USE_BUFFER_OPS=0
# launcher .c 实际是 C++（HIP 头），CC 不设会 fallback 到 gcc 按 C 编译炸
export CC=$DTKROOT/bin/hipcc
[ -x "$CC" ] || export CC=g++
export LD_PRELOAD=$HOME/glibc-shim/libglibc232shim.so${LD_PRELOAD:+:$LD_PRELOAD}

PY=$HOME/miniconda3/envs/das185/bin/python
$PY $HOME/中科曙光/验收/slurm/triton_probe.py
