#!/bin/bash
#SBATCH -J probe-arch
#SBATCH -p kshdnormal
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH -t 00:10:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/probe-arch-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/probe-arch-%j.out

# 坑15 架构普查：vllm_hcu 的 aiter 门槛是 on_gfx93x() = gfx936/gfx938，
# 而默认分配到的 b14/b15/b16 节点是 gfx906（老 DCU）→ MXFP4 MoE 无后端。
# 历史日志里 f11r1n18/19 是 gfx936，当前分区有 f11r2n[00-14,16-19] 组。
# 目的：确认哪些节点组是 gfx936（且 glibc 版本），好把 serve 钉到那些节点。
# 用法：sbatch probe_arch.sh          （普查默认分配 + f11r2n 组）
#       sbatch -w f11r2n[00-07] probe_arch.sh   （只查指定组）
set -u
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
export LD_PRELOAD=$HOME/glibc-shim/libglibc232shim.so${LD_PRELOAD:+:$LD_PRELOAD}

PY=$HOME/miniconda3/envs/das185/bin/python
echo "########## probe-arch on $(hostname) nodes=$SLURM_NODELIST ($(date)) ##########"

srun --export=ALL --ntasks-per-node=1 --cpus-per-task=4 bash -c '
  echo "=== $(hostname) ==="
  echo "  glibc     : $(ldd --version 2>/dev/null | head -1)"
  echo -n "  DCU count : "; ls /dev/kfd >/dev/null 2>&1 && echo -n "kfd ok; " ; rocm-smi --showproductname 2>/dev/null | grep -ci "card" || echo "n/a"
  rocm-smi --showproductname 2>/dev/null | grep -aiE "series|model|sku" | head -4 | sed "s/^\s*/  product   : /"
  '"$PY"' - <<EOF 2>&1 | sed "s/^/  /"
import torch
try:
    p = torch.cuda.get_device_properties(0)
    print("gcnArchName :", p.gcnArchName)
    print("device name :", p.name)
    print("capability  :", torch.cuda.get_device_capability(0))
    print("total mem   : %.1f GiB" % (p.total_memory / 1024**3))
except Exception as e:
    print("torch probe failed:", repr(e))
EOF
'
echo "--- 汇总（arch x 节点）---"
grep -aE "^=== |gcnArchName" /public/home/scndfy1yuj/中科曙光/验收/logs/probe-arch-${SLURM_JOB_ID}.out \
  | paste - - 2>/dev/null | sed "s/\s\+/ /g" | sort -k4
echo "########## probe-arch done $(date) ##########"
