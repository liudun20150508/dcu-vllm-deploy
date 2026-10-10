#!/bin/bash
#SBATCH -J diag-numpy
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j36-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j36-%j.err

# J36（诊断，非验收项）：定位 `RuntimeError: Numpy is not available`
#   j33 的 WorkerProc 挂在 torch.distributed 的 `tensor.numpy()` 上，
#   根因疑似系统 numpy 2.4.0 与「用 NumPy 1.x 编译的海光版 torch」ABI 不匹配。
#   本作业只做诊断：① 现状 ② warning 原文 ③ 隔离验证 numpy 1.26.4（--target，不动全局）
set -u
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J36 numpy 诊断 on $(hostname) ($(date)) ##########"

echo "===== ① 现状 ====="
$PY - <<'PYEOF'
import numpy, torch
print(f"  numpy: {numpy.__version__}  ({numpy.__file__})")
print(f"  torch: {torch.__version__}  ({torch.__file__})")
t = torch.zeros(3)
try:
    print("  torch.zeros(3).numpy() ->", t.numpy())
except Exception as e:
    print("  ✗ tensor.numpy() 失败:", type(e).__name__, str(e)[:200])
PYEOF

echo "===== ② 'Failed to initialize NumPy' 原文 ====="
$PY - <<'PYEOF' 2>&1 | head -25
import warnings
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    import torch
for w in ws:
    if "NumPy" in str(w.message) or "numpy" in str(w.message):
        print("  warning:", str(w.message)[:500].replace("\n", " ¶ "))
PYEOF

echo "===== ③ 隔离验证：numpy 1.26.4（--target /tmp/np126，不动全局）====="
rm -rf /tmp/np126
$PY -m pip install -q --no-index --no-deps --target /tmp/np126 $HOME/hcu/wheels/deps/numpy-1.26.4-*.whl 2>&1 | tail -2
PYTHONPATH=/tmp/np126 $PY - <<'PYEOF'
import numpy, torch
print(f"  numpy: {numpy.__version__}  ({numpy.__file__})")
t = torch.zeros(3)
try:
    print("  ✓ torch.zeros(3).numpy() ->", t.numpy())
except Exception as e:
    print("  ✗ 仍失败:", type(e).__name__, str(e)[:200])
PYEOF

echo "===== ④ numpy 1.26.4 下 numba/llvmlite 是否仍正常 ====="
PYTHONPATH=/tmp/np126 $PY - <<'PYEOF' 2>&1 | tail -6
import numba, numpy, llvmlite
print(f"  numba={numba.__version__} numpy={numpy.__version__} llvmlite={llvmlite.__version__}")
from numba import njit
@njit
def f(a):
    return a.sum()
print("  njit 自检 ->", f(numpy.arange(5.0)))
PYEOF
echo "########## J36 done $(date) ##########"
