#!/bin/bash
#SBATCH -J probe-dup-import
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:20:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/probe-dup-import-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/probe-dup-import-%j.out

# 探针：定位 worker 里官方 vllm deepseek_v4_attention 模块是谁先 import 的
# （导致 PluggableLayer 重名注册 Duplicate op name）。
set -u
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
export LD_PRELOAD=$HOME/glibc-shim/libglibc232shim.so
export VLLM_WORKER_MULTIPROC_METHOD=spawn

$HOME/miniconda3/envs/das185/bin/python - <<'EOF' 2>&1
import sys, traceback, builtins

WATCH = {
    "vllm.model_executor.layers.deepseek_v4_attention",
    "vllm.model_executor.models.deepseek_v4",
    "vllm_hcu.model_executor.layers.deepseek_v4_attention",
    "vllm_hcu.models.deepseek_v4",
}

_orig_import = builtins.__import__
def rec_import(name, *a, **k):
    m = _orig_import(name, *a, **k)
    if name in WATCH:
        print(f"\n===== IMPORT {name} =====", file=sys.stderr)
        traceback.print_stack(file=sys.stderr)
        print("===== END IMPORT STACK =====\n", file=sys.stderr)
    return m
builtins.__import__ = rec_import

print(">>> step1: import vllm", flush=True)
import vllm
from vllm import ModelRegistry
print(">>> step2: import vllm_hcu + activate plugin", flush=True)
import vllm_hcu
vllm_hcu.hcu_platform_plugin()
vllm_hcu.hcu_platform_register_model()
vllm_hcu.hcu_platform_register_ops()

from vllm.model_executor.custom_op import op_registry
print(">>> op_registry 已有目标名?", "deepseek_v4_multi_head_latent_attention" in op_registry)
print(">>> 官方 attention 模块已加载?", "vllm.model_executor.layers.deepseek_v4_attention" in sys.modules)
print(">>> 官方 model 模块已加载?", "vllm.model_executor.models.deepseek_v4" in sys.modules)
print(">>> HCU attention 模块已加载?", "vllm_hcu.model_executor.layers.deepseek_v4_attention" in sys.modules)

m = ModelRegistry.models["DeepseekV4ForCausalLM"]
print(">>> 注册入口 module_name:", getattr(m, "module_name", "?"), "class:", getattr(m, "class_name", "?"))
print(">>> step3: 模拟 worker load_model_cls()", flush=True)
try:
    cls = m.load_model_cls()
    print(">>> load_model_cls OK:", cls)
except Exception:
    print(">>> load_model_cls FAILED:")
    traceback.print_exc()
EOF
echo "########## probe done $(date) ##########"
