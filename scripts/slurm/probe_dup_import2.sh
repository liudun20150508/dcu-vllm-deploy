#!/bin/bash
#SBATCH -J probe-dup-import2
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:20:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/probe-dup-import2-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/probe-dup-import2-%j.out

# 探针2：复刻 worker init_device 之前的导入路径，找出谁先 import 官方
# vllm.model_executor.layers.deepseek_v4_attention。
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
        tb = traceback.extract_stack()
        # 只打印栈中非 <stdin> 的帧，定位真实调用者
        for fr in tb:
            if '<stdin>' in fr.filename or 'rec_import' in fr.name:
                continue
            print(f"  {fr.filename}:{fr.lineno} in {fr.name}", file=sys.stderr)
        print("===== END =====", file=sys.stderr)
    return m
builtins.__import__ = rec_import

def snapshot(tag):
    from vllm.model_executor.custom_op import op_registry
    print(f"[{tag}] target in op_registry=", "deepseek_v4_multi_head_latent_attention" in op_registry,
          "| official attn loaded=", "vllm.model_executor.layers.deepseek_v4_attention" in sys.modules,
          "| official model loaded=", "vllm.model_executor.models.deepseek_v4" in sys.modules,
          "| hcu attn loaded=", "vllm_hcu.model_executor.layers.deepseek_v4_attention" in sys.modules,
          flush=True)

print(">>> step1 import vllm", flush=True)
import vllm
from vllm import ModelRegistry
snapshot("after vllm")

print(">>> step2 import vllm_hcu + plugin", flush=True)
import vllm_hcu
vllm_hcu.hcu_platform_plugin()
vllm_hcu.hcu_platform_register_model()
vllm_hcu.hcu_platform_register_ops()
snapshot("after plugin")

print(">>> step3 import vllm_hcu.v1.hcu_model_runner (worker 会构造它)", flush=True)
import vllm_hcu.v1.hcu_model_runner as hmr
snapshot("after hcu_model_runner import")

print(">>> step4 初始化单卡分布式 (模拟 worker init_worker_distributed_environment)", flush=True)
import os, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29555")
torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
snapshot("after dist init")

print(">>> step5 模拟 worker load_model_cls()", flush=True)
m = ModelRegistry.models["DeepseekV4ForCausalLM"]
try:
    cls = m.load_model_cls()
    print(">>> load_model_cls OK:", cls, flush=True)
except Exception:
    print(">>> load_model_cls FAILED:")
    traceback.print_exc()
snapshot("after load")
EOF
echo "########## probe2 done $(date) ##########"
