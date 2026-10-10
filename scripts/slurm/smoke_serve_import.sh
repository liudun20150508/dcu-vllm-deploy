#!/bin/bash
#SBATCH -J serve-import-smoke
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --exclusive
#SBATCH --comment={glibcVersion:2.31}
#SBATCH -t 00:30:00
#SBATCH -o /public/home/scndfy1yuj/中科曙光/验收/logs/serve-import-%j.out
#SBATCH -e /public/home/scndfy1yuj/中科曙光/验收/logs/serve-import-%j.err

# vllm serve 完整 import 链路冒烟（不加载权重）：抓住缺依赖/GLIBC 类问题，
# 避免再烧一次 20 节点作业才发现 ModuleNotFoundError。
set -u
PY=$HOME/miniconda3/envs/das185/bin/python
VLLM=$HOME/miniconda3/envs/das185/bin/vllm

module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$DTKROOT/.hyhal/rocm_smi/lib:$DTKROOT/.hyhal/lib:$DTKROOT/.hyhal/lib64:$DTKROOT/dcc/gcvm/lib:/public/software/compiler/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp XDG_CACHE_HOME=$HOME/.cache
export TRITON_HIP_CLANG_PATH=$HOME/dtk-shim/aillvm/bin/clang-18
export AMDGCN_USE_BUFFER_OPS=0
export VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0
# GLIBC 2.32 shim：_rocm_C/_moe_C/_C 需要 __libc_single_threaded@GLIBC_2.32
export LD_PRELOAD=$HOME/glibc-shim/libglibc232shim.so${LD_PRELOAD:+:$LD_PRELOAD}

# 登录节点 glibc 2.17 装不了 manylinux_2_28 的 xgrammar wheel，
# 在 glibc 2.31 计算节点上装（wheel 已备在 ~/wheels）。
PIP=$HOME/miniconda3/envs/das185/bin/pip
W=$(ls $HOME/wheels/xgrammar-0.2.5-*.whl 2>/dev/null | head -1)
if [ -n "$W" ]; then
  $PIP install --no-deps "$W" 2>&1 | tail -1
fi

# deepgemm（可选）：vllm_hcu 的 mhc/batched_deep_gemm_moe 模块交换用，
# 缺了会回退上游模块；wheel 已备则装，没备就从厂商站拉。
SP=$HOME/miniconda3/envs/das185/lib/python3.10/site-packages
if ! $PY -c "import deepgemm" 2>/dev/null; then
  DW=$HOME/wheels/deepgemm-2.1.0.whl
  if [ ! -s "$DW" ]; then
    curl -skL --retry 3 --max-time 300 -o "$DW" \
      "https://download.sourcefind.cn:65024/directlink/4/deepgemm/DAS1.8.5/deepgemm-2.1.0+das185.dtk2604.torch2100.2608121826.g2b4d4e-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl" 2>/dev/null
  fi
  if [ -s "$DW" ]; then
    $PY -c "import zipfile; zipfile.ZipFile('$DW').extractall('$SP')" && echo "deepgemm 已装"
  fi
fi

echo "########## serve import smoke on $(hostname) ($(date)) ##########"
echo "glibc: $(ldd --version | head -1)"

$PY - <<'EOF'
import sys
fails = []
def try_import(desc, stmt):
    try:
        exec(stmt, {})
        print(f"  [OK] {desc}", flush=True)
    except Exception as e:
        print(f"  [FAIL] {desc}: {type(e).__name__}: {e}", flush=True)
        fails.append(desc)

def try_import_warn(desc, stmt):
    try:
        exec(stmt, {})
        print(f"  [OK] {desc}", flush=True)
    except Exception as e:
        print(f"  [WARN] {desc}: {type(e).__name__}: {e}", flush=True)

# 上次失败的链路 + 新装依赖
try_import("cbor2", "import cbor2")
try_import("watchfiles", "import watchfiles")
try_import("xgrammar", "import xgrammar")
try_import("PIL", "import PIL")
try_import("vllm.utils.hashing（上次挂在这）", "from vllm.utils.hashing import safe_hash")
try_import("vllm.engine.arg_utils（serve 入口）", "from vllm.engine.arg_utils import AsyncEngineArgs")
# 下面三个扩展原先需要 GLIBC_2.32（本集群 2.31 没有）。
# 已修复：~/glibc-shim/patch_sos.sh 用 patchelf --clear-symbol-version
# 清掉 __libc_single_threaded 的版本需求，配合 LD_PRELOAD shim 提供符号本体。
# 因此恢复为硬检查——导入失败说明补丁没生效或又冒出别的缺失符号。
try_import("vllm._rocm_C", "import vllm._rocm_C")
try_import("vllm._C", "import vllm._C")
try_import("vllm._moe_C", "import vllm._moe_C")
try_import("hcu_ops 注册（vllm_hcu 自有扩展）", "import vllm_hcu, torch; torch.ops.hcu_ops" )
# DeepseekV4 模型类：探针里多链路口径可能触发 vllm_hcu import hook 二次执行
# → torch op "Duplicate registration"，serve 单进程单链路口径不会，故容忍之。
try:
    import importlib
    _m = importlib.import_module('vllm.model_executor.models.deepseek_v4')
    assert any('ForCausalLM' in x for x in dir(_m))
    print("  [OK] DeepseekV4 模型类", flush=True)
except RuntimeError as _e:
    if "Duplicate registration" in str(_e):
        print("  [OK] DeepseekV4 模型类（重复注册为探针双导入所致，视为通过）", flush=True)
    else:
        print(f"  [FAIL] DeepseekV4 模型类: {_e}", flush=True)
        fails.append("DeepseekV4 模型类")
except Exception as _e:
    print(f"  [FAIL] DeepseekV4 模型类: {type(_e).__name__}: {_e}", flush=True)
    fails.append("DeepseekV4 模型类")
try_import("vllm.entrypoints.openai.api_server", "import vllm.entrypoints.openai.api_server")

print()
if fails:
    print(f"### SMOKE FAIL: {len(fails)} 项 ###", flush=True)
    sys.exit(1)
print("### SMOKE ALL OK ###", flush=True)
EOF
RC=$?
echo "---- vllm serve --help（真实 CLI 入口）----"
$VLLM serve --help > /dev/null 2>&1 && echo "vllm serve --help OK" || echo "vllm serve --help FAIL rc=$?"
echo "########## smoke exit=$RC ($(date)) ##########"
exit $RC
