#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
# ============================================================
# 一键脚本：gfx936 单卡 DCU 环境适配 + 模型权重下载
#   下次开机（同账号 + gfx936 + DTK26.04）直接跑这一个脚本即可。
#   幂等：已装好/已下好的步骤会跳过，可反复重跑。
#
# 用法：
#   bash setup_all.sh            # 适配环境 + 下载 0731 和 Flash-Base
#   bash setup_all.sh --no-model # 只适配环境，不下载模型
# ============================================================
set -u
WS=$WS
D=$DAS185
DOWNLOAD_MODEL=1
[ "${1:-}" = "--no-model" ] && DOWNLOAD_MODEL=0

echo "########## [0/6] 系统依赖：python3.10 头文件 + virtualenv ##########"
# triton 编译 hip_utils.c 需要 Python.h（python3.10-minimal 不带，apt 源也没有）
if [ ! -f /usr/include/python3.10/Python.h ]; then
  cd /tmp
  curl -skL -o libpython3.10-dev.deb "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/pool/main/p/python3.10/libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb"
  curl -skL -o python3.10-dev.deb "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/pool/main/p/python3.10/python3.10-dev_3.10.12-1~22.04.18_amd64.deb"
  dpkg -i libpython3.10-dev.deb python3.10-dev.deb 2>/dev/null || true
  rm -f libpython3.10-dev.deb python3.10-dev.deb
fi
[ -f /usr/include/python3.10/Python.h ] && echo "  Python.h OK" || echo "  ⚠️ Python.h 仍缺失，triton 编译会失败"

# virtualenv（系统 python3.11 装，用于创建 py3.10 环境）
python3 -m virtualenv --version >/dev/null 2>&1 || python3 -m pip install -q virtualenv

# venv（py3.10，wheel 都是 cp310）
if [ ! -x "$D/env/bin/python" ]; then
  python3 -m virtualenv -p /usr/bin/python3.10 "$D/env"
fi
PY="$D/env/bin/python"
echo "  venv: $PY ($($PY --version 2>&1))"

echo "########## [1/6] 下载全家桶 wheel ##########"
bash "$WS/scripts/deploy/slurm/das185_dl_wheels.sh"

echo "########## [2/6] 安装 das wheel（--no-deps 防 resolver 拉错 torch）##########"
W="$D/wheels"
cd "$W"
$PY -m pip install -q --no-deps torch-2.10.0*.whl
$PY -m pip install -q --no-deps vllm-0.21.0*.whl vllm_hcu-0.21.0*.whl \
  aiter-0.1.5*.whl triton-3.6.0*.whl tilelang-0.1.9*.whl flash_mla-1.2.0*.whl
$PY -m pip install -q --no-deps torchvision-0.25.0*.whl \
  lightop-0.6.0*.whl deepgemm-2.1.0*.whl 2>/dev/null || true

echo "########## [3/6] 安装 python 依赖 ##########"
# 核心依赖（import vllm 必需）
$PY -m pip install -q "numpy==1.26.4" pyyaml typing_extensions packaging pydantic \
  filelock tqdm requests psutil einops aiohttp pyzmq msgspec cloudpickle cbor2 \
  py-cpuinfo blake3 cachetools diskcache pybase64 python-json-logger regex \
  setproctitle watchfiles prometheus_client sentencepiece 2>&1 | tail -2
# 模型加载 + 版本敏感（这些版本是踩坑定的，别乱动）
#   xgrammar==0.2.0：0.2.8 需要 apache-tvm-ffi>=0.1.11 会崩；0.2.0 兼容 vllm 钉的 0.1.9
#   apache-tvm-ffi==0.1.9：vllm 0.21.0 钉死；tilelang 也用 0.1.9（0.1.12 会 __dict__ 报错）
$PY -m pip install -q "transformers>=4.56.0,<5.0" safetensors tokenizers tiktoken \
  "xgrammar==0.2.0" "setuptools<80.0.0,>=77.0.3" "apache-tvm-ffi==0.1.9" z3-solver 2>&1 | tail -2
# vllm 其余依赖
$PY -m pip install -q "compressed-tensors==0.15.0.1" gguf "lark==1.2.2" \
  "lm-format-enforcer==0.11.3" ninja "numba==0.65.0" "outlines_core==0.2.14" \
  partial-json-parser peft pillow protobuf timm "model-hosting-container-standards<1.0.0" \
  setuptools-scm "tensorizer==2.10.1" 2>&1 | tail -2
# ★ 收尾：pip 装依赖时会把 torch/triton 换成 PyPI 版，必须 --force-reinstall 换回 das 版
$PY -m pip install -q --no-deps --force-reinstall torch-2.10.0*.whl
$PY -m pip install -q --no-deps --force-reinstall triton-3.6.0*.whl

echo "########## [4/6] 打补丁 + vllm_hcu post_install ##########"
$PY "$WS/scripts/deploy/slurm/das185_patch.py"
$PY -m vllm_hcu.post_install

echo "########## [5/6] aillvm（triton 需要 clang-18）##########"
if [ ! -d "$D/dtk-shim/aillvm" ]; then
  bash "$W/dtk_llvm.run" --dtk_dir "$D/dtk-shim" </dev/null || true
fi
ln -sfn /opt/dtk-26.04/amdgcn "$D/dtk-shim/amdgcn"
ln -sfn "$D/dtk-shim/aillvm" /opt/dtk-26.04/aillvm
echo "  aillvm: $D/dtk-shim/aillvm/bin/clang-18"

echo "########## [6/6] 验证环境 ##########"
source /opt/dtk-26.04/env.sh 2>/dev/null
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TRITON_HIP_CLANG_PATH="$D/dtk-shim/aillvm/bin/clang-18"
export AMDGCN_USE_BUFFER_OPS=0
$PY "$WS/scripts/deploy/slurm/das185_check.py" 2>&1 | grep -vE "^WARNING|rocm_smi|UserWarning|_ARRAY_API|cpu = |functional_tensor|NUMA balancing|torch-c-dlpack|warnings.warn"

echo "########## 适配完成 ##########"

if [ "$DOWNLOAD_MODEL" = "1" ]; then
  echo "########## 开始下载模型（0731 + Flash-Base → $MODELS）##########"
  bash "$WS/dl_flash_0731_base.sh"
else
  echo "（--no-model：跳过模型下载）"
fi

echo "ALL DONE"
