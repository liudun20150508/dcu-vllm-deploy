#!/bin/bash
#SBATCH -J acc-j33-v2lite
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:2
#SBATCH -t 00:40:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j33-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j33-%j.err

# J33（优先 5/6/7）：用 vLLM 在海光 DCU 上跑真模型 DeepSeek-V2-Lite（MoE + MLA）
#   这一步是 PD 分离的地基 —— 先确认真模型能起、能生成、KV Cache 实际占多少
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
PY=/usr/local/python3.12/bin/python3
echo "########## J33 on $(hostname) ($(date)) ##########"
echo "模型: $(du -sh $HOME/models/DeepSeek-V2-Lite 2>/dev/null | cut -f1)"

# 文本推理用不到音频栈：把 librosa 卸掉，transformers 的 is_librosa_available() 就返回 False，
# 不再 import 它（比一个个补 soxr/lazy_loader… 快得多）
# 注意：必须在装完 deps 之后卸 —— deps 里也有一份 librosa，会把它装回来
echo "[dep] 逐个装本地 deps（共 $(ls $HOME/hcu/wheels/deps/*.whl | wc -l) 个）"
ok=0; bad=0
for w in $HOME/hcu/wheels/deps/*.whl; do
  if $PY -m pip install --user --no-index --no-deps -q "$w" >/dev/null 2>&1; then
    ok=$((ok+1))
  else
    bad=$((bad+1)); echo "  [skip] $(basename "$w")"
  fi
done
echo "[dep] 装上 $ok 个，跳过 $bad 个"

# ★ llvmlite 必须和 numba 配套，这是一次真实的缺件事故：
#     vLLM 的 v1/worker/gpu_model_runner.py 顶层就 `from numba import ...`（spec decode 的
#     ngram_proposer），而 numba 0.67.0 在 __init__ 里做 _ensure_llvm() 校验
#     —— 要求 llvmlite >=0.49,<0.50，机器上系统 site-packages 里那份是 DTK 预装的 0.43.0，
#     于是 WorkerProc 还没初始化 GPU 就在 import 阶段抛 ImportError，
#     表象是 "WorkerProc initialization failed"，根因却藏在子进程 stderr 里。
#     deps 目录原先没有 llvmlite wheel（全是 --no-deps 装，缺件不会被自动补），
#     这里补装 llvmlite 0.49.0（--user 会盖住系统的 0.43.0）。
LLVM=$(ls $HOME/hcu/wheels/deps/llvmlite-*.whl 2>/dev/null | head -1)
if [ -n "$LLVM" ]; then
  $PY -m pip install --user --no-index --no-deps --force-reinstall "$LLVM" 2>&1 | tail -1
  $PY -c "
import numba, llvmlite
print(f'  numba={numba.__version__}  llvmlite={llvmlite.__version__}')" 2>&1 | tail -1
else
  echo "  [warn] 没找到 llvmlite wheel（numba 会在 import 时失败）"
fi

# ★ numpy ABI pin —— j33 的第三个坑，也是最深的一个：
#     海光适配版 torch 2.5.1 是**用 NumPy 1.x 编译**的，而系统预装的是 numpy 2.4.0：
#       torch 内部 import numpy 失败 → 打出 "Failed to initialize NumPy"
#       → 之后 torch.distributed 的 `tensor.numpy()` 抛 RuntimeError: Numpy is not available
#       → vLLM 的 shm_broadcast 第一次调用就崩，表象依旧是 "WorkerProc initialization failed"。
#     修法：numpy 锁 1.26.4。（NumPy2 编译的包能向下兼容 1.x，反向不行 —— 所以只能降 numpy，
#     不能指望升级 torch：AMD 官方 +rocm6.2 那版在这台机上 is_available() 静默为 False。）
#     连带项：deps 原先给的 scipy 1.18.1 声明要 numpy>=2.0，一起降到 1.17.1
#     （要求 numpy>=1.26.4,<2.7，与 1.26.4 自洽）；原 1.18.1 已移到 hcu/wheels/deps-parked/。
#     注意：系统 numpy 2.4.0 在 /usr/local 下**无写权限**，必须 --ignore-installed 直接铺到 user site，
#     否则 pip 会先尝试卸载系统那份、报权限错，结果 numpy 还是 2.4.0。
NPW=$(ls $HOME/hcu/wheels/deps/numpy-1.26.4-*.whl 2>/dev/null | head -1)
SCW=$(ls $HOME/hcu/wheels/deps/scipy-1.17.1-*.whl 2>/dev/null | head -1)
if [ -n "$NPW" ]; then
  $PY -m pip install --user --no-index --no-deps --ignore-installed "$NPW" 2>&1 | tail -1
else
  echo "  [warn] 没找到 numpy 1.26.4 wheel（torch 的 Tensor.numpy() 会不可用）"
fi
if [ -n "$SCW" ]; then
  $PY -m pip install --user --no-index --no-deps --force-reinstall "$SCW" 2>&1 | tail -1
fi
$PY -c "
import numpy, scipy, torch
print(f'  numpy={numpy.__version__}  scipy={scipy.__version__}  torch={torch.__version__}')
print('  torch.zeros(3).numpy() ->', torch.zeros(3).numpy())" 2>&1 | tail -3
# ★ amdsmi 这一处是本次最拧巴的地方，两个需求互相矛盾：
#     · vLLM 的 rocm_platform_plugin() 靠 `import amdsmi` + amdsmi_get_processor_handles()
#       来判定"这是 ROCm/DCU 平台" —— 没有它 vLLM 就报 No platform detected
#     · 而 PyPI 上那份 amdsmi 在 import 时就去绑 libamd_smi.so 的符号，
#       海光 HAL 里没有 amdsmi_get_gpu_enumeration_info → 连 torch 都 import 不了
#   解法：卸掉真 amdsmi，给一个最小替身 —— 报出正确的卡数（两边都够用），
#        未实现的接口一律返回 None，不去碰任何缺失符号。
$PY -m pip uninstall -y amdsmi 2>&1 | tail -1
cat > $HOME/.local/lib/python3.12/site-packages/amdsmi.py <<'PYEOF'
"""amdsmi 的最小替身（本集群的海光 HAL 与 PyPI 版 amdsmi 的符号不匹配）。

vLLM 需要它来识别 ROCm/DCU 平台；torch 在它可用时会调用部分查询接口。
这里只实现"数卡"这一个真实语义，其余一律返回 None。
"""
import sys


class AmdSmiException(Exception):
    def __init__(self, err_code=0, msg=""):
        self.err_code = err_code
        super().__init__(msg)


def amdsmi_init(*a, **k):
    return None


def amdsmi_shut_down(*a, **k):
    return None


def amdsmi_get_processor_handles():
    """关键的一个：返回非空列表，vLLM 才认定这是 DCU 平台。

    这里直接数 /dev/dri/renderD*（不能调 torch.cuda.device_count()——
    这个函数在 torch 初始化之前就会被调用，那时它返回 0）。
    数设备节点也正好符合 Slurm cgroup 的限制：分到几张就是几张。
    """
    import glob
    n = len(glob.glob("/dev/dri/renderD*"))
    return list(range(n))


def amdsmi_topo_get_link_type(*a, **k):
    """查询两卡之间的链路拓扑（XGMI/PCIe）。

    海光 HAL 没有这个查询，落到 __getattr__ 返回 None 会让 vLLM 的
    is_fully_connected() 里 link_type["hops"] 直接 TypeError（WorkerProc 起不来）。
    抛 AmdSmiException → is_fully_connected 返回 False → vLLM 退回用 NCCL/RCCL 建 TP 组。
    """
    raise AmdSmiException(0, "topology link type query not supported on this platform")


def __getattr__(name):
    # 其余未实现的接口：给一个什么都不做的函数，避免 AttributeError
    def _stub(*a, **k):
        return None
    return _stub
PYEOF
echo "[amdsmi] 已装最小替身"
$PY -c "import amdsmi, torch; print('  amdsmi stub ok, 卡数=', len(amdsmi.amdsmi_get_processor_handles()), ' torch=', torch.__version__)" 2>&1 | tail -2
# ★ deps 里那份 huggingface-hub 是 2.0.0，会把之前降好的 1.33.0 顶掉，
#   而 transformers 5.17 硬要求 <2.0 —— 装完再强制装回 1.33.0
$PY -m pip install --user --no-index --no-deps --force-reinstall \
  $HOME/hcu/wheels/deps/huggingface_hub-1.33.0-*.whl 2>&1 | tail -1
# ★ 最后再卸一次 librosa：deps 里也有一份，刚才那轮把它装回来了，
#   而 transformers 只有在"librosa 不可用"时才不 import 音频栈
$PY -m pip uninstall -y librosa 2>&1 | tail -1
# ★ 海光版 vLLM 里 import 了 `numa`（PyPI 上没有这个包）。
#   看它的用法：只有设了 VLLM_NUMA_NODE 才真调 numa.bind()，否则只调 get_max_node()。
#   所以给一个最小实现即可（不是伪造行为，是真绑核这件事在 Slurm 里本来就该由 srun --cpu-bind 做）
STUB=$HOME/.local/lib/python3.12/site-packages/numa.py
if [ ! -f "$STUB" ]; then
  cat > "$STUB" <<'PYEOF'
"""numa 的最小替身 —— 海光版 vLLM 依赖此模块（PyPI 无发行版）。
只有 VLLM_NUMA_NODE 被设置时才会走到 bind()；此时这里会明确提示未生效。"""


def get_max_node():
    return 0


def get_membind():
    return (0, 0, 0)


def bind(nodes):
    import sys
    print(f"[numa-stub] 收到绑核请求 {nodes}，但本环境未安装真实 numa 模块（未生效）",
          file=sys.stderr)
PYEOF
  echo "[numa] 已生成 stub: $STUB"
else
  echo "[numa] stub 已存在"
fi

# ★ lmslim：海光私有的量化算子包 —— vLLM 的 fp8_utils.py 里硬依赖它
#   （PyPI 上没有，只能从光合社区取 dtk2604.torch251.cp312 对应的那一版）
LMSLIM=$(ls $HOME/hcu/wheels/lmslim-*.whl 2>/dev/null | head -1)
if [ -n "$LMSLIM" ]; then
  $PY -m pip install --user --no-index --no-deps --force-reinstall "$LMSLIM" 2>&1 | tail -1
  $PY -c "import lmslim; print('  lmslim ok')" 2>&1 | tail -1
else
  echo "  [warn] 没找到 lmslim wheel（vLLM 的 fp8_utils 会 import 失败）"
fi
# ★ lightop：又一层的海光私有算子包（lmslim 依赖它）
LIGHTOP=$(ls $HOME/hcu/wheels/lightop-*.whl 2>/dev/null | head -1)
if [ -n "$LIGHTOP" ]; then
  $PY -m pip install --user --no-index --no-deps --force-reinstall "$LIGHTOP" 2>&1 | tail -1
  $PY -c "import lightop; print('  lightop ok')" 2>&1 | tail -1
else
  echo "  [warn] 没找到 lightop wheel"
fi
# ★ flash_attn：海光适配版（vLLM 的 attention 后端要用）
FA=$(ls $HOME/hcu/wheels/flash_attn-*.whl 2>/dev/null | head -1)
if [ -n "$FA" ]; then
  $PY -m pip install --user --no-index --no-deps --force-reinstall "$FA" 2>&1 | tail -1
  $PY -c "import flash_attn; print('  flash_attn ok')" 2>&1 | tail -1
else
  echo "  [warn] 没找到 flash_attn wheel"
fi
# ★ 版本对齐：vLLM 0.11.0 声明的是 transformers==4.57.6，
#   而 deps 里那份 5.17.0 的 tokenizer API 已经变了
#   （报错：TokenizersBackend has no attribute all_special_tokens_extended）。
#   4.57.6 又要求 huggingface-hub<1.0 和 tokenizers<=0.23.0，所以这两个也一起钉住。
$PY -m pip install --user --no-index --no-deps --force-reinstall \
  $HOME/hcu/wheels/deps/transformers-4.57.6-*.whl \
  $HOME/hcu/wheels/deps/huggingface_hub-0.36.2-*.whl \
  $HOME/hcu/wheels/deps/tokenizers-0.22.2-*.whl 2>&1 | tail -1
$PY -c "
import transformers, huggingface_hub, tokenizers, vllm
print(f'  transformers={transformers.__version__}  hub={huggingface_hub.__version__}  tokenizers={tokenizers.__version__}  vllm={vllm.__version__}')" 2>&1 | tail -1

cd $WS/experiments
export VLLM_LOGGING_LEVEL=DEBUG
# ★ 完整日志必须落盘：WorkerProc 的根因 traceback 会打在子进程 stderr 里，
#   一旦用 `| tail -N` 收尾就会被截掉（上一次就是这么丢的根因）。
FULL=$WS/验收/logs/j33-full-${SLURM_JOB_ID}.log
VLLM_MODEL=$HOME/models/DeepSeek-V2-Lite VLLM_TP=2 \
  $PY v2lite_infer.py > "$FULL" 2>&1
rc=$?
echo "[v2lite_infer.py] exit=$rc  完整日志：$FULL（$(wc -l < "$FULL") 行）"
echo "----- 尾部 40 行 -----"
grep -a -v -E "ProcessGroupNCCL|watchdog|heart beat|^\s*$" "$FULL" | tail -40
echo "----- 全部 Traceback / Error 行 -----"
grep -a -n -E "Traceback|Error|error:|Exception|RuntimeError|AssertionError" "$FULL" | head -40
echo "########## J33 done $(date) ##########"
