# 部署指南（DEPLOYMENT）

> 从零到把 DeepSeek-V4 跑起来的分步手册。四个模型、两种集群环境、两种部署形态
> （Slurm 多机 / 单机冒烟）全覆盖。命令统一走 `deploy.sh`，也可直接调 `scripts/` 下的
> 底层脚本。

---

## 〇、先搞清楚你在哪个集群

本框架实测过**两套集群**，网卡、卡数、显存都不同，命令不能混用：

| 环境 | 分区 | DCU | 每节点卡数 | 网卡 | 适用模型 |
|---|---|---|---|---|---|
| **A. 老集群** | `kshdnormal` | gfx906（16G） | 4 | Mellanox mlx5 200G（钉 `mlx5_0`） | Flash-0731 / Flash-Base / Pro-0813(20n80c) |
| **B. 新集群** | `hx1hdnormal01` | gfx936 海光 BW（64G） | 8 | 自研 shca 400G（`librccl-net-shca.so`） | Pro-0813(4n32c) |

> 判断方法见 `SUPPORTED_HARDWARE.md` 第六节的探针命令。**先确认再动手**，尤其是
> `NCCL_IB_HCA=mlx5_0` vs `NCCL_NET_PLUGIN=librccl-net-shca.so` 这一条，选错直接退化/失败。

---

## 一、部署前准备（一次性）

```bash
# 1. 克隆仓库
git clone <your-repo-url> && cd dcu-vllm-deploy

# 2. 设置路径（不设就用默认 $HOME/models 等）
export MODELS=/path/to/weights      # 模型权重盘（要够大，Pro-Base 需 1.46TB）
export DAS185=/path/to/das185       # das185 环境目录
export WORK=/path/to/work           # 大容量工作盘（aillvm 解包、miniconda）

# 3. 装环境 + 下 wheel + 打补丁（幂等，可重跑）
bash deploy.sh install --no-model

# 4. 探针确认 torch 认卡 + 各模块 import
bash deploy.sh install --dry-run 2>/dev/null || true
# 实际探针：source /opt/dtk-26.04/env.sh 后跑 scripts/install/das185_check.py
```

> `install` 会做的事：装 python3.10 头文件 → 下载 das185 全家桶 wheel → 装 wheel →
> 打 4 个补丁（`das185_patch.py`）→ vllm_hcu post_install → 解包 aillvm → 验证环境。
> 全部幂等，断了重跑即可。

---

## 二、下载模型权重

| 模型 | 命令 | 体积 | 备注 |
|---|---|---|---|
| Flash-0731 | `bash deploy.sh download --model flash0731` | ~156G | 同时下 Flash-Base（脚本打包两个） |
| Flash-Base | 同上（或 `--model flashbase`） | ~275G | |
| Pro-0813 | `bash deploy.sh download --model pro0813 --script scripts/download/dl_v4pro_base.sh`（改 model id） | ~832G | 见下注 |
| Pro-Base | `bash deploy.sh download --model probase` | ~1.46TB | 需本地大盘；`PAR=8` 可加速 |

> **注意**：`dl_v4pro_base.sh` 原生下载的是 Pro-Base。下载 Pro-0813 需把脚本里的
> `M=DeepSeek-V4-Pro-Base` 改成 `M=DeepSeek-V4-Pro-0813`，或直接用 `--script` 指向
> 你自己改好的副本。hf-mirror 限流时换 ModelScope 版 `dl_ms_flash_0731_base.sh`。

---

## 三、部署（Slurm 多机多卡）

统一入口，`--dry-run` 先预览：

```bash
bash deploy.sh serve --model flash0731 --dry-run
```

### Flash-0731（老集群 A，4 节点 × 4 卡 = 16 卡）

```bash
bash deploy.sh serve --model flash0731
# 默认 TP=4 PP=4 DP=1 EP=4，MAXLEN=8192
# 等 /health 就绪后自动跑 /v1/completions 冒烟
```

- 显存账：153GB 权重 / 16 卡 ≈ 9.6GB/卡，16G 卡 × util 0.85 够用。
- **已知卡点**：一进 MoE 前向就崩（fp4 dtype 不支持），见 `docs/KNOWN_ISSUES.md` §1。

### Flash-Base（老集群 A，8 节点 × 4 卡 = 32 卡）

```bash
bash deploy.sh serve --model flashbase
# 默认 TP=4 PP=8 DP=1 EP=4，MAXLEN=4096（坑26 已降到 4096）
```

- **已知卡点**：FP8 sorting kernel VM Fault，同见 KNOWN_ISSUES §1。
- Base 不是 instruct，冒烟 prompt 用续写式（"人工智能是"），别问指令。

### Pro-0813（两种选一种）

**B 集群 4n32c（推荐，最接近跑通）**：

```bash
bash deploy.sh serve --model pro0813
# 默认 4 节点 × 8 卡，TP=8 PP=2 DP=2 EP=16，走 shca RDMA 四平面
```

**A 集群 20n80c（老卡 16G 拆 20 节点）**：

```bash
bash deploy.sh serve --model pro0813 \
  --script scripts/deploy/slurm/v4pro_20n80c_serve.sh
# TP=4 PP=20 DP=1，832G / 80 卡 ≈ 10.4GB/卡
```

- **已知卡点**：差 `VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0` 未复跑（脚本已带）。
  这是四个模型里**最接近跑通的**，建议从这里接手复跑。

### Pro-Base（1.46TB，尚未部署）

```bash
bash deploy.sh serve --model probase    # 会报错：无 serve 脚本
```

这是**故意的**：Pro-Base 目前只有下载脚本，从未部署验证。要部署需先按 Pro-0813 的
4n32c 脚本改 `--script`，并把并行参数按 1.46TB 重算（参考 README 节点数表：gfx936 需
3 节点起，gfx928 需 2 节点）。

---

## 四、冒烟（单机 8 卡，无 Slurm）

新集群 B 的 gfx936 单机（如 notebook 实例）上，先冒烟再上多机：

```bash
bash deploy.sh smoke --model flash0731     # TP=8 真实权重，卡 MoE
bash deploy.sh smoke --model flashbase     # TP=8，卡 MoE
```

底层还有两个探针/冒烟（不走 deploy.sh）：

```bash
python3 scripts/deploy/smoke/gfx936_smoke_probe.py   # aiter shuffle + triton 编译 + fp8 dot
python3 scripts/deploy/smoke/gfx936_dummy_smoke.py   # dummy 权重全链路（能通的那条链）
```

> **dummy 冒烟能通、真实权重卡住**，这个反差本身就是结论：链路/环境没问题，
> 卡在真实 256 experts 规模触发的 MoE 算子缺口。

---

## 五、排障速查

按"基线 → 定位 → 根因 → 修复 → 验证"五步，坑位都记在 `docs/GOTCHAS.md`：

| 症状 | 查哪条 | 一句话修复 |
|---|---|---|
| ZMQ `Cannot assign requested address` | 坑1 | 走 fabric 地址，管理网一律不用 |
| `HSACOError: File operation failed` | 坑6 | `TMPDIR=/tmp` |
| triton 编译失败 | 坑7/8/21 | `TRITON_HIP_CLANG_PATH=aillvm/clang-18` + `AMDGCN_USE_BUFFER_OPS=0` + `CC=hipcc` |
| `expected BF16 but found Float8` | 坑9 | `VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0` |
| `Duplicate op name` | 坑12/14 | `PYTHONPATH=$WS/patches` |
| `amdsmi` NameError | 坑13 | `--disable-custom-all-reduce` |
| KV cache 断言 / OOM | 坑25/26 | `--block-size 256` + `--gpu-memory-utilization 0.95` |
| worker 挂起不生成 | 坑27 | `VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK=1` |
| RCCL 只通单口 | 坑10 | `NCCL_IB_HCA=mlx5_0`（shca 则 `NCCL_NET_PLUGIN`） |

---

## 六、部署矩阵总表

| 模型 | 集群 | 节点×卡 | TP/PP/DP/EP | 状态 |
|---|---|---|---|---|
| Flash-0731 | A | 4×4 | 4/4/1/4 | 卡 MoE |
| Flash-Base | A | 8×4 | 4/8/1/4 | 卡 MoE |
| Pro-0813 | B | 4×8 | 8/2/2/16 | 差 1 开关未复跑 |
| Pro-0813 | A | 20×4 | 4/20/1/4 | 同上 |
| Pro-Base | — | 未部署 | — | 仅下载 |

> 完整并行策略说明、节点数估算、显存预留见 `README.md` 三、四节和
> `SUPPORTED_HARDWARE.md`。
