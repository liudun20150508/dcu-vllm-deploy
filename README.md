# dcu-vllm-deploy

在海光 DCU（Hygon DCU，HIP/ROCm 生态）上用 vLLM 部署 DeepSeek-V4 系列模型的
**部署框架 + 踩坑记录 + 条件矩阵**。

> ⚠️ 本项目只开源**代码、脚本、补丁、文档**，不包含任何模型权重、厂商专有
> wheel（das185）、或专有 RDMA 插件。这些一律通过脚本从官方源拉取，或向集群
> 管理员索取。详见 [支持与限制](#支持与限制)。

## 文档导航

| 文档 | 看什么 |
|---|---|
| [`README.md`](README.md)（本文件） | 4 模型状态、前置条件、节点数、快速开始、缺口 |
| [`SUPPORTED_HARDWARE.md`](SUPPORTED_HARDWARE.md) | DCU 架构 × 模型 条件矩阵、自检清单 |
| [`docs/PREREQUISITES.md`](docs/PREREQUISITES.md) | 前置条件满足度清单（✅/❌/⚠️ 逐条，不满足的后果） |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | 真机实测数据（66倍调优、带宽、P2P、KV Cache、RDMA） |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | 从零部署的分步手册（4 模型 × 两集群） |
| [`docs/GOTCHAS.md`](docs/GOTCHAS.md) | 30+ 条踩坑记录（按症状查） |
| [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) | 缺口清单（还差什么、谁来解决） |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 如何适配新硬件/新模型、贡献约定 |

**第一次来？** 先跑 `./deploy.sh list` 看模型状态，再读 `docs/DEPLOYMENT.md` 的
「部署前准备」。要接手适配，直接看 `CONTRIBUTING.md` 第五节「最希望被解决的缺口」。

---

## 一、支持哪些模型

| 模型 | 精度 | 权重体积 | 分片数 | 部署状态 |
|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 | FP4 专家 | ~156 GB | 48 | ⚠️ 环境/权重/通信通，**推理卡 MoE 算子**（见缺口） |
| DeepSeek-V4-Flash-Base | FP8 专家 | ~275 GB | 46 | ⚠️ 同上，卡 MoE 算子 |
| DeepSeek-V4-Pro-0813（"0831pro"） | MXFP4/FP8 | ~832 GB | 66 | ⚠️ 8 次部署排障至最后一步，**差一个环境开关未复跑** |
| DeepSeek-V4-Pro-Base | FP8 | ~1.46 TB | 64 | ❌ 仅提供下载脚本，**从未部署验证** |

> 状态说明是**如实记录**的：前三个模型的"环境 + 权重加载 + RCCL 通信"三层都已
> 跑通，瓶颈集中在底层 MoE 算子，不是配置问题。Pro-Base 是体积最大的一个，
> 目前只到"下载脚本"这一步，适配者可以从这里接手。

## 二、前置条件（能不能跑，先看这张表）

### 硬件

| DCU 架构 | 型号 | 单卡显存 | 每节点卡数 | 备注 |
|---|---|---|---|---|
| gfx906 | 深算一号 Z100L | 16 GB | **4（实测）** | 老架构，部分算子不支持 |
| gfx938 | 海光 BW（C-3000，BW-1 队列） | 64 GB | 8 | 本框架实测目标 |
| gfx928 | 深算二号 K100 | 96 GB（**需真机核实**） | 8 | 未实测，仅按公开规格推算 |

**关键结论：不同 gfx 架构支持的 MoE 算子不同。** 这是本框架踩坑的核心：
fp4/fp8 的 DeepSeek-V4 MoE 后端在 gfx950 系列才完整，gfx938 缺 kernel，
gfx906 缺得更多。适配前**先跑 `scripts/install/das185_check.py` 探针确认架构**。

### 软件栈（版本锁定，不要动）

| 组件 | 版本 | 来源 |
|---|---|---|
| DTK | 26.04 | 集群 module |
| torch | 2.10.0+das.opt1.dtk2604 | 光合开发者社区 |
| vllm / vllm_hcu | 0.21.0+das.dtk2604.torch2100 | 光合开发者社区 |
| aiter | 0.1.5+das185 | 光合开发者社区 |
| triton | 3.6.0+das185 | 光合开发者社区 |
| tilelang | 0.1.9+das185 | 光合开发者社区 |
| flash_mla / lightop / deepgemm | 1.2.0 / 0.6.0 / 2.1.0 | 光合开发者社区 |
| aillvm (clang-18) | dtk_llvm.run 解包 | 光合开发者社区 |

> 这些版本是逐坑踩出来的（见 `docs/GOTCHAS.md`），随意升级会引入新问题。

### 网卡 / RCCL（两套实测集群，别混）

| 集群 | 分区 | DCU | 集群总规模 | 每节点卡数 | 每节点 RDMA | RCCL 怎么走 |
|---|---|---|---|---|---|---|
| 老集群 | `kshdnormal` | gfx906 | **494 节点 / 1976 卡** | 4 卡 | **Mellanox mlx5 200G ×4** | 内置 IB verbs，实测仅 `mlx5_0` 能建 QP → `NCCL_IB_HCA=mlx5_0` |
| 新集群 | `hx1hdnormal01` | gfx938 | **95 节点 / 760 卡** | 8 卡 | **自研 shca 400G ×4** | `NCCL_NET_PLUGIN=librccl-net-shca.so`（专有插件，向集群管理员索取，不入库） |

> 单节点 scha RDMA 设备数 = **4 张**（`shca_0..3`，link_layer InfiniBand，PORT_ACTIVE）；
> 单节点 mlx5 设备数 = **4 张**（`mlx5_0..3`，200G IB）。选错网卡配置会直接退化/失败。

## 三、节点数估算（纯权重下限，按各架构实测卡数）

| 模型 | gfx906 (16G×4=64G/节点) | gfx928 (96G×8=768G*/节点) | gfx938 (64G×8=512G/节点) |
|---|---|---|---|
| Flash-0731 (156G) | 3 | 1 | 1 |
| Flash-Base (275G) | 5 | 1 | 1 |
| Pro-0813 (832G) | 13 | 2 | 2 |
| Pro-Base (1.46T) | 24 | 2 | 3 |

\* gfx928 显存按 96G 推算，需真机核实；906 按实测 4 卡/节点、928/938 按 8 卡/节点。
真实推理还要加 KV Cache + 激活 + workspace，**按 1.2~1.5× 预留**；
长上下文场景（1M token）KV Cache 会额外吃掉几十 GB，务必按 `--max-model-len` 反推后放宽。

## 四、快速开始

```bash
# 1. 克隆（本仓库不含权重/wheel/插件）
git clone <your-repo-url> && cd dcu-vllm-deploy

# 2. 环境 + 下载 das185 wheel + 打补丁（幂等，可重跑）
bash deploy.sh install --no-model

# 3. 下载模型权重（走 hf-mirror / ModelScope）
bash deploy.sh download --model flash0731   # Flash-0731 + Flash-Base
bash deploy.sh download --model probase     # Pro-Base（1.46TB，需本地大盘）

# 4. 部署
bash deploy.sh list                          # 先看 4 个模型及状态
bash deploy.sh serve --model flash0731       # Slurm 多机多卡（默认 TP4 PP4；--gres 按集群卡数）
bash deploy.sh serve --model pro0813         # 4n×8卡 TP8 PP2 DP2 EP16
bash deploy.sh serve --model flash0731 --nodes 2 --tp 8 --dry-run  # 先 dry-run 预览
bash deploy.sh smoke --model flashbase       # 单机 8 卡冒烟（无 Slurm）
```

### CLI 说明

统一入口 `deploy.sh`，五个子命令：

| 命令 | 作用 | 示例 |
|---|---|---|
| `list` | 列出 4 模型 + 状态 | `./deploy.sh list` |
| `serve` | Slurm 多机多卡部署 | `./deploy.sh serve --model flash0731 --nodes 2 --tp 8` |
| `smoke` | 单机 8 卡冒烟 | `./deploy.sh smoke --model flashbase` |
| `download` | 下载权重 | `./deploy.sh download --model probase` |
| `install` | 环境 + 补丁 | `./deploy.sh install --no-model` |

通用选项：`--dry-run`（只打印不执行）、`--script <path>`（跳过模型默认脚本的逃生舱）。
`serve` 可覆盖默认并行策略：`--nodes --gres --tp --pp --dp --maxlen --partition`。

**路径配置**：脚本里所有路径走环境变量（`WS`/`MODELS`/`DAS185`/`WORK`），
默认 `WS=仓库根`、`MODELS=$HOME/models`、`DAS185=$HOME/das185`、`WORK=$HOME/work`，
用 `export MODELS=/data/models` 之类覆盖即可。

### 安装 / 验证命令速查

**一键安装**（`deploy.sh install` 背后是 `scripts/install/setup_all.sh`，幂等可重跑）：

```bash
bash deploy.sh install --no-model      # 只装环境，不下载模型
bash deploy.sh install                 # 装环境 + 下载 0731/Base 权重

# 等价底层命令（setup_all.sh 的 6 步）：
source scripts/install/env.sh                          # ① 加载 DTK/HIP 环境（每次运行前都要）
bash scripts/install/das185_dl_wheels.sh               # ② 下载 das185 全家桶 wheel + dtk_llvm.run
bash scripts/install/setup_all.sh --no-model           # ③~⑥ 装 wheel→依赖→补丁→aillvm→验证
```

**独立命令**：

```bash
source scripts/install/env.sh                          # 运行环境（DTK 26.04 + HIP 库路径）
bash scripts/install/build_glibc_shim.sh               # 老集群 glibc 2.31 才需要：编 libglibc232shim.so
python3 scripts/install/das185_check.py                # 探针：torch 认卡 + 各模块 import
```

**部署前验证**（对应 `docs/PREREQUISITES.md` 的 ✅ 条件）：

```bash
bash scripts/verify/check_hw.sh                        # 硬件：架构/显存/CU/卡数/网卡/DTK
bash scripts/verify/rccl_allreduce.sh --np 8           # RCCL 通信正确性（单机 8 卡）
sbatch scripts/verify/rccl_allreduce.sh                # RCCL 跨机（Slurm）
```

## 五、目录结构

```
dcu-vllm-deploy/
├── deploy.sh                        # 统一 CLI 入口（list/serve/smoke/download/install）
├── README.md                        # 项目总览 + 文档导航
├── SUPPORTED_HARDWARE.md            # DCU 架构 × 模型 条件矩阵（详细）
├── CONTRIBUTING.md                  # 适配新硬件/新模型贡献指南
├── LICENSE                          # MIT
├── docs/
│   ├── PREREQUISITES.md              # 前置条件满足度清单（不满足的条件写在这）
│   ├── BENCHMARKS.md                 # 真机实测数据（带宽/P2P/KV Cache/RDMA）
│   ├── DEPLOYMENT.md                # 从零部署分步手册（4 模型 × 两集群）
│   ├── GOTCHAS.md                   # 30+ 条踩坑记录（脱敏）
│   └── KNOWN_ISSUES.md              # 缺口清单（还差什么）
├── scripts/
│   ├── download/                    # 模型权重下载（hf-mirror / ModelScope）
│   ├── install/                     # 环境安装 + 补丁 + 探针 + glibc-shim 编译
│   ├── verify/                      # 前置条件验证（check_hw + rccl_allreduce）
│   └── deploy/
│       ├── slurm/                   # Slurm 多机多卡 serve
│       └── smoke/                   # 单机 8 卡冒烟
├── experiments/                     # 算子层 HIP kernel（FP4 反量化等，绕开 aiter 的起点）
└── patches/                         # 运行期注入补丁（sitecustomize + glibc-shim 源码）
```

## 六、已知缺口（还差什么）

| 缺口 | 谁来解决 | 状态 |
|---|---|---|
| gfx938 版 DeepSeek-V4 MoE 算子（fp4 dtype + fp8 sorting bug） | 海光/曙光厂商（换新 aiter 或要补丁） | 未解决 |
| `libgcvm.so.17git`（海光 triton 依赖） | 海光驱动包 | 未解决 |
| Pro-0813 fast-woa dtype（`VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0`） | 适配者复跑验证 | 修复已写、**未复跑** |
| gfx928 真机规格（显存/算子支持） | 适配者 | 未核实 |
| 从 HIP 手写 fp4 W4A16 MoE GEMM 绕开 aiter | 可选，工作量大 | 只有 dequant 第一步 |

完整清单见 `docs/KNOWN_ISSUES.md`。

## 七、逐步适配指南（给接手的人）

1. **先跑探针**：`scripts/install/das185_check.py` 确认 torch 认卡 + 各模块 import；
   `scripts/deploy/smoke/` 里带 `dummy` 的脚本用假权重走通全链路（不含真实 MoE）。
2. **从"最接近跑通"的模型切入**：Pro-0813 已排掉 8 个坑、只差一个开关，
   建议先复跑它，再回头啃 Flash-0731/Base 的 MoE 算子。
3. **换架构先改两处**：`das185_patch.py` 里的 gfx 白名单、`run_*_node.sh` 里的
   `NCCL_IB_HCA` / `NCCL_NET_PLUGIN`（shca vs mlx5）。
4. **遇到新坑先看 `docs/GOTCHAS.md`**，按"基线 → 定位 → 根因 → 修复 → 验证"
   五步记录，再回来补文档。

## 八、支持与限制

- **可自由使用**：本仓库所有脚本/补丁/文档（见 LICENSE）。
- **不包含**：模型权重（DeepSeek，从 hf-mirror/ModelScope 拉）、das185 wheel
  （海光专有，从光合社区拉）、`librccl-net-shca.so`（曙光专有，向管理员索取）。
- **集群相关信息已脱敏**：IP、账号、绝对路径一律用变量/占位符，适配者替换成
  自己环境的路径。
