# 贡献指南（CONTRIBUTING）

本仓库的目标是让「DeepSeek-V4 在国产 DCU 上的部署」这件事**可复现、可接力**。
核心资产不是代码量，而是**如实记录的踩坑和条件矩阵**——所以欢迎的贡献主要有两类：
**适配新硬件/新模型**，和**补全/纠错文档**。

---

## 一、先读这些（别重复造轮子）

| 文档 | 内容 |
|---|---|
| `README.md` | 4 模型状态、前置条件、节点数、快速开始 |
| `SUPPORTED_HARDWARE.md` | DCU 架构 × 模型 条件矩阵、自检清单 |
| `docs/DEPLOYMENT.md` | 从零部署的分步手册 |
| `docs/GOTCHAS.md` | 30+ 条踩坑（按症状查） |
| `docs/KNOWN_ISSUES.md` | 当前缺口（还差什么、谁来解决） |

---

## 二、两类主要贡献

### A. 适配新硬件（新 gfx 架构 / 新网卡）

**判断你是不是真适配**：`SUPPORTED_HARDWARE.md` 里那张「MoE 后端 × gfx 架构」矩阵，
你的架构对应的一行还是空的，才有适配空间。

**步骤**：

1. **跑探针**，拿到真实规格（架构代号、CU 数、显存、warpSize、网卡）：
   ```bash
   rocminfo | grep -iE "gfx|Compute Unit"
   cat /sys/class/drm/card*/device/mem_info_vram_total
   ibstat | grep -iE "^CA|Rate|Link layer"
   ```
2. **先冒烟**：`scripts/deploy/smoke/gfx936_dummy_smoke.py` 用 dummy 权重走通链路
   （不含真实 MoE），确认「环境/结构加载/前向」没问题，再上真实权重。
3. **记录真实权重结果**，分三层判断：
   - 环境/权重/RCCL 通 ✅ → 卡 MoE 算子 → 记录到 `KNOWN_ISSUES.md`，附后端报错原文；
   - 权重都加载不了 → 查显存/节点数（`README.md` 三节）；
   - RCCL 不通 → 查网卡插件（`GOTCHAS.md` 坑10）。
4. **把结论写进矩阵**：更新 `SUPPORTED_HARDWARE.md` 的架构速查表 + MoE 后端表，
   标注「实测 / 推算」。

**提交内容**：探针输出、适配脚本（若新增）、`SUPPORTED_HARDWARE.md` 改动、
`KNOWN_ISSUES.md` 或 `GOTCHAS.md` 的对应条目。**真机证据优先于推理。**

### B. 适配新模型（DeepSeek-V4 之外的 MoE 模型）

1. 在 `scripts/download/` 加一个下载脚本（复用 `dl_v4pro_base.sh` 的断点续传+尺寸校验骨架，
   只改 `M=<model-id>` 和分片逻辑）。
2. 在 `scripts/deploy/slurm/` 加 serve 脚本（复制 `flash0731_serve.sh`，
   改 `MODEL_DIR` / `SERVED_NAME` / 默认 TP/PP/DP/EP）。
3. 在 `deploy.sh` 的注册表加一个 case 分支（`model_dir`/`model_weight`/`model_serve`/
   `model_smoke`/`model_defaults` 五个函数）。
4. 更新 `README.md` 模型表 + 节点数表。

---

## 三、文档与脚本的硬性约定

1. **脱敏**：任何提交**不得包含**真实 IP、账号名、`/public/home/...`、`/work2/...`、
   `/root/...`、主机名。路径一律用环境变量 `$WS` / `$MODELS` / `$DAS185` / `$WORK`，
   脚本头部带上默认值（`${VAR:-...}`）。
   - 提交前跑一次自查：
     ```bash
     rg -n "/public/home/|/work2/|/root/|f[0-9]+r[0-9]+n[0-9]+|10\.[0-9]+\.[0-9]+\.[0-9]+" .
     ```
     有输出 = 不干净，先脱敏。
2. **版本锁定**：das185 全家桶的版本是逐坑踩出来的，改动要说明理由并附实测。
3. **如实标注状态**：没跑通的就写「卡在 X」，不要写成「支持」。`README.md` 模型表
   的「部署状态」列是仓库信誉所在。
4. **踩坑记录五步**：基线 → 定位 → 根因 → 修复 → 验证，每条带关键数字和报错原文，
   写进 `GOTCHAS.md`。
5. **脚本语法**：提交前 `bash -n` 所有改动的 `.sh`、`python3 -c "import ast; ..."`
   所有改动的 `.py`。

---

## 四、提 PR 流程

1. Fork → 新分支（`feat/gfx928`、`fix/moe-backend` 之类）。
2. 改代码 + 改文档**一起提交**（脚本和它对应的矩阵/条目要同步）。
3. 自查脱敏 + 语法（见上）。
4. PR 描述里写清楚：**在什么真机/什么架构上、跑了什么、结果如何**。
   - 有真机证据 → 直接写结果；
   - 没有真机（纯推算/骨架）→ 明确标注「未实测」，不要假装跑过。

---

## 五、最希望被解决的缺口（捡一个做）

按优先级，详见 `docs/KNOWN_ISSUES.md`：

1. **复跑 Pro-0813**（差 `VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0` 未验证）——最小工作量、
   价值最高，`sbatch scripts/deploy/slurm/v4pro_4n32c_serve.sh` 即可。
2. **gfx928 真机核实**：显存/算子支持，补全 `SUPPORTED_HARDWARE.md` 那一列。
3. **FP4/FP8 MoE 算子**：拿到新 aiter 或厂商补丁后复测 `flash*_8card_smoke.sh`。
4. **Pro-Base 首次部署**：按 1.46TB 重算并行参数，补 serve 脚本。
