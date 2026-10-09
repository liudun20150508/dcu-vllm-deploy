# 模型部署记录（MODEL_RUNBOOK）

> 按 **4 个模型**记录：之前部署/跑过什么、用哪个脚本、什么命令、**改了什么代码
> （补丁 / 算子 / 编译器）**。给接手的人一个"这个模型目前走到哪、改动在哪"的索引。
> 与 `DEPLOYMENT.md`（怎么部署）互补：本文件是**每个模型的历史轨迹 + 改动台账**。

---

## 〇、四模型速览

| 模型 | 精度/权重 | 状态 | serve 脚本 | smoke 脚本 | 下载脚本 |
|---|---|---|---|---|---|
| Flash-0731 | FP4 / 156G | 卡 MoE 算子 | `flash0731_serve.sh` | `flash0731_8card_smoke.sh` | `dl_flash_0731_base.sh` |
| Flash-Base | FP8 / 275G | 卡 MoE 算子 | `flash_base_serve.sh` | `flash_base_8card_smoke.sh` | 同上 |
| Pro-0813 | MXFP4/FP8 / 832G | 差 1 开关未复跑 | `v4pro_4n32c_serve.sh` / `v4pro_20n80c_serve.sh` | — | 无独立脚本 |
| Pro-Base | FP8 / 1.46T | 从未部署 | — | — | `dl_v4pro_base.sh` |

> 状态含义见 `docs/PREREQUISITES.md`：前三个模型「环境/权重/RCCL」三层都通，
> 卡点集中在 MoE 算子（等厂商补丁），不是脚本问题。

---

## 一、Flash-0731（FP4，156G）

**部署历史**：单卡 dummy 冒烟通（`gfx938_dummy_smoke.py`）→ 8 卡真实权重冒烟
卡 MoE → Slurm 多机 serve 同样卡 MoE。

**命令**：
```bash
bash deploy.sh download --model flash0731                # 下载权重（156G/48 分片）
bash deploy.sh smoke   --model flash0731                 # 单机 8 卡冒烟（TP=8）
bash deploy.sh serve   --model flash0731                 # Slurm 4n×4卡 TP4 PP4（老集群 A）
```

**改过的代码**：
- `scripts/install/das185_patch.py` 补丁3：放开 gfx938 的 MXFP4 MoE 门槛（`on_gfx950()` → 加 gfx938）
- `scripts/install/das185_patch.py` 补丁4：给 aiter `fused_moe` 补签名 kwarg
- `patches/sitecustomize.py`：op 冲突兜底（Duplicate op name）

**算子**：FP4 专家走 AITER CK MoE，卡在 `Unsupported data type 45`（见 GOTCHAS 坑位）。

---

## 二、Flash-Base（FP8，275G）

**部署历史**：8 卡真实权重冒烟卡 MoE（FP8 sorting VM Fault）→ Slurm 8n×4卡 serve 同卡。

**命令**：
```bash
bash deploy.sh download --model flashbase                # 与 0731 同一下载脚本
bash deploy.sh smoke   --model flashbase                 # 单机 8 卡冒烟（TP=8）
bash deploy.sh serve   --model flashbase                 # Slurm 8n×4卡 TP4 PP8（老集群 A）
```

**改过的代码**：
- `patches/sitecustomize.py` 坑17 实验开关 `DUPFIX_FORCE_FP8=1`：老卡 gfx906 走 vllm 内置 TRITON fp8 核
- 其余补丁与 Flash-0731 共用（`das185_patch.py` 补丁1/2/4）

**算子**：FP8 专家，卡在 `MoeSortingMultiPhaseKernel` VM Fault（见 GOTCHAS 坑位）。

---

## 三、Pro-0813（"0831pro"，832G）

**部署历史**：8 次多机部署排障，一路修到最后一个开关，**尚未复跑验证**。

**命令**（两种集群各一套）：
```bash
# 新集群 B（gfx938，8卡/节点，shca 400G）—— 推荐，最接近跑通
bash deploy.sh serve --model pro0813                    # 4n×8卡 TP8 PP2 DP2 EP16

# 老集群 A（gfx906，4卡/节点，mlx5 200G）—— 20 节点拆小
bash deploy.sh serve --model pro0813 --script scripts/deploy/slurm/v4pro_20n80c_serve.sh
```

**改过的代码（8 个坑的修复都在这几个文件里）**：
- `patches/aiter_ops_shuffle_a16w4.patch`：补 `shuffle_weight_a16w4`/`shuffle_scale_a16w4`（坑5）
- `scripts/deploy/slurm/run_v4pro_node.sh`：ZMQ 走 fabric 网 / TMPDIR / aillvm / fast-woa 开关
- `patches/sitecustomize.py`：op 冲突 + `DUPFIX_FORCE_AITER`
- 关键开关 `VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0`：fp8 `wo_a` 反量化路径（坑9）

**当前卡点**：`fast-woa` 修复已写入但最后一次作业结束后才改好，**复跑
`sbatch scripts/deploy/slurm/v4pro_4n32c_serve.sh` 即可验证**（见 KNOWN_ISSUES §3）。

---

## 四、Pro-Base（FP8，1.46TB）

**部署历史**：从未下载、从未部署。

**命令**（仅下载）：
```bash
bash deploy.sh download --model probase                 # 1.46TB / 64 分片，需本地大盘
```

**改过的代码**：无（无部署脚本）。

**要部署的话**：按 Pro-0813 的 4n32c 脚本改 `--script`，并行参数按 1.46TB 重算
（README 节点数表：gfx938 需 3 节点起）。

---

## 五、跨模型的代码改动台账（补丁 / 算子 / 编译器）

### 补丁（patches/ + install/）

| 文件 | 改了什么 | 服务于 |
|---|---|---|
| `scripts/install/das185_patch.py` | 4 个幂等补丁：aiter shuffle / triton flags / gfx938 MXFP4 门槛 / fused_moe 签名 | 全部 4 模型 |
| `patches/sitecustomize.py` | op 冲突兜底 + `DUPFIX_FORCE_AITER` / `DUPFIX_FORCE_FP8` + triton arch-spoof | 全部（按开关启用） |
| `patches/aiter_ops_shuffle_a16w4.patch` | 补 aiter 缺的 A16W4 shuffle 函数 | Pro-0813 |
| `patches/triton_hcu_compiler_flags.patch` | 跳过 aillvm 不认识的 2 个 flag | 全部 |

### 算子（experiments/）

| 文件 | 内容 | 状态 |
|---|---|---|
| `experiments/v4_fp4_dequant.hip` | FP4 e2m1fn 反量化 kernel（绕开 aiter 的起点） | ✅ 反量化 GPU==CPU 验证通过，只是 fp4_gemm 第一步 |
| `experiments/vector_add.hip` | HIP 最简向量加法 | ✅ 可跑 |
| `experiments/dcu_kernel_lab.hip` | wave64 / 带宽 / 归约 / P2P 实验台 | ✅ 可跑 |

### 编译器 / 运行时

| 组件 | 版本/路径 | 作用 |
|---|---|---|
| aillvm (clang-18) | `$WORK/dtk-shim/aillvm/bin/clang-18` | triton HCU 编译（DTK 自带 clang-17 不认 HCU 参数） |
| hipcc | `$DTKROOT/bin/hipcc` | triton launcher 的 `.c` 实为 C++，必须用它 |
| glibc shim | `patches/glibc232shim.c` → `libglibc232shim.so` | 老集群 glibc 2.31 缺 `__libc_single_threaded` |
| `AMDGCN_USE_BUFFER_OPS=0` | — | aillvm buffer rsrc intrinsic 签名不一致，关 buffer ops |

> 完整踩坑（每条带修复命令）见 `docs/GOTCHAS.md`；完整缺口见 `docs/KNOWN_ISSUES.md`。
