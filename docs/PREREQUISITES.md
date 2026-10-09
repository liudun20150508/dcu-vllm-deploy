# 前置条件与满足度清单（PREREQUISITES）

> 把「跑通 DeepSeek-V4 推理」需要满足的条件**逐条列出来**，每条标注当前状态：
> ✅ 满足 · ⚠️ 部分满足/未核实 · ❌ 不满足。
> **不满足的条件就是当前跑不通的原因**，看这一份就够。

---

## 〇、一句话结论

**「跑通 DeepSeek-V4 真实推理」这个目标，当前不满足——卡在「MoE 算子」这一个
硬条件上**。硬件、软件栈、通信、权重四层都已就绪，唯独推理前向需要的 fp4/fp8
MoE 算子缺口，本地无快速路径，要等厂商补丁。

---

## 一、条件总表

| # | 条件 | 状态 | 不满足的后果 |
|---|---|---|---|
| 1 | DCU 硬件（gfx938 实测目标） | ✅ 满足 | — |
| 2 | DCU 硬件（gfx928） | ⚠️ 未核实 | 换卡要重新确认显存/算子 |
| 3 | das185 软件栈（torch/aiter/vllm） | ✅ 满足 | — |
| 4 | 4 个适配补丁（`das185_patch.py`） | ✅ 满足 | — |
| 5 | RCCL 通信 | ✅ 满足 | — |
| 6 | 模型权重下载 + 加载 | ⚠️ Flash 系满足 / Pro-Base 未部署 | Pro-Base 装不进显存 |
| 7 | **FP4 MoE 算子（Flash-0731）** | ❌ **不满足** | 推理前向一进 MoE 就崩 |
| 8 | **FP8 MoE 算子（Flash-Base）** | ❌ **不满足** | 同上 |
| 9 | Pro-0813 fast-woa 修复验证 | ⚠️ 修复已写、未复跑 | 不能确认已跑通 |
| 10 | 海光 triton（`libgcvm.so.17git`） | ❌ 不满足 | triton / Megatron 不可用 |

---

## 二、逐条明细

### ✅ 满足的（不用管）

**1. DCU 硬件（gfx938）**
海光 BW（C-3000），gfx938，80 CU，64 GB，warpSize 64。真机量过，是本框架主测目标。

> **验证代码**：`scripts/verify/check_hw.sh`
> **实测输出**（验收报告）：
> ```
> gcnArchName : gfx938:sramecc+:xnack-
> multiProcessor: 80
> warpSize      : 64
> totalGlobalMem: 64.0 GB
> ```

**3. das185 软件栈**
torch 2.10.0+das.opt1.dtk2604 / aiter 0.1.5+das185 / vllm_hcu 0.21.0+das.dtk2604 /
triton 3.6.0 / tilelang 0.1.9 / flash_mla 1.2.0。全部装好、版本锁定。

> **验证代码**：`scripts/install/das185_check.py`
> **实测输出**：
> ```
> torch 2.10.0  hip 6.3.26113  avail True  dev 8
> OK  vllm / vllm_hcu / aiter / triton / tilelang / flash_mla
> ```

**4. 适配补丁**
`scripts/install/das185_patch.py` 四个补丁（aiter shuffle / triton flags /
gfx938 MXFP4 门槛 / fused_moe 签名）已应用，幂等可重跑。

> **验证代码**：`scripts/install/das185_patch.py`（幂等，重跑输出 `已存在，跳过` 即已应用）

**5. RCCL 通信**
all_reduce 正确性通过；单机 8 卡 256MB busbw **109 GB/s**，跨机 shca 四平面
16 卡 **68.4 GB/s**（2→4 节点掉 19%、4→8 持平）。**通信层不是瓶颈。**

> **验证代码**：`scripts/verify/rccl_allreduce.py` + `scripts/verify/rccl_allreduce.sh`
> **实测输出**：
> ```
> [rank 0] ALLREDUCE OK sum=8.0 ws=8
> [bw] all_reduce 256MB  busbw=109.22 GB/s   （单机 8 卡）
> ```
> 用法：`bash scripts/verify/rccl_allreduce.sh --np 8`（单机）/ `sbatch scripts/verify/rccl_allreduce.sh`（跨机）

### ⚠️ 部分满足 / 未核实

**2. gfx928**
- gfx928（深算二号 K100）：显存按公开规格推 96 GB，**未真机量过**，算子支持未知。
- 后果：拿这块卡跑之前，先跑 `scripts/install/das185_check.py` 探针，别套 gfx938 结论。

**6. 模型权重**
| 模型 | 权重 | 状态 |
|---|---|---|
| Flash-0731 | 156 GB | ✅ 48 分片全下载、校验通过、加载进显存 |
| Flash-Base | 275 GB | ✅ 46 分片全下载、校验通过、加载进显存 |
| Pro-0813 | 832 GB | ✅ 已下载、进过显存（8 次排障） |
| **Pro-Base** | **1.46 TB** | ❌ **从未下载、从未部署**（只有下载脚本） |

**9. Pro-0813 fast-woa 修复**
`VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0` 已写入脚本，但**最后一次作业结束后才改好，
未重新提交验证**。这是四模型里最接近跑通的，复跑 `sbatch scripts/deploy/slurm/v4pro_4n32c_serve.sh` 即可确认。

### ❌ 不满足的（跑不通的根因）

**7. FP4 MoE 算子（Flash-0731）—— 三条后端全死**

| 后端 | 失败 |
|---|---|
| AITER | `ck_moe` 不认识 fp4 dtype（`Float4_e2m1fn_x2`=45 → `Unsupported data type 45`） |
| TRITON | 汇编出 gfx950 指令 `v_cvt_pk_f32_fp8_e32`，gfx938 报 `instruction not supported` |
| EMULATION | `Mxfp4MoEMethod` 硬拒绝（`Unsupported mxfp4_backend`） |

**8. FP8 MoE 算子（Flash-Base）—— 三条后端全死**

| 后端 | 失败 |
|---|---|
| AITER | `MoeSortingMultiPhaseKernel` VM Fault（非法地址，8 worker 一起崩） |
| TRITON | 不支持 `block 128×128` 量化方案 |
| MARLIN | 仅 `is_cuda()`，gfx938 不支持 |

**解决方（唯一现实路径）**：向海光/曙光要 gfx938 版 DeepSeek-V4 MoE 算子
（换更新版 aiter 或要补丁）。拿到后直接复测：
`scripts/deploy/smoke/flash0731_8card_smoke.sh` / `flash_base_8card_smoke.sh`。

**10. 海光 triton 缺 `libgcvm.so.17git`**
该库不在 DTK 26.04、不在 `/opt/hyhal`、不在 triton 自带 `triton.libs/`。
**后果**：triton 及依赖它的 Megatron 不可用（**vLLM 不受影响**）。
**绕过**：换 PyPI 官方 triton 3.1.0（本项目 Megatron 就是靠这条绕过去的）。

---

## 三、要在今天跑通推理，还差哪几步

按难度从低到高，对应 `CONTRIBUTING.md` 的缺口优先级：

1. **复跑 Pro-0813**（5 分钟）—— 只差一个开关没验证，`sbatch .../v4pro_4n32c_serve.sh`。
   ⚠️ 但 Pro-0813 用的是 mxfp4，同样可能撞上条件 7 的 fp4 缺口，复跑才知道。
2. **确认 gfx928 规格**（半小时）—— 跑探针，补全 `SUPPORTED_HARDWARE.md`。
3. **等厂商补丁**（外部依赖）—— 解决条件 7/8 的 MoE 算子，这是唯一能真正跑通
   Flash-0731 / Flash-Base 的路。
4. **补 libgcvm**（外部依赖）—— 海光驱动包，恢复 triton/Megatron。
5. **Pro-Base 首次部署**（1 天+）—— 下载 1.46TB + 按显存重算并行参数。

**结论：除非拿到条件 7/8 的厂商算子补丁，Flash-0731 和 Flash-Base 的推理前向
在当前 gfx938 上是跑不通的。这不是脚本能解决的，如实记录。**
