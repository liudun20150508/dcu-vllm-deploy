# KNOWN_ISSUES — 缺口清单（还差什么）

> 本文件如实记录"当前还差什么、卡在哪、谁来解决"。开源的目的之一就是让
> 适配者从这里接手，而不是以为开箱即用。

---

## 一、阻塞级缺口（推理前向跑不通的根因）

### 1. gfx938 版 DeepSeek-V4 MoE 算子缺失（FP4 + FP8）

**现象**：环境/权重/RCCL 三层全通，但一进 MoE 前向就崩。

| 精度 | 后端 | 失败根因 |
|---|---|---|
| FP4 (Flash-0731) | AITER | ck_moe 不认识 fp4 dtype（`Float4_e2m1fn_x2`=45 → `Unsupported data type 45`） |
| FP4 | TRITON | 汇编出 gfx950 指令 `v_cvt_pk_f32_fp8_e32`，gfx938 报 `instruction not supported` |
| FP4 | EMULATION | `Mxfp4MoEMethod` 硬拒绝（`Unsupported mxfp4_backend`） |
| FP8 (Flash-Base) | AITER | `MoeSortingMultiPhaseKernel` VM Fault（非法地址，8 worker 一起崩） |
| FP8 | TRITON | 不支持 `block 128×128` 量化方案 |
| FP8 | MARLIN | 仅 `is_cuda()`，gfx938 不支持 |

**根因**：das185 这套 wheel（aiter 0.1.5 + vllm 0.21.0）的 DeepSeek-V4 MoE 算子
在 gfx938 上支持不完整。**本地无快速路径。**

**解决方**：
1. 换更新版 aiter（带 gfx938 的 fp4/fp8 DeepSeek-V4 MoE 支持）；
2. 或向海光/曙光要 gfx938 的 DeepSeek-V4 MoE 算子适配补丁。

**复测脚本**（拿到新 wheel 后直接复用）：
- `scripts/deploy/smoke/flash_base_8card_smoke.sh`（FP8，TP=8）
- `scripts/deploy/smoke/flash0731_8card_smoke.sh`（FP4，TP=8）

---

### 2. 海光 triton 缺 `libgcvm.so.17git`

海光 triton 包缺 `libgcvm.so.17git`，该库不在 DTK 26.04、不在 `/opt/hyhal`、
不在 triton 自带 `triton.libs/`（其 RPATH 正指向那里）。
**⇒ triton 及依赖它的 Megatron 目前不可用**；vLLM 不受影响。

**解决方**：海光驱动包（对应版本的 gcvm）。或换 PyPI 官方 triton 3.1.0
（本项目 Megatron 就是靠这条绕过去的）。

---

## 二、半成品缺口（差一步复跑验证）

### 3. Pro-0813 fast-woa dtype 修复未复跑

Pro-0813（"0831pro"）8 次部署排障已到根因：
`deepseek_v4_attention` 的 fast-woa 分支拿 fp8 `wo_a` 直接做 bf16 einsum → dtype 错。

**修复已写入**：`export VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0`
（走 `rocm_inv_rope_einsum`，内部先反量化）。

**但最后一次作业结束后脚本才改好，未重新提交验证。** 适配者第一步就复跑这个：
```bash
sbatch scripts/deploy/slurm/v4pro_4n32c_serve.sh
```

---

## 三、未核实项

| 项 | 状态 |
|---|---|
| gfx928 真机规格（显存 / 算子支持） | 未实测，节点数按 96G 推算 |
| Pro-Base（1.46TB）部署 | 仅下载脚本，从未部署验证 |

---

## 四、可选的自力更生路线（工作量大）

从 HIP 手写 fp4 W4A16 MoE GEMM 顶掉 aiter。
`experiments/v4_fp4_dequant.hip`（在原工作区）只做了 dequant 第一步；
完整 GEMM + 可用性能工作量大、风险高，且 fp8 的 sorting bug 同样绕不开。

**结论：优先走厂商补丁，HIP 重写是最后手段。**
