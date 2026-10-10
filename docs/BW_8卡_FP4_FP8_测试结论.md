# BW 8 卡 FP4 / FP8 DeepSeek-V4-Flash 测试结论

> 机器：海光 BW（gfx936，C-3000 卡）单机 8 卡 · 测试日期：2026-10-07（深夜收尾）
> 对象：DeepSeek-V4-Flash-0731（FP4 专家，156G）+ DeepSeek-V4-Flash-Base（FP8 专家，275G）
> 结论：**环境全通、权重全进、RCCL 通信正常，但真实推理前向卡死在底层 MoE 算子，本地已无快速路径。**

---

## 一、一句话结论

FP4 和 FP8 都**没跑通真实推理**，且**卡在同一个根子**上：das185 这套 wheel（aiter 0.1.5 + vllm 0.21.0）的 DeepSeek-V4 MoE 算子在 gfx936 上支持不完整。这不是配置/脚本能绕过的硬门槛。

---

## 二、环境层面：已全部就绪 ✅

| 项 | 结果 |
|---|---|
| das185 环境重建 | torch 2.10.0 认到 **8 张 BW(gfx936)** |
| 8 卡 RCCL 通信 | all_reduce 正确（sum=8.0），256MB busbw 109 GB/s |
| 0731 权重（156G fp4） | 48 分片全下载、校验通过、实际加载进显存 |
| Base 权重（275G fp8） | 46 分片全下载、校验通过、实际加载进显存 |

权重加载进度实证：`Loading safetensors checkpoint shards: …/48`、`…/46` 均实际推进（日志抓到 40/48、36/46 等快照）。

---

## 三、卡死位置：一进 MoE 前向就崩 ❌

时间线：权重下载全 → 校验通过 → vllm 读权重进显存 → **`profile_run`（MoE kernel 编译/探测）→ 报错退出**。

### FP4（Flash-0731，专家 e2m1fn）—— 三条后端全死

| 后端 | 失败根因 |
|---|---|
| AITER | ck_moe 不认识 fp4 dtype（`Float4_e2m1fn_x2`=45）→ `CKPyInterface: Unsupported data type 45` |
| TRITON | 汇编出 gfx950 才有的指令 `v_cvt_pk_f32_fp8_e32`，gfx936 报 `instruction not supported` |
| EMULATION | `Mxfp4MoEMethod` 硬拒绝（`Unsupported mxfp4_backend`） |

### FP8（Flash-Base，专家 fp8）—— 三条后端也全死

| 后端 | 失败根因 |
|---|---|
| AITER | `MoeSortingMultiPhaseKernel` VM Fault（非法地址 `0x7f8b8bc6b000`，8 worker 一起崩） |
| TRITON | 不支持 `block 128×128` 量化方案 |
| MARLIN | 仅 `is_cuda()`，gfx936 不支持 |

---

## 四、根因定性

分三层看，本地能改的已改完，剩下的是底层缺口：

1. **本地已改掉（表层门槛）** ✅
   - aiter MXFP4 门槛写死 `on_gfx950()` → 补丁改成 `on_gfx950() or gfx936/gfx938`
   - aiter `fused_moe` 签名缺 kwarg → 补丁加上
   - 环境依赖：numpy 2.1.3 崩 torch C ABI、缺 openai-harmony/uvloop/grpcio 等、triton 补丁被覆盖、core dump 写满 50G 持久盘

2. **本地改不了（算子库内部缺失 / bug）** ❌ 主因
   - FP4：ck_moe 内部缺整个 fp4 GEMM 的 kernel 实现（dtype 45 不认识）
   - FP8：aiter 的 sorting kernel 内存访问 bug（VM Fault）

3. **硬件指令 / 编译器层面（真·不支持）** ❌
   - triton 为 gfx936 汇编出 `v_cvt_pk_f32_fp8_e32` —— gfx950 才有，gfx936 硬件没有
   - MARLIN 后端只 `is_cuda()`，gfx936 不可用

> 关键澄清：之前**单卡 dummy 冒烟能通**，是因为把规模缩到 16 experts、且用 fp8 专家绕开了 fp4；真实 **256 experts 完整规模**才触发这些缺口。

---

## 五、下一步（厂商侧，本地已无快速路径）

1. 换**更新版 aiter**（带 gfx936 的 fp4/fp8 DeepSeek-V4 MoE 支持）；
2. 或向海光/曙光要 **gfx936 的 DeepSeek-V4 MoE 算子适配补丁**；
3. 拿到新 wheel 后，直接复用现成脚本复测：
   - `验收/slurm/flash_base_8card_smoke.sh`（FP8，TP=8）
   - `验收/slurm/flash0731_8card_smoke.sh`（FP4，TP=8）
   - 环境变量、8 卡 TP=8 参数均已就绪。

本地唯一理论空间：用 HIP 从零写 fp4 W4A16 MoE GEMM 顶掉 aiter（`experiments/v4_fp4_dequant.hip` 是第一步，只做了 dequant），但完整 GEMM + 可用性能工作量大、风险高，且 fp8 的 bug 同样绕不开。

---

## 六、相关文件索引

| 文件 | 说明 |
|---|---|
| `曙光环境/gfx936单卡适配记录.md` | gfx936 环境重建 + dummy 冒烟全过程（踩坑 7 条 + 环境变量） |
| `验收/slurm/das185_patch.py` | 4 个补丁（MXFP4 门槛 / fused_moe 签名 / triton flags / aiter shuffle），幂等 |
| `验收/slurm/das185_check.py` / `gfx936_smoke_probe.py` | 环境探针 / triton 编译探针 |
| `验收/slurm/gfx936_dummy_smoke.py` | dummy 权重单卡冒烟（能通的那条链） |
| `验收/slurm/flash0731_8card_smoke.{py,sh}` | FP4 真实权重 8 卡冒烟（卡 MoE） |
| `验收/slurm/flash_base_8card_smoke.{py,sh}` | FP8 真实权重 8 卡冒烟（卡 MoE） |
| `验收/slurm/dupfix/sitecustomize.py` | op 冲突兜底（Duplicate op name） |
| `8卡测试_reasonix会话记录/` | 服务器 reasonix 完整会话 jsonl（含全部测试日志，本结论原始出处） |
