# experiments/ — 算子层实验代码

> 这些是「绕开 aiter 缺口的自力更生路线」的 HIP kernel，对应 `docs/KNOWN_ISSUES.md`
> §四 和 `docs/PREREQUISITES.md` 条件 7/8 的失败根因。**当前都是半成品**：
> 只证明思路能跑，还没有做到"顶掉 aiter 的完整 MoE GEMM"。

| 文件 | 作用 | 状态 |
|---|---|---|
| `v4_fp4_dequant.hip` | DeepSeek-V4 FP4 专家权重反量化 kernel（e2m1fn 语义） | ✅ 反量化 GPU==CPU 验证通过，只是 fp4_gemm 的第一步 |
| `vector_add.hip` | HIP 最简向量加法（gfx936 上手用） | ✅ 可跑 |
| `dcu_kernel_lab.hip` | wave64 / 显存带宽 / 归约 / P2P 实验台 | ✅ 可跑 |

## 编译运行（DTK 环境）

```bash
module load compiler/dtk/26.04
hipcc -O2 --offload-arch=gfx936 -o v4_fp4_dequant v4_fp4_dequant.hip && ./v4_fp4_dequant
# 预期： [PASS] FP4 反量化 kernel：4096 个值 GPU==CPU 全部一致
```

## 为什么在这里

FP4/FP8 真实推理卡在 aiter 的 MoE 算子缺口（`docs/PREREQUISITES.md` 条件 7/8）。
这些 HIP kernel 是"如果厂商补丁拿不到，就从 HIP 手写 W4A16 MoE GEMM 顶掉 aiter"
的起点。**当前只到 dequant 第一步，完整 GEMM + 可用性能工作量大、风险高**，
优先走厂商补丁，这条是最后手段。
