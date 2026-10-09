# SUPPORTED_HARDWARE — DCU 架构 × 模型 条件矩阵

> 本文件回答一个问题：**"我手头的卡能不能跑 DeepSeek-V4？差在哪？"**
> 所有结论来自真机实测 + 公开规格；标注 `*` 的项是**未实测、按公开资料推算**，
> 使用前务必先用 `scripts/install/das185_check.py` 探针实测。

---

## 一、DCU 架构速查

| 架构代号 | 型号 | 单卡显存 | CU 数 | warpSize | 每节点卡数 | 实测状态 |
|---|---|---|---|---|---|---|
| gfx906 | 深算一号 Z100L | 16 GB | 64 | 64 | **4（实测）** | ✅ 真机量过（`rocminfo`/`mem_info_vram_total`） |
| gfx938 | 海光 BW（C-3000，BW-1 队列） | 64 GB | 80 | 64 | 8 | ✅ 真机量过（本框架主测目标） |
| gfx928 | 深算二号 K100 | 96 GB* | — | — | 8 | ❌ 未实测，显存按公开规格推算 |

> **共性**：海光/AMD 走 wave64（warpSize=64），与 NVIDIA 的 warp32 不同，
> 写 kernel 时锁步和分支粒度要按 64 处理。

---

## 二、核心结论：架构决定 MoE 后端支持

DeepSeek-V4 是 MoE（256/384 专家）架构，推理前向的主要瓶颈在 **MoE 算子的
后端支持**，而这**强依赖 gfx 架构**：

| MoE 后端 | gfx906 | gfx938 | gfx950 |
|---|---|---|---|
| AITER CK MoE（fp4/fp8） | ❌ 无 code object | ⚠️ 门槛可放开但 dtype 不全 | ✅ 完整 |
| TRITON 量化 MoE | ❌ 缺量化方案 | ⚠️ 汇编出 gfx950 指令 | ✅ |
| MARLIN | ❌ 仅 is_cuda() | ❌ 仅 is_cuda() | ❌ |
| EMULATION（反量化） | ⚠️ mxfp4 硬拒 | ⚠️ mxfp4 硬拒 | ✅ |

**一句话**：gfx950 系列（后续旗舰）才完整支持 DeepSeek-V4 的 fp4/fp8 MoE；
gfx938 目前卡在算子库内部缺失；gfx906 老架构缺得最多。

---

## 三、模型 × 架构 × 节点数（每节点 8 卡口径，纯权重下限）

| 模型（权重） | gfx906 (16G×4=64G) | gfx928 (96G×8=768G*) | gfx938 (64G×8=512G) |
|---|---|---|---|
| Flash-0731 (156G) | 3 | 1 | 1 |
| Flash-Base (275G) | 5 | 1 | 1 |
| Pro-0813 (832G) | 13 | 2 | 2 |
| Pro-Base (1.46T) | 24 | 2 | 3 |

**加 1.5× 余量**（KV Cache + 激活 + workspace）：

| 模型 | gfx906 | gfx928 | gfx938 |
|---|---|---|---|
| Flash-0731 | 4 | 1 | 1 |
| Flash-Base | 7 | 1 | 1 |
| Pro-0813 | 20 | 2 | 3 |
| Pro-Base | 35 | 3 | 5 |

\* gfx928 显存按 96G 推算，需真机核实；906 按实测 4 卡/节点、928/938 按 8 卡/节点。
**你目标集群每节点几张卡，先 `sinfo`/`scontrol` 确认，再套表。**

---

## 四、软件栈条件（DTK / DAS 两代）

| 线 | DTK | torch | vllm_hcu | aiter | triton | 适用架构 |
|---|---|---|---|---|---|---|
| **das185（本框架）** | 26.04 | 2.10.0+das.opt1 | 0.21.0+das.dtk2604 | 0.1.5+das185 | 3.6.0+das185 | gfx938 |
| **das（旧）** | 25.04 | 2.5.1+das.opt1 | 0.11.0+das | — | 3.1.0+das | gfx906 |

> 两条线对应不同 torch/vllm 大版本，**不可混用**。das185 是给 gfx938 这代
> 新卡准备的；gfx906 老集群用 das（DTK 25.04）那套。

---

## 五、网卡 / RCCL 条件（两套实测集群，别混）

| 网卡 | 所属集群（规模） | 每节点设备数 | 链路 | 速率 | RCCL 怎么走 | 条件 |
|---|---|---|---|---|---|---|
| 自研 `shca_*` | hx1hdnormal01（gfx938，95 节点 / 760 卡） | **4 张**（shca_0..3） | InfiniBand | 400G | `NCCL_NET_PLUGIN=librccl-net-shca.so` | 专有插件，向管理员索取 |
| Mellanox `mlx5_*` | kshdnormal（gfx906，494 节点 / 1976 卡） | **4 张**（mlx5_0..3） | InfiniBand | 200G | 内置 IB verbs | 实测仅 `mlx5_0` 能建 QP，需 `NCCL_IB_HCA=mlx5_0` |

**性能参照（实测）**：
- shca 四平面 2 节点 16 卡 AllReduce：**68.4 GB/s**（超过单口 400G 理论值，多网卡并行生效）
- 4 节点 32 卡：55.2 GB/s；8 节点 64 卡：54.8 GB/s（2→4 掉 19%，4→8 持平，多平面未吃满）
- 单节点 8 卡片内：256MB AllReduce **109 GB/s**（不走网卡）

---

## 六、适配前自检清单

```bash
# 1. 架构
rocminfo | grep -iE "gfx|Compute Unit"          # 拿到 gfxXXX + CU 数
# 2. 显存（权威口径）
cat /sys/class/drm/card*/device/mem_info_vram_total   # 或 rocm-smi/hy-smi
# 3. 每节点卡数 + 分区
sinfo -o "%P %D %c %m %G"                        # Slurm 集群
# 4. 网卡
ibstat | grep -iE "^CA|Rate|Link layer"          # shca_* vs mlx5_*
# 5. DTK
module load compiler/dtk/26.04 && echo $DTKROOT
```

**对照上面的矩阵**，判断你的卡属于哪一行，再决定走哪套脚本、能不能跑通。
