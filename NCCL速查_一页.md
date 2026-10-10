# NCCL 速查 · 一页（背这个）

> 用途：被问"你了解 NCCL 吗"，把这一页讲出来就够了。
> 出处：`refs/nccl/`（NCCL 2.32.3），每条都有源码依据，不是背书。

---

## 一句话总览

> **NCCL 的算法和协议不是写死的** —— 它用一个代价模型给每个 (算法, 协议) 组合估时间，选最小的那个；**而且这个选择过程对外部插件开放。**

---

## ① 谁在选 —— `src/tuning/tuning.cc` 的 `ncclTuningCompute()`

三步：

```
① ncclTuningComputeAllTunings()   → 给每个 (algo, proto) 组合估一个 timeUs
② if (comm->tuner)                → ★ 外部 tuner 插件可以改写这张代价表
③ ncclTuningSelectBestTuning()    → 选 timeUs 最小的
```

另外：**单卡（`nRanks <= 1`）直接给 Ring + Simple** —— 只有一个 rank 时没有选择问题。

---

## ② 凭什么选 —— 代价模型

**`src/tuning/ring.cc`**：时间 = 数据量 / 带宽 + 延迟

- 延迟**按拓扑层级（机内 / 机间）× 算法 × 协议**分别标定，不是一个常数
- `Simple` 协议的机间延迟**算两遍**（推送 + ACK 是一个往返）

**`src/tuning/cost_model.h`**：有 `treeCorrectionFactor`

```c
// Trees are not perfectly sticking to the model for medium sizes.
// Applying a static correction factor is not ideal but works quite well.
```

> 翻译：**Tree 在中等消息大小上不符合模型，所以要打静态修正因子**，注释自己承认"不理想，但管用"。
> **结论：这是"模型 + 实测校准"的混合，不是纯理论。**

---

## ③ 能改吗 —— 两条路

| 路径 | 怎么做 | 动不动核心 |
|---|---|---|
| 改源码 | 改 `src/tuning/` 的代价模型或权重 | ✅ 动 |
| **写 tuner plugin** | 实现 `getCollInfo()` 改写代价表 | ❌ **不动** |

插件失手（不返回 `ncclSuccess`）→ **NCCL fallback 到默认**，不会崩。

而 NET plugin（接网卡）、CollNet plugin（集合通信卸载）是同一套设计：

> **NCCL 把"会随硬件变的部分"全部做成了扩展点。**

---

## ④ 跟我自己做的有什么关系（★ 这是差异化）

| | 我的 ccl-lab | NCCL |
|---|---|---|
| 模型骨架 | `T = 2(N-1)/N · (S/B + α)` | 同样是"数据量/带宽 + 延迟" |
| 延迟 α | 一个常量 | 按拓扑×算法×协议分别标定 |
| 实测校准 | 无（纯模型） | **有**（`treeCorrectionFactor`） |

**我注入 5ms RTT 后算法选择整体翻转（ring 1.27→51ms，recdouble 0.23→9.7ms），本质就是 α 项压过了 S/B 项。**
→ **我做的正是"模型"那一半，而且我很清楚模型在哪里会不够、哪里必须靠实测补。**

---

## ⑤ 直接背这一段

> "NCCL 的算法和协议选择是**可插拔**的。
> 它由 `tuning.cc` 里的 `ncclTuningCompute` 决定：给每个 (算法, 协议) 组合估一个耗时填进 `generalTable`，然后选最小的那个。
> **代价模型本身是'数据量/带宽 + 延迟'**，但延迟不是常数——它按拓扑层级、算法、协议分别标定；而且 `cost_model.h` 里专门有个 `treeCorrectionFactor`，注释写着 Tree 在中等消息大小不严格符合模型，**加静态修正因子'不理想但管用'**。所以真实做法是**模型 + 实测校准**。
> **关键设计是它允许外部 tuner 插件改写这张代价表**——`getCollInfo` 拿到的就是表的指针，插件失手还有 fallback。**所以适配新硬件可以不改核心，写个 tuner 就行。** NET plugin 接网卡、CollNet plugin 做卸载，是同一套思路：**把会随硬件变的部分都做成扩展点。**
> **我自己那套集合通信原型做的就是模型那一半**：用 α-β 模型解释小消息看步数、大消息看搬运量，注入长 RTT 后选择翻转——本质是 α 项压过了 S/B 项。**所以我清楚模型能解释到哪、哪里必须靠实测。**"

---

## 附：能用上的源码路径（被追问时甩出来）

```
src/tuning/tuning.cc          ← 选择逻辑（ncclTuningCompute / generalTable）
src/tuning/ring.cc            ← Ring 的代价模型
src/tuning/cost_model.h       ← treeCorrectionFactor（模型不够的证据）
src/plugin/tuner/tuner_v6.cc  ← tuner 插件接口
src/plugin/net/net_v12.cc     ← NET 插件怎么被加载（dlopen + dlsym）
src/include/plugin/nccl_net.h ← NET 插件符号名：ncclNetPlugin_v12
src/transport/net_socket.cc   ← 没有 RDMA 时的 TCP 兜底
```
