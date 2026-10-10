# NCCL 源码阅读路径（面向「通信库性能优化」岗）

> 依据：NCCL **2.32.3**，源码已拉到 `refs/nccl/`（`git clone --depth 1 https://github.com/NVIDIA/nccl.git`）
> 为什么读：JD 职责 2 原文——"参与 **NCCL、RCCL、MPI、UCX、UCC** 等通信库的**源码分析**、功能开发和性能调优"；优先条件 1——"有集合通信算法、**通信库源码分析**或性能优化相关科研、项目或实习经历"
> 在哪读：**Mac 上就行**。源码是文本，`rg` 直接查。
>
> ⚠️ 行号会随版本变动，**用 `rg` 搜符号名**，别死记行号。

---

## 〇、先建一张地图（这是新版结构，网上老文章写的是旧的）

```
refs/nccl/src/
├── collectives.cc          ← API 入口（ncclAllReduce 在这里）
├── enqueue/enqueue.cc      ← 主机侧任务编排：把 API 变成 device 任务
├── tuning/                 ← ★算法与协议怎么选出来（本岗最核心）
│   ├── tuning.cc           ← 主选择逻辑
│   ├── cost_model.cc       ← 代价模型
│   ├── ring.cc / tree.cc / nvls.cc / collnet.cc  ← 各算法的成本推导
│   └── tuning_general.cc
├── device/                 ← ★device 端 kernel 与协议实现
│   ├── all_reduce.h / all_gather.h / reduce_scatter.h / sendrecv.h
│   ├── prims_ll.h          ← LL 协议
│   ├── prims_ll128.h       ← LL128 协议
│   ├── prims_simple.h      ← Simple 协议
│   └── primitives.h / op128.h / common_kernel.h
├── transport/              ← ★传输层：同一个算法落到不同后端
│   ├── net_ib/             ← InfiniBand / RoCE（verbs）
│   │   └── gdr.cc          ← GPUDirect RDMA ← JD 优先条件 2
│   ├── net_socket.cc       ← ★没有 RDMA 网卡时的 TCP 兜底
│   ├── shm.cc              ← 单机共享内存（P2P）
│   ├── p2p.cc / nvls.cc / coll_net.cc
│   └── net.nc / net_ib_stub.cc / net_socket_stub.cc
├── plugin/net/             ← ★NET plugin 接口（多版本 ABI：net_v6 ~ net_v12）
├── graph/                  ← 拓扑探测与图搜索
│   ├── search.cc / topo.cc ← 探拓扑
│   ├── rings.cc / trees.cc ← 怎么排 ring / tree 的顺序
│   └── paths.cc / connect.cc
└── include/                ← 内部头文件（collectives.h / device.h / comm.h …）
```

**对上你的 ccl-lab**：`algos.py` ≈ `device/*.h` + `tuning/`，`transport.py` ≈ `transport/`，`run.py` ≈ 自建 benchmark。**你已经在正确的分层上做过一遍了。**

---

## 一、主线 1 · 一个 `ncclAllReduce` 走下去（30 分钟）

**读什么**

| 步 | 文件 / 符号 | 看什么 |
|---|---|---|
| 1 | `src/collectives.cc` → `rg -n "ncclResult_t ncclAllReduce"` | 公开 API；注意它只是薄壳 |
| 2 | 同文件 `ncclAllReduceConfigImpl` | 参数校验、`ncclEnqueueCheck` 之前做了什么 |
| 3 | `src/enqueue/enqueue.cc` | **任务编排**：host 侧怎么把这次调用变成 device 任务 |
| 4 | `src/include/collectives.h` | `ncclFunc_t` 等内部数据结构 |
| 5 | `src/device/all_reduce.h` | device 端真正干活的地方 |

**回答这三个问题**
1. 从 API 到 device kernel，中间隔了几层？
2. host 侧负责什么、device 侧负责什么？（**这决定"优化往哪改"**）
3. 一次调用里有没有同步点？

**价值**：被问"你了解 NCCL 吗"，你能说清它的分层，而不是只报算法名。

---

## 二、主线 2 · 算法与协议是怎么"选"出来的 ★ 最核心

**这一条是曙光这个岗的正面考点。**

**读什么**

```
src/tuning/tuning.cc
  rg -n "ncclTuningCompute\(" src/tuning/tuning.cc
  rg -n "ncclTuningSelectBestTuning|ncclTuningComputeAllTunings" src/tuning/tuning.cc
  rg -n "generalTable\[" src/tuning/tuning.cc        # ← "算法 × 协议"的代价表
```

关键位置（2.32.3）：
- `ncclTuningCompute()` 在 `tuning.cc:180` 附近——主选择函数
- 默认兜底：`bestTuning.algo = NCCL_ALGO_RING; bestTuning.proto = NCCL_PROTO_SIMPLE;`
- `generalTable[tuning.algo][tuning.proto] = tuning.timeUs`——**这张表就是"什么配置要多少时间"**

再往下：
- `src/tuning/cost_model.cc`——**代价模型**（新版的重要变化：不只是查表，而是建模估算）
- `src/tuning/ring.cc` / `tree.cc`——各算法各自的成本是怎么推出来的
- `src/tuning/tuning_general.cc`

**回答这两个问题（这就是常见题）**
1. **小消息为什么可能选 Tree 或低延迟协议？大消息为什么回落到 Ring？** → 去 `ring.cc` / `tree.cc` 看它们的成本公式里，**步数项和带宽项各占多少权重**
2. **那张"选择表"是怎么来的？** → 是实测出来的还是模型算的？（2.32 是**模型 + fallback** 的混合）

**这一步读完，你就能说**：
> "NCCL 的算法/协议选择不是拍脑袋，它有代价模型：`tuning/ring.cc` 和 `tree.cc` 分别给出 Ring 和 Tree 的时间估计，`tuning.cc` 拿这些估计比大小选最优，`generalTable[algo][proto]` 是最终的代价表。**这就是我在自己那套原型里做的事**——只是我用的模型更简单：Ring 的时间是 `2(N-1)/N · (S/B + α)`，所以小消息时 α 项主导（步数说话）、大消息时 S/B 项主导（搬运量说话）。"

**这一段话，比背十条结论都值钱。**

---

## 三、主线 3 · 三种协议（LL / LL128 / Simple）分别解决什么

**读什么**

```
src/device/prims_ll.h        ← 低延迟协议
src/device/prims_ll128.h     ← 128B 粒度
src/device/prims_simple.h    ← 直接写 + ACK
src/include/device.h         ← 协议相关的实现类选择；搜 NCCL_PROTO_SIMPLE
```

**三个问题**
1. **LL 为什么要 flag 同步、为什么只适合小消息？**（提示：它用更激进的同步换延迟，代价是每步开销固定）
2. **LL128 的 128 字节粒度是在折中什么？**
3. **Simple 为什么大消息赢？**（它数据直接写、用 ACK 兜正确性，流水更长但开销摊薄）

**价值**：这题答得出来的人很少。而且它跟你 ccl-lab 的结论**是同构的**——小消息付不起固定开销，所以要"步数少 + 同步激进"；大消息固定开销被摊薄，所以"流水长 + 每步搬得多"反而赢。

---

## 四、主线 4 · 传输层：同一个算法怎么落到不同后端 ★ 你的真实问题在这

**读什么**

```
src/transport/net_ib/        ← IB / RoCE：verbs 路径
  rg -n "ibv_post_send|ibv_poll_cq" src/transport/net_ib/ | head
  src/transport/net_ib/gdr.cc   ← GPUDirect RDMA（网卡直读显存）
src/transport/net_socket.cc  ← ★TCP 兜底：没有 RDMA 网卡时走这里
src/transport/shm.cc         ← 单机共享内存（同一台机器内的 rank 之间）
src/plugin/net/net_v*.cc     ← ★NET plugin 接口（多版本 ABI，net_v6 ~ net_v12）
```

**回答这三个问题（全都直接对着你的经历）**
1. **`net_ib` 和 `net_socket` 是怎么被选中的？** → 找 transport 注册和探测逻辑
2. **没有 RDMA 网卡时 NCCL 怎么办？** → **它退化到 socket（TCP）**。这一条直接回答你前面纠结的"拿不到网卡能不能做多机"——**能。**
3. **NET plugin 的接口长什么样？** → `src/plugin/net/net_v*.cc` 里的函数表（`ncclNet_t` 那套：`init / devices / listen / connect / regMr / isend / irecv / flush`）。**"给第三方/国产网卡做适配"就是实现这张表。**

**这一步读完，你就能说**：
> "通信库的传输层是**插件化**的：NCCL 通过 NET plugin 接口接不同网卡，IB 走 `net_ib`、没有 RDMA 就退化到 `net_socket`、单机走 `shm`。**换后端不动算法层**——我在自己那套原型里验证过同样的分层：`algo/` 只依赖 `transport` 的四个原语，换 transport 算法一行不改。"

**这句"我验证过"比"我知道"强十倍。而且这是真的。**

---

## 五、主线 5 · 拓扑感知与图搜索（为什么"选择表要按规模重标定"）

**读什么**

```
src/graph/topo.cc      ← 探测机器/网络拓扑
src/graph/search.cc    ← 搜索通信结构
src/graph/rings.cc     ← 怎么排 ring 的顺序（这是性能的关键之一）
src/graph/trees.cc     ← 怎么建树
src/graph/paths.cc     ← 路径选择（多路径）
```

**回答两个问题**
1. NCCL 怎么知道"哪张卡到哪张卡走 NVLink、哪条走 PCIe、哪条要过网"？
2. **为什么换个集群/换拓扑，最优算法就会变？**

**价值**：这一条 + 主线 2 合起来，就是你那句"**算法选择表必须按集群规模重新标定，不能抄**"的源码依据。

---

## 六、读完之后你能说什么（话术）

> "我通读过 NCCL 2.32 的主干：算法与协议的选择在 `tuning/`——`ring.cc` 和 `tree.cc` 各自给出代价估计，`tuning.cc` 用 `generalTable[algo][proto]` 比时间选最优，2.32 还是**代价模型 + 实测**的混合；协议层在 `device/prims_ll.h / prims_ll128.h / prims_simple.h`，分别对应小消息低延迟、折中、大消息吞吐；传输层是插件化的，IB 走 `net_ib`、没有 RDMA 就退化到 `net_socket`、单机走 `shm`，对外是 NET plugin 的接口表。
> **而我之所以能快速看懂这套结构，是因为我在自己的集合通信原型里做过同样的分层**——四种 AllReduce 算法 + 可替换的传输后端，用黄金模型对账误差 0，并且量化出算法选择在什么规模和什么 RTT 下会翻转。"

---

## 七、你的实验 ↔ NCCL 源码 对照表（讲的时候直接摆这个）

| 你在 ccl-lab 里验证的 | NCCL 里对应的实现 |
|---|---|
| 算法选择随消息大小切换 | `src/tuning/tuning.cc` 的 `generalTable[algo][proto]` |
| 小消息由**步数**决定 | `src/device/prims_ll.h`（低延迟协议）+ Tree 的 O(log N) 步 |
| 大消息由**每步搬运量**决定 | `src/device/prims_simple.h`（长流水）+ Ring 的 `2(N-1)/N` |
| **换 transport 不改算法** | `src/transport/` 多后端 + `src/plugin/net/` 插件接口 |
| 交叉点**随规模右移** | `src/graph/` 拓扑感知 + `src/tuning/cost_model.cc` |
| 长 RTT 下算法翻转 | 这是 NCCL 数据中心假设的失效场景——**广域**要自己重标定 |

**这张表的作用**：证明你的实验不是玩具，**它和工业实现是同构的**。你不是"看了看论文"，你是"从下往上搭过一遍，再回头看工业实现"。

---

## 八、动手建议（今天就能开始）

```bash
cd refs/nccl

# 主线 2 一个下午能过完
rg -n "ncclTuningCompute|bestTuning.algo|generalTable\[" src/tuning/tuning.cc
rg -n "NCCL_ALGO_RING|NCCL_ALGO_TREE|NCCL_PROTO_LL" src/tuning/*.cc | head -30

# 主线 4 半小时
rg -n "ncclNet_t|\.isend|\.irecv|\.regMr" src/plugin/net/net_v12.cc | head -20
rg -n "ibv_post_send|ibv_poll_cq" src/transport/net_ib/ | head -10
rg -n "socket|connect|listen" src/transport/net_socket.cc | head -20
```

**读完的顺序建议**：主线 4（最贴你的经历）→ 主线 2（最贴考点）→ 主线 3 → 主线 1 → 主线 5。

**别通读**——780 行的 `all_reduce.h` 不用逐行看，那是 device 端实现细节，不会挖到那一层。
