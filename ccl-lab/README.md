# ccl-lab · 集合通信库原型实验台

> 为**中科曙光「通信库性能优化工程师（智算网络方向）」**准备的一套可跑实验。
> 零第三方依赖（Python 3.10+ 标准库），在 macOS / Linux 上都能直接跑，不需要 RDMA 网卡、不需要 GPU。

---

## 为什么做这个

那份 JD 要的是**通信库源码理解 + 集合通信算法 + 性能测试与分析体系**，不要 RTL、不要 FPGA。
所以这个实验直接打 JD 的正面，而不是拿硬件背景去硬凑：

| JD 原话 | 本实验对应 |
|---|---|
| 参与 NCCL、RCCL、MPI、UCX、UCC 等通信库的**源码分析、功能开发**和性能调优 | 自己实现 4 种 AllReduce 算法 + 传输抽象层，而不是调 API |
| 针对 **AllReduce、AllGather、ReduceScatter、AllToAll**、点对点通信等场景开展优化 | **四个集合操作全部实现并通过对账**（AllReduce 四种算法 + 另外三个操作各一种） |
| 结合 **RDMA、InfiniBand、RoCE** 等高速网络特性，优化通信路径、数据传输、集合通信算法及网络资源利用效率 | 用可替换的 transport（零拷贝单边写 vs 双边 socket）建模"换网络后端"，并量化差距 |
| 建立**通信性能测试与分析体系**，开发性能测试、数据分析、问题定位及**自动化调优工具** | `run.py` 扫矩阵出 CSV、`plots.py` 出曲线与算法选择表，一条命令跑完 |
| 解决大规模集群中的**性能波动、扩展性**问题 | N=2/4/8 的扩展性对照；`ratio`（理论搬运量）与实际耗时的对账 |
| 从应用、通信库、操作系统、网络及**硬件**多个层面分析性能问题 | 硬件层是已有的 RoCEv2 数据面原型（BTH/ICRC/DCQCN），这里补的是它上面那几层 |

---

## 快速开始

```bash
cd ccl-lab

# 1) 正确性对账（快，几十秒内）
python3 -m cclab.run check --nranks 2,4,8

# 2) 全量扫描 + 出 CSV
python3 -m cclab.run bench --csv results/raw.csv

# 3) 出图 + 算法选择表
python3 -m cclab.plots results/raw.csv results

# 4) 跨节点（TCP）+ 注入长 RTT
python3 -m cclab.run bench --nranks 8 --transports tcp --nodes 2 --rtt-ms 0 --csv results/tcp_rtt0.csv
python3 -m cclab.run bench --nranks 8 --transports tcp --nodes 2 --rtt-ms 5 --csv results/tcp_rtt5.csv
python3 -m cclab.plots results/raw.csv results/tcp_rtt0.csv results/tcp_rtt5.csv results
```

三种 transport：

| `--transports` | 是什么 | 证明了什么 | 需要什么 |
|---|---|---|---|
| `shm` | 单边写共享内存 + credit 流控 | 零拷贝路径（≈ RDMA WRITE 语义） | 无 |
| `socket` | multiprocessing Pipe（AF_UNIX） | 跨进程拷贝路径 | 无 |
| `tcp` | **真 TCP socket + 真端口 + 可注入 RTT** | **跨节点**通信、长 RTT 下的算法翻转 | 无（`--nodes` 模拟多节点） |

预期看到：

```
ALL MATCH —— 四个算法 × 两种 transport × 三档规模，结果均与黄金模型一致
```

常见参数：

```bash
--nranks 2,4,8                          # 规模（tree/recdouble/rabenseifner 要求 2 的幂）
--transports shm,socket                 # 零拷贝单边写 / 双边 socket
--algos ring,tree,recdouble,rabenseifner
--counts 64,256,1024,4096,16384,65536   # 每个 rank 的 float32 元素数（须能被 nranks 整除）
--repeats 3                             # 每档取 min，压噪声
```

---

## 结构

```
ccl-lab/
├── cclab/
│   ├── transport.py   传输抽象层  ≈ UCX UCT / NCCL NET plugin（libnccl-net.so）
│   │                  ShmTransport  零拷贝单边写 + per-direction slot + credit 流控
│   │                  SocketTransport 双边进程间 socket，两次拷贝，天然无需流控
│   ├── net.py         TcpTransport  真 TCP + 背压，可模拟多节点、可注入 RTT
│   ├── algos.py       四个集合操作：AllReduce（4 种算法）+ AllGather / ReduceScatter / AllToAll（Ring）
│   ├── golden.py      黄金模型：闭式解，独立于任何算法路径
│   ├── worker.py      rank 进程
│   ├── run.py         实验编排（多进程 + BSP barrier）+ 性能测试与分析
│   └── plots.py       手写 SVG（无 matplotlib 依赖）+ algorithm selection 表
└── results/
    ├── raw.csv            shm/socket 的原始记录
    ├── tcp_rtt{0,5}.csv   跨节点 + 长 RTT 对照
    ├── curve_*.svg        曲线图（含 tcp-rtt0ms / tcp-rtt5ms）
    └── FINDINGS.md        ⭐ 结论、三个真 bug 的定位过程、边界
```

**分层是刻意的**，它就是 UCX / NCCL 的分法：

```
algos.py       算法层（Ring / Tree / Rabenseifner）  ≈ NCCL 的 algorithm
transport.py   传输层（零拷贝 / 拷贝，可替换）        ≈ UCX UCT / NCCL transport+net plugin
run.py         测试与分析层                          ≈ nccl-tests + 自建 tuning 表
```

换 transport 不改算法一行代码 —— 这就是"给国产/第三方网卡做适配"这份活的结构。

---

## 结论速览

详见 **[`results/FINDINGS.md`](results/FINDINGS.md)**。三句话：

1. **小消息由步数决定**：N=8 时 recdouble 3 步 vs ring 14 步，0.2 KB 上差 **6 倍**。
2. **大消息由每步搬运量决定**：tree/recdouble 的 `ratio` 是 6.0 / 3.0，ring 和 Rabenseifner 只有 1.75 —— 256 KB 上 ring 比 tree **快 2.4 倍**。
3. **交叉点随 N 右移**（N=2 约 1–4 KB，N=8 约 4–16 KB）→ **算法选择表必须按集群规模重新标定，不能抄。**
4. **长 RTT 下算法选择会翻转**：注入 5 ms RTT 后，**所有消息大小的胜者都变成 recdouble**（3 步），ring 在 0.2 KB 上从 1.27 ms 涨到 51.08 ms（**+49.8 ms**）—— RTT 微秒级时看带宽，RTT 几十毫秒时**看步数**。
5. **四个集合操作全覆盖**（AllReduce / AllGather / ReduceScatter / AllToAll），252 组配置误差 0。其中 AllGather 与 ReduceScatter 步数、搬运量完全相同但**耗时差 17 倍** —— 因为 ReduceScatter 要归约：**AllReduce 与 AllGather 的差异根源是"要不要算"，不是"要不要传"。**

还有一段值得单独讲：**第一版零拷贝 transport 算错了**，根因是单边写缺 credit 流控；
补上"per-direction 独立 slot + 消费完才回 ack"才正确 —— 这跟真实网卡的 pre-posted receive WQE 是同一个约束。定位过程写在 FINDINGS 第三节。

---

## 边界（别越过这条线说）

- 本机 **没有 RDMA 网卡、没有 GPU**。测的是算法步数、消息数、搬运量与调度开销，**不是真实 RoCE/IB 带宽**。
- `tcp` transport 是**真 TCP**，但**不是 RoCE**；RTT 是应用层 sleep 注入，不经过真实链路；"多节点"是回环地址/端口模拟的，**不是真多机**。
- 绝对交叉点不可移植（Python 解释器的每跳开销比真实网卡大两个数量级）。可移植的是"小消息看步数、大消息看搬运量"这个结构。
- 实现的是**算法层 + 传输抽象层**；QP/WQ-CQ/MR/PCIe-DMA/内核旁路这些硬件语义不在里面。
- **没有跑真实多机多卡集群** —— 这是缺口，不要包装成成果。
