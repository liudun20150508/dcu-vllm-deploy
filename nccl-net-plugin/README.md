# nccl-net-plugin —— NCCL 网卡适配插件骨架（C++17）

> **这份代码回答一个问题：要给一张 NCCL 还不支持的网卡（自研 / 国产）做适配，具体要写什么？**
>
> 答案是：**不改 NCCL 核心，实现一张函数表，编成一个 `.so`。**

## 一句话理解它

```
NCCL 核心  ──dlopen/dlsym──▶  libnccl-net.so  ──▶  你的网卡
                                  ↑
                       你在这里写代码（22 个函数）
```

NCCL 用 `dlopen` 加载插件、用 `dlsym` 找两个符号：

| 符号 | 必需 | 作用 |
|---|---|---|
| `ncclNetPlugin_v12` | ✅ | 点对点传输（本文件导出的） |
| `ncclCollNetPlugin_v12` | 可选 | 集合通信卸载（SHARP 那类） |

（见 NCCL 源码 `src/plugin/net/net_v12.cc` 的 `getNcclNet_v12`）

## 22 个函数分别在干什么

| 分组 | 函数 | 职责 | 不实现会怎样 |
|---|---|---|---|
| **生命周期** | `init` | 拿日志句柄、开设备、建线程池 | 起不来 |
| | `devices` | 报"我有几张网卡" | 起不来 |
| | **`getProperties`** | **声明能力**（见下） | 起不来 |
| **建链** | `listen` / `connect` / `accept` | 交换建连信息（RDMA 是 QPN+GID+rkey，TCP 是端口） | 通信建不起来 |
| | `closeSend` / `closeRecv` / `closeListen` | 与上一步一一配对释放 | 泄漏资源 |
| **内存** | `regMr` | 注册内存（RDMA 里就是 `ibv_reg_mr`） | **没有内核旁路** |
| | `regMrDmaBuf` | 容器/DMA-BUF 场景的注册 | 可报不支持，NCCL 会回退 |
| | `deregMr` | 注销 | 泄漏 |
| **传输** | `isend` / `irecv` | **异步**提交，返回 request | 传不了数据 |
| | `test` | NCCL 轮询 request 的完成状态 | 传不了数据 |
| | `iflush` | 有网卡侧卸载时才需要 | 纯软件路径可空实现 |
| **卸载（可选）** | `getDeviceMr` / `irecvConsumed` | 网卡侧内存相关 | 返回空即可 |
| | `makeVDevice` | 多卡聚合成虚拟设备 | 可报不支持 |
| **其他** | `finalize` / `setNetAttr` | 收尾 / 接收 NCCL 的 op-algo-proto 信息 | 可空实现 |

## ★ 最关键的一个函数：`getProperties`

它不是"报个名字"就完了——**它是插件向 NCCL 声明能力的地方，NCCL 后面所有决策都基于它**：

| 字段 | 影响什么 |
|---|---|
| **`ptrSupport`** | **这块网卡能直接访问哪种内存**：`NCCL_PTR_HOST` / `NCCL_PTR_CUDA` / `NCCL_PTR_DMABUF`。**不报 `NCCL_PTR_CUDA` 就没有 GPUDirect RDMA**，NCCL 只能走"显存→主机内存→网卡"的绕路，性能差一个量级 |
| `speed` / `latency` | NCCL 用它算带宽预算、**选算法** |
| `maxComms` / `maxRecvs` | 能建多少连接、能分组收几段 |
| `railId` / `planeId` | 多 rail / 多平面网络里的位置（智算网络常见） |
| `netDeviceType` | 有没有网卡侧卸载 |

**所以适配一张新卡的第一件事，就是搞清楚它能报什么 `ptrSupport`。**

## 编译与加载

```bash
# 需要 NCCL 源码里的头文件（src/include/plugin/nccl_net.h）
NCCL_SRC=/path/to/nccl
g++ -std=c++17 -O2 -fPIC -shared \
    -I${NCCL_SRC}/src/include -I${NCCL_SRC}/src/include/plugin \
    mycard_net.cpp -o libnccl-net.so -lpthread

# 让 NCCL 用它
export NCCL_NET_PLUGIN=./libnccl-net.so
export NCCL_DEBUG=INFO            # 加载成功会打：NET/Plugin: Loaded net plugin MyCard (v12)
```

> 真跑还需要 GPU + NCCL。**这份骨架的目的是说清接口，不是提供可用的网卡驱动。**

## 这份代码里标注的简化（`[简化]`）

- 只用 TCP 做传输（真实网卡走 verbs / 厂商 SDK）
- `ptrSupport` 只报 `NCCL_PTR_HOST`（真卡应报 `| NCCL_PTR_CUDA`）
- `isend`/`irecv` 每次起线程（生产用线程池 + 环形缓冲）
- request 的回收没做完（生产要引用计数或由 close 统一回收）
- 没实现 `regMrDmaBuf` / `makeVDevice`（报不支持）

## 可以直接讲

> "给一张 NCCL 不支持的网卡做适配，标准做法**不是改 NCCL 核心**，而是实现 NET plugin 的接口表，编成 `libnccl-net.so`，NCCL 用 `dlopen` + `dlsym` 加载它。
> 那张表大概 22 个函数，分四块：生命周期、建链、内存注册、异步传输。**最关键的是 `getProperties`——它不是报名字，是声明能力**：`ptrSupport` 决定这块卡能不能直接访问显存，**那一行就是 GPUDirect RDMA 的开关**；`speed` 和 `latency` 会被 NCCL 拿去算带宽预算、选算法。
> **接口的设计意图很清楚：让网卡成为可替换的部件。** 我在自己那套集合通信原型里验证过同一件事——换 transport 后端，算法层一行没改。"

## 边界

- 这只是**接口骨架 + 教学实现**，不是可用的网卡驱动
- 真实适配还要处理：流控、重传、多 QP、乱序、错误注入、性能调优
- 集合通信卸载（`ncclCollNetPlugin_v12`，对应 SHARP）这里没做
