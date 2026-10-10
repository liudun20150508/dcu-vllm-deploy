# rdma-demo · RDMA Verbs 最小实测（C++17）

> **为什么写这个**：JD 任职 3 要 C/C++，优先条件 2 要"**RDMA Verbs**、GPUDirect RDMA、InfiniBand 或 RoCE **实际开发、调试**经验"。
> 这份代码同时补这两块——而且**用 SoftRoCE，不需要任何 RDMA 硬件**。

## 它演示什么

| 文件 | 操作 | 关键点 |
|---|---|---|
| `send_recv.cpp` | **SEND / RECV**（双边） | 对端**必须提前** `post_recv`，否则 RNR 重试 —— 双边要惊动远端 CPU |
| `write_read.cpp` | **WRITE / READ**（单边） | 对端**全程不知情**；WRITE 是"推"、READ 是"拉"，**READ 多一个 RTT** |

`write_read.cpp` 跑完你会亲眼看到：**server 进程一行数据搬运都没做，它那块内存的内容却被 client 改掉了。** 这就是"单边操作"和"内核旁路"的实际含义。

## 环境准备（SoftRoCE，5 分钟）

SoftRoCE（内核模块 `rdma_rxe`）是 Linux 自带的**软件 RDMA 实现**——它给你一个真实的 `/dev/infiniband/uverbs0`、真实的 verbs API、真实的 QP/MR/CQ。**不需要物理网卡。**

```bash
sudo apt update && sudo apt install -y rdma-core ibverbs-utils perftest build-essential

sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev eth0     # eth0 换成你的网卡

ibv_devices          # ← 应该看到 rxe0
ibv_devinfo -d rxe0  # ← PORT_ACTIVE / GID / max_qp
```

> 顺手也把官方基准跑一遍，能拿到真实数字：
> `ib_send_bw -d rxe0`（server） / `ib_send_bw -d rxe0 127.0.0.1`（client）
> 同还有 `ib_write_bw`、`ib_send_lat`。

## 跑

```bash
make
# 终端 1
./send_recv server
# 终端 2
./send_recv client 127.0.0.1

# 换单边那组
./write_read server
./write_read client 127.0.0.1
```

第三个参数可以指定设备名（默认 `rxe0`）。

## 预期输出

**`send_recv`**

```
[server] posted RECV, waiting for SEND...
[server] received 64 bytes: "HELLO-FROM-CLIENT"
[server] sent reply
```
```
[client] sent 64 bytes
[client] got reply: "ACK-FROM-SERVER"
```

**`write_read`（重点看这两行）**

```
[server] 我的 buf 初始内容: "SERVER-BUFFER"
[server] 我不参与任何数据搬运，等对端完工信令...
[server] 对方干完了。我的 buf 现在是: "CLIENT-WROTE-HERE"
[server] ↑ 注意：这个变化是对方 WRITE 进来的，我的 CPU 全程没有碰过这段内存
```
```
[client] READ 到对端内存: "SERVER-BUFFER"
[client] WRITE 完成：对端那段内存已被我覆盖
[client] 再 READ 一次确认: "CLIENT-WROTE-HERE"
```

## 用 `tc netem` 把 READ 和 WRITE 的差距放大

SoftRoCE 走内核网络栈，所以 `tc` 对它有效：

```bash
sudo tc qdisc add dev eth0 root netem delay 5ms
./write_read server & ./write_read client 127.0.0.1
sudo tc qdisc del dev eth0 root
```

加延迟后观察：**WRITE 的首字节几乎不受影响（推过去就完），READ 会明显多出往返延迟**——因为 READ 必须"请求先到、数据才回"。这就是为什么 READ 的首字节延迟比 WRITE 多一个 RTT。

## 代码里的三处关键（问答会被问）

1. **QP 状态机四步**（`common.cpp` 的 `to_init / to_rtr / to_rts`）
   `RESET → INIT → RTR → RTS`。RTR 时需要**对端**的 QPN 和 GID，所以数据面之前必须先跑控制面（TCP 换 `ConnInfo`）。

2. **MR 内存注册**（构造函数第 3 步）
   `ibv_reg_mr` 把页 pin 住并生成 `lkey/rkey`。**没有 MR 就没有内核旁路**——网卡每次 DMA 都要回内核查页表。
   `lkey` 给自己用，`rkey` 给对端用（单边操作）。

3. **`send_flags = IBV_SEND_SIGNALED`**
   带上它才有 CQE，才有 `ibv_poll_cq` 的完成通知。不带就不会产生 CQE（省开销）。

## 问答可以直接讲的一段

> "为了搞清楚 verbs 这一层，我在 **SoftRoCE**（内核里的软件 RDMA 实现）上把四个操作都写了一遍：SEND/RECV 要双方配对，对端必须提前 post 接收缓冲；WRITE 是推、READ 是拉，**READ 的首字节延迟多一个 RTT**——用 `tc netem` 加 5 ms 延迟能明显看出来。
> 实现上最要紧的是三件事：**QP 状态机要走完 RESET→INIT→RTR→RTS**（RTR 要拿对端的 QPN 和 GID，所以必须先有控制面）；**MR 注册决定能不能内核旁路**（lkey 自己用、rkey 给对端）；**完成通知靠带 SIGNALED 的 WR 产生 CQE**。
> 我是先用软件实现把整条链走通，再回头看 NCCL 的 transport 层——**它把 verbs 包在 NET plugin 的接口表后面，换网卡就换那张表。**"

## 边界（别越线）

- **SoftRoCE 是软件实现**：功能是真的，**性能数字不能代表真实硬件**（没有硬件卸载、没有真正的零拷贝路径）
- 这份代码是**单 QP、单 CQ、无重传处理**的最小示例，不是生产级实现
- 没做 **GPUDirect RDMA**（那需要 GPU + 驱动的 peer memory 支持）
