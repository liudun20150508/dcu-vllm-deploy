"""传输抽象层 —— 算法层只依赖这里的接口，不关心底下是共享内存还是 socket。

这一层的存在本身就是论点：**同一套集合通信算法，换一个 transport 后端就能跑，
这就是 UCX 的 UCT transport 抽象 / NCCL 的 NET plugin（libnccl-net.so）
在做的事** —— 也是"给国产/第三方网卡做通信库适配"这份活的结构。

两个实现：

  ShmTransport     单边写共享内存。发送方直接把数据 memmove 进对端的 buffer，
                   接收端只等一条控制消息 —— 零拷贝、无内核参与。
                   ≈ RDMA WRITE 的语义（单边、不惊动对端 CPU）。
                   代价：**必须自己做流控**（见下）。

  SocketTransport  双边 socket 字节流。发送方先拷成 bytes 交给后台发送线程，
                   接收方收全了再落到 data —— 两次拷贝 + 内核协议栈。
                   ≈ TCP socket / 没有 RDMA 时的兜底路径。
                   好处：send 时数据已被拷走，天然不需要流控。

------------------------------------------------------------
关于流控（这是本实验最重要的一个发现，别跳过）

第一版 ShmTransport 没有流控，跑 ring 直接算错（err ~3.0）。
根因：单边写只在**发送方**手里有数据，接收方还没消费，
发送方就已经进入下一步把同一块 scratch 覆盖掉了 —— 一个纯粹的竞态。

这跟真实网卡上的问题是同一个：**零拷贝单边传输必须配 credit**，
否则接收缓冲随时可能被覆盖。RDMA 里的对应物就是 receive WQE 队列
（pre-posted buffer）和 QP 的 credit。

本实现在这里显式建模了它。两层保护缺一不可：

  ① **每个 (src -> dst) 方向一块独立的接收 slot**，而不是全 rank 共用一块 scratch。
     只加 credit 是不够的：tree 里 rank0 会依次从 rank1/rank2/rank4 收数据，
     它们共用一块 scratch 就会互相覆盖，而且会死锁（后到的 src 占着 buffer，
     却没人在等它，先到的 src 永远拿不到 credit）。
     真实网卡的做法就是**接收方按 QP 预先 post 独立的接收 buffer** —— 这里照做。
  ② **每方向初始 1 个 credit**：send 前先取 credit，取不到就等对端的 ack；
     recv_add / recv_copy 在**消费完之后**才回 ack。
     等价于"接收方预先 post 了一个 buffer"。

两个一起才成立：slot 隔离解决"不同 src 互相覆盖"，
credit 解决"同一个 src 的前后两条消息互相覆盖"。

顺带一个对照结论：SocketTransport 因为 send 时已经拷贝，天然免疫这两个问题。
**这就是零拷贝换来的性能要用协议复杂度去还。**

顺带一个对照结论：SocketTransport 因为 send 时已经拷贝，天然免疫这个问题。
**这就是零拷贝换来的性能要用协议复杂度去还。**
------------------------------------------------------------

buffer 布局（max_count 为本轮最大元素数）：
    每个 rank 占 (1 + nranks) * max_count 个 float32
        [0, count)                             data
        [count + peer*count, ...)              peer 专属接收 slot（每个 peer 一块）
    所有 rank 的段拼在一块共享内存里。
"""

from __future__ import annotations

import queue
import threading


class Transport:
    """算法层唯一依赖的接口。send + 两个 recv + barrier。"""

    def __init__(self, rank: int, nranks: int, max_count: int, buf):
        self.rank = rank
        self.nranks = nranks
        self.max_count = max_count
        self.count = max_count          # 当前轮次的元素数，由 run 循环改写
        self.buf = buf
        self.bytes_moved = 0            # 单 rank 视角的累计搬运字节数
        self.msgs = 0                   # 累计消息条数
        self._mv = memoryview(buf).cast("f")

    # ---------- 视图 ----------
    def _seg(self, rank: int) -> int:
        return rank * (self.nranks + 1) * self.max_count

    def data(self):
        """本 rank 的数据区视图（count 个 float32）。"""
        b = self._seg(self.rank)
        return self._mv[b: b + self.count]

    def slot(self, owner: int, peer: int):
        """owner 的段里，专门留给 `peer` 的接收 slot（count 个 float32）。

        每个方向一块，互不覆盖 —— 这是 pre-posted receive buffer 的直接建模。
        """
        b = self._seg(owner) + self.count + peer * self.count
        return self._mv[b: b + self.count]

    # ---------- 原语 ----------
    def send(self, dst: int, src_off: int, n: int) -> None:
        """把我的 data[src_off : src_off+n] 送到 dst 的 scratch[0:n]。"""
        raise NotImplementedError

    def recv_add(self, src: int, n: int, data_off: int = 0) -> None:
        """收 src 的 n 个元素，累加到我的 data[data_off : data_off+n]。"""
        raise NotImplementedError

    def recv_copy(self, src: int, n: int, data_off: int = 0) -> None:
        """收 src 的 n 个元素，写进我的 data[data_off : data_off+n]（覆盖）。"""
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError

    def reset_stats(self) -> None:
        self.bytes_moved = 0
        self.msgs = 0

    def close(self) -> None:
        pass


class ShmTransport(Transport):
    """单边写共享内存：零拷贝 + 显式 credit 流控。"""

    def __init__(self, rank, nranks, max_count, buf, conns, ack_conns, bar_conn):
        super().__init__(rank, nranks, max_count, buf)
        self._conns = conns          # conns[peer]：数据到达通知（duplex）
        self._acks = ack_conns       # acks[peer]：credit 回收（我 → peer 方向）
        self._bar = bar_conn
        self._credit = {p: 1 for p in range(nranks) if p != rank}

    def _acquire_credit(self, dst: int) -> None:
        while self._credit[dst] <= 0:
            self._acks[dst].recv()        # 等对端消费完，回收一个 credit
            self._credit[dst] += 1
        self._credit[dst] -= 1

    def send(self, dst: int, src_off: int, n: int) -> None:
        self._acquire_credit(dst)
        self.slot(dst, self.rank)[:n] = self.data()[src_off: src_off + n]   # 单边写
        self._conns[dst].send((self.rank, n))                               # 到达通知
        self.bytes_moved += n * 4
        self.msgs += 1

    def _wait_data(self, src: int, n: int) -> None:
        who, cnt = self._conns[src].recv()
        if who != src or cnt != n:
            raise ValueError(f"rank{self.rank}: 来自 {src} 的消息不匹配 ({who}, {cnt}) != ({src}, {n})")

    def recv_add(self, src: int, n: int, data_off: int = 0) -> None:
        self._wait_data(src, n)
        d, s = self.data(), self.slot(self.rank, src)
        for i in range(n):
            d[data_off + i] += s[i]
        self._acks[src].send(b"A")      # 消费完才释放 credit

    def recv_copy(self, src: int, n: int, data_off: int = 0) -> None:
        self._wait_data(src, n)
        self.data()[data_off: data_off + n] = self.slot(self.rank, src)[:n]
        self._acks[src].send(b"A")

    def barrier(self) -> None:
        self._bar.send(b"B")
        self._bar.recv()


class SocketTransport(Transport):
    """双边 socket 字节流：两次拷贝 + 内核协议栈，不需要 credit。"""

    def __init__(self, rank, nranks, max_count, buf, conns, ack_conns, bar_conn):
        super().__init__(rank, nranks, max_count, buf)
        self._conns = conns
        self._bar = bar_conn
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._thread.start()

    def _sender_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            dst, n, payload = item
            self._conns[dst].send((self.rank, n, payload))

    def send(self, dst: int, src_off: int, n: int) -> None:
        payload = bytes(self.data()[src_off: src_off + n])   # 拷贝 1：发之前就拷出来
        self._q.put((dst, n, payload))
        self.bytes_moved += n * 4
        self.msgs += 1

    def _recv_payload(self, src: int, n: int) -> memoryview:
        who, cnt, payload = self._conns[src].recv()
        if who != src or cnt != n:
            raise ValueError(f"rank{self.rank}: 来自 {src} 的消息不匹配 ({who}, {cnt}) != ({src}, {n})")
        return memoryview(payload).cast("f")

    def recv_add(self, src: int, n: int, data_off: int = 0) -> None:
        mv = self._recv_payload(src, n)
        d = self.data()
        for i in range(n):
            d[data_off + i] += mv[i]

    def recv_copy(self, src: int, n: int, data_off: int = 0) -> None:
        mv = self._recv_payload(src, n)
        self.data()[data_off: data_off + n] = mv[:n]

    def barrier(self) -> None:
        self._bar.send(b"B")
        self._bar.recv()

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=2.0)


TRANSPORTS = {
    "shm": ShmTransport,
    "socket": SocketTransport,
}
