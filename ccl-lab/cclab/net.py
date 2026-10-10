"""真 TCP socket 传输 —— 把"跨节点"这件事在没有 root、没有 RDMA 网卡的条件下做出来。

为什么要有它（而不是复用 SocketTransport）：
  SocketTransport 走的是 multiprocessing 的 Pipe（AF_UNIX），那只能证明"跨进程"。
  本文件走**真正的 TCP/IP socket + 真实端口**，并且允许给每个 rank 绑定不同的
  回环地址（127.0.0.2 / 127.0.0.3 …），于是**一台机器上就能模拟出多个节点**：
      node 0 = 127.0.0.2，node 1 = 127.0.0.3 …
  把 host 换成真实 IP，同一份代码就是跨机 —— 算法层一行都不用改。

再叠一层 RTT 注入（--rtt-ms）：在接收侧给每条消息加半个 RTT 的单程延迟。
  于是"控制环延迟 = 一个 RTT"这件事可以被直接量化：
  ring 每步的 send→recv 是半 RTT，一收一发合成一个 RTT —— 正是长 RTT 下
  DCQCN 类闭环控制退化的根源（鹏城那条线的同一套模型）。

边界（不许夸大）：
  * 这是 TCP，**不是 RoCE/IB**。没有 RDMA 语义、没有 PFC/ECN、没有单边操作。
  * RTT 是应用层 sleep 注入的，不经过真实链路，**测的是协议行为不是链路性能**。
  * "多节点"是回环地址模拟出来的，**不是真多机**。
"""

from __future__ import annotations

import queue
import socket
import struct
import threading
import time

from .transport import Transport

_HDR = struct.Struct("!I")


class TcpTransport(Transport):
    """跨（模拟）节点的 TCP 传输。

    握手规则（避免双向竞争）：
        rank i 主动 connect 所有 rank j > i；
        rank i 被动 accept 所有 rank j < i 的连接。
    连上之后，发送方先发 4 字节自己的 rank 号，接收方据此建索引。
    """

    def __init__(self, rank, nranks, max_count, buf, addrs, rtt_ms=0.0, bar_conn=None):
        super().__init__(rank, nranks, max_count, buf)
        self._addrs = addrs              # [(host, port), ...] 长度 nranks
        self._bar = bar_conn
        self._rtt = rtt_ms / 1000.0 / 2.0        # 单程 = 半个 RTT
        self._inbox = {p: queue.Queue() for p in range(nranks) if p != rank}
        self._out = {}                   # peer -> socket（我主动连的）
        self._ready = {}                 # peer -> True，握手完成
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()

        # 1) 监听自己的地址
        host, port = addrs[rank]
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(max(1, nranks))

        self._acc = threading.Thread(target=self._accept_loop, daemon=True)
        self._acc.start()

        # 2) 主动连接所有比我大的 rank。
        #    注意：连出去的 socket 同样要有接收线程 —— 否则对方发回来的数据没人收，
        #    双向都会卡死（第一版就是漏了这一步）。
        for j in range(rank + 1, nranks):
            sock = self._connect(j)
            self._out[j] = sock
            self._ready[j] = True
            threading.Thread(target=self._recv_loop, args=(sock, j), daemon=True).start()

        # 3) 等所有入站连接到齐（保证建链完成再开跑），带超时防止静默挂死
        deadline = time.time() + 60
        while any(p not in self._ready for p in self._inbox):
            if time.time() > deadline:
                missing = [p for p in self._inbox if p not in self._ready]
                raise RuntimeError(f"rank{self.rank}: 等待入站连接超时，缺 {missing} "
                                   f"（地址表 {self._addrs}）")
            time.sleep(0.002)

        self._sen = threading.Thread(target=self._send_loop, daemon=True)
        self._sen.start()

    # ------------------------------------------------------------ 建链
    def _connect(self, j):
        host, port = self._addrs[j]
        deadline = time.time() + 30
        while True:
            try:
                s = socket.create_connection((host, port), timeout=5)
                break
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError(f"rank{self.rank}: 连接 rank{j} ({host}:{port}) 超时")
                time.sleep(0.05)
        s.settimeout(None)                    # 清掉 connect 的超时，否则后续收发会超时
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(_HDR.pack(self.rank))
        return s

    def _accept_loop(self):
        need = set(p for p in self._inbox)
        while need:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            raw = self._recv_exact(conn, 4)
            if raw is None:
                conn.close()
                continue
            peer = struct.unpack("!I", raw)[0]
            need.discard(peer)
            # ★ TCP 连接是全双工的：accept 进来的这条同样可以往对端写。
            #   第一版只把 connect 出去的登记成出边，结果 rank N-1 一条出边都没有
            #   （KeyError: 0），大 rank 根本发不出数据。
            self._out[peer] = conn
            self._ready[peer] = True
            threading.Thread(target=self._recv_loop, args=(conn, peer),
                             daemon=True).start()

    @staticmethod
    def _recv_exact(sock, n):
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except OSError:
                return None                  # close() 期间的正常中断
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    # ------------------------------------------------------------ 收发
    def _recv_loop(self, conn, peer):
        while not self._stop.is_set():
            hdr = self._recv_exact(conn, 4)
            if hdr is None:
                return
            nbytes = struct.unpack("!I", hdr)[0]
            payload = self._recv_exact(conn, nbytes) if nbytes else b""
            if payload is None:
                return
            if self._rtt > 0:
                time.sleep(self._rtt)          # 单程延迟注入（半 RTT）
            self._inbox[peer].put((nbytes, payload))

    def _send_loop(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            dst, n, payload = item
            try:
                # ★ header 里必须是**字节数**。第一版写的是元素数 n，
                #   接收方按字节读，于是 mv[i] 越界（IndexError: dimension 1）。
                self._out[dst].sendall(_HDR.pack(len(payload)) + payload)
            except (OSError, KeyError, AttributeError):
                return

    def send(self, dst, src_off, n):
        payload = bytes(self.data()[src_off: src_off + n])
        self._q.put((dst, n, payload))
        self.bytes_moved += n * 4
        self.msgs += 1

    def _take(self, src, n):
        nbytes, payload = self._inbox[src].get()
        if nbytes != n * 4:
            raise ValueError(f"rank{self.rank}: 来自 {src} 的消息长度不符 "
                             f"{nbytes} 字节 != {n * 4} 字节（{n} 个 float32）")
        return memoryview(payload).cast("f")

    def recv_add(self, src, n, data_off=0):
        mv = self._take(src, n)
        d = self.data()
        for i in range(n):
            d[data_off + i] += mv[i]

    def recv_copy(self, src, n, data_off=0):
        mv = self._take(src, n)
        self.data()[data_off: data_off + n] = mv[:n]

    def barrier(self):
        if self._bar is None:
            return
        self._bar.send(b"B")
        self._bar.recv()

    def close(self):
        self._stop.set()
        self._q.put(None)
        for s in list(self._out.values()):
            try:
                s.close()
            except OSError:
                pass
        try:
            self._srv.close()
        except OSError:
            pass
