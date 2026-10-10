"""rank 进程：填充输入 → 跑算法 → 用黄金模型对账 → 上报统计。

错误处理有个坑值得记下来：

  worker 一旦崩掉，主进程就会一直等 result_q，直到超时。而崩溃本身往往发生在
  **收尾阶段**（比如 shm.close() 报 BufferError，因为 Transport 还持有 memoryview），
  于是真正的原因（建链失败）被二次异常盖掉；又因为 multiprocessing.Queue 是异步的、
  没 flush 出去，最后表现成"什么都不打印，就是卡住"。

所以这里的分工是：
  * 任何异常都收敛成 (ok, why)，走 done 消息上报；
  * 上报后 close() + join_thread() 强制 flush，确保主进程一定收得到；
  * shm.close() 一律包 try/except，绝不让收尾异常掩盖真因。
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory

from . import golden
from .algos import ALGOS, op_of
from .net import TcpTransport
from .transport import TRANSPORTS


def worker_main(rank, nranks, max_count, transport_name, conns, ack_conns, bar_conn,
                shm_name, result_q, tasks, addrs=None, rtt_ms=0.0):
    shm = shared_memory.SharedMemory(name=shm_name)
    t = None
    ok, why = True, ""

    try:
        if transport_name == "tcp":
            t = TcpTransport(rank, nranks, max_count, shm.buf, addrs, rtt_ms, bar_conn)
        else:
            t = TRANSPORTS[transport_name](rank, nranks, max_count, shm.buf,
                                           conns, ack_conns, bar_conn)

        for algo, count, rep in tasks:
            t.count = count
            golden.fill_data(t.data(), rank)     # 输入：与 rank 和位置都相关
            t.reset_stats()
            t.barrier()                          # 所有 rank 对齐后再开始计时
            t0 = time.perf_counter()
            ALGOS[algo](t, count, nranks, rank)
            t1 = time.perf_counter()
            good, err = golden.check(t.data(), nranks, op_of(algo), rank)
            result_q.put((rank, algo, count, rep, t1 - t0, t.bytes_moved, t.msgs,
                          good, err, ""))
        t.close()
    except Exception as e:
        ok, why = False, f"{type(e).__name__}: {e}"
    finally:
        t = None                                 # 放掉 memoryview，否则 close 会报 BufferError
        try:
            shm.close()
        except Exception:
            pass                                 # 收尾异常不许掩盖真因

    result_q.put((rank, "__done__", 0, 0, 0.0, 0, 0, ok, 0.0, why))
    result_q.close()
    result_q.join_thread()                       # 强制 flush：主进程一定收得到
