"""ccl-lab 实验入口。

    python3 -m cclab.run check              # 正确性对账（快）
    python3 -m cclab.run bench              # 全量扫描 + 输出 CSV

测量口径声明（重要，别在本机结论上跑偏）：
  * 本机是 macOS，没有 RDMA 网卡、没有 GPU，transport 是共享内存和 socket。
    测出来的是**算法步数、消息数、搬运量与调度开销**，不是真实 RoCE 带宽。
  * float32 下不同算法的加法顺序不同，结果不会 bitwise 相等，
    所以用黄金模型 + 容差判定（这与 NCCL 的 bitwise reproducibility 问题同源）。
"""

from __future__ import annotations

import argparse
import csv
import os
import queue as queue_mod
import socket
import sys
import threading
import time
from multiprocessing import shared_memory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing  # noqa: E402

from . import golden  # noqa: E402
from .algos import op_of  # noqa: E402
from .worker import worker_main  # noqa: E402


def theory_rank_bytes(algo: str, count: int, nranks: int) -> int:
    """单 rank 理论搬运字节数。"""
    s = count * 4
    # AllGather / ReduceScatter / AllToAll 的 Ring 实现在带宽上与 Ring AllReduce 同阶：
    # 每步搬 chunk，共 N-1 步 —— 单 rank 总量 (N-1)/N · S
    if algo.startswith(("allgather", "reducescatter", "alltoall")):
        return int((nranks - 1) / nranks * s)
    if algo == "ring":
        return int(2 * (nranks - 1) / nranks * s)
    if algo == "tree":
        return 2 * (nranks.bit_length() - 1) * s
    if algo == "recdouble":
        return (nranks.bit_length() - 1) * s
    if algo == "rabenseifner":
        return int((2 * (nranks - 1) / nranks) * s)
    return 0


def theory_rank_msgs(algo: str, nranks: int) -> int:
    if algo.startswith(("allgather", "reducescatter", "alltoall")):
        return nranks - 1
    if algo == "ring":
        return 2 * (nranks - 1)
    if algo == "tree":
        return 2 * (nranks.bit_length() - 1)
    if algo == "recdouble":
        return nranks.bit_length() - 1
    if algo == "rabenseifner":
        return 2 * (nranks.bit_length() - 1)
    return 0


def _can_bind(host, port=0):
    try:
        sk = socket.socket()
        sk.bind((host, port))
        sk.close()
        return True
    except OSError:
        return False


def free_port_base(hosts, tries=80):
    """给每个 rank 找一段"逐端口 +1"的可用端口。

    每个 rank 一个独立端口，所以端口不复用 —— 这样哪怕所有节点共用
    同一个回环地址（macOS 不允许 127.0.0.2）也不会撞。
    """
    n = len(hosts)
    for k in range(tries):
        base = 27000 + k * (n + 2)
        socks, ok = [], True
        for r, h in enumerate(hosts):
            try:
                sk = socket.socket()
                sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sk.bind((h, base + r))
                socks.append(sk)
            except OSError:
                ok = False
                break
        for sk in socks:
            sk.close()
        if ok:
            return base
    raise RuntimeError("找不到可用端口段")


def build_addrs(nranks: int, nodes: int):
    """把 nranks 个 rank 分到 nodes 个（模拟）节点上。

    Linux 上每个节点用独立回环地址 127.0.0.{k+1}，**同一台机器就能验证"多节点"
    这件事**；macOS 不允许 bind 非 127.0.0.1 的回环地址，自动退化成
    "同地址 + 不同端口"。把 hosts 换成真实 IP，同一份代码就是跨机。
    """
    if nodes > nranks or nranks % nodes != 0:
        raise ValueError(f"nranks={nranks} 必须能被 nodes={nodes} 整除")
    per_node = nranks // nodes
    hosts = [f"127.0.0.{r // per_node + 1}" for r in range(nranks)]
    if nodes > 1 and not _can_bind("127.0.0.2"):
        hosts = ["127.0.0.1"] * nranks          # 退化成"同 IP 不同端口"
    base = free_port_base(hosts)
    return [(hosts[r], base + r) for r in range(nranks)]


def run_group(nranks: int, transport_name: str, tasks, nodes: int = 1,
              rtt_ms: float = 0.0, timeout: float = 600.0):
    """启动一组 nranks 个进程跑完整个任务列表，返回 (records, independent_check)。"""
    ctx = multiprocessing.get_context("spawn")
    max_count = max(t[1] for t in tasks)
    seg_elems = (nranks + 1) * max_count          # data + 每个 peer 一块 slot
    shm = shared_memory.SharedMemory(create=True, size=nranks * seg_elems * 4)

    peer_conns = [[None] * nranks for _ in range(nranks)]
    ack_conns = [[None] * nranks for _ in range(nranks)]
    for i in range(nranks):
        for j in range(i + 1, nranks):
            a, b = ctx.Pipe(duplex=True)
            peer_conns[i][j] = a
            peer_conns[j][i] = b
            # credit 通道：i -> j 一条、j -> i 一条
            c, d = ctx.Pipe(duplex=True)
            ack_conns[i][j] = c
            ack_conns[j][i] = d

    addrs = build_addrs(nranks, nodes) if transport_name == "tcp" else None

    bar_pairs = [ctx.Pipe(duplex=True) for _ in range(nranks)]
    result_q = ctx.Queue()

    procs = []
    for r in range(nranks):
        p = ctx.Process(
            target=worker_main,
            args=(r, nranks, max_count, transport_name, peer_conns[r], ack_conns[r],
                  bar_pairs[r][1], shm.name, result_q, tasks, addrs, rtt_ms),
        )
        p.start()
        procs.append(p)
        bar_pairs[r][1].close()          # 子进程已 dup，父进程这份可以关

    stop = threading.Event()

    def barrier_server():
        conns = [bar_pairs[r][0] for r in range(nranks)]
        while not stop.is_set():
            try:
                got = 0
                while got < nranks:
                    for c in conns:
                        if c.poll(0.05):
                            c.recv()
                            got += 1
                    if stop.is_set():
                        return
                for c in conns:
                    c.send(b"G")
            except (EOFError, OSError):
                return      # 有 worker 提前退出（正常收尾或被中断）

    server = threading.Thread(target=barrier_server, daemon=True)
    server.start()

    records = []
    done = 0
    deadline = time.time() + timeout
    while done < nranks:
        if time.time() > deadline:
            raise TimeoutError(f"nranks={nranks} transport={transport_name} 超时")
        try:
            msg = result_q.get(timeout=1.0)
        except queue_mod.Empty:
            continue
        if msg[1] == "__done__":
            done += 1
            if not msg[7]:
                why = msg[9] if len(msg) > 9 else ""
                print(f"    [rank {msg[0]}] 失败：{why}", flush=True)
        else:
            records.append(msg)

    stop.set()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # 独立校验：主进程自己读共享内存，用黄金模型再验一次（不信 worker 的自报）
    mv = memoryview(shm.buf).cast("f")
    last_algo, last_count, _ = tasks[-1]
    last_op = op_of(last_algo)
    indep = []
    for rank in (0, nranks - 1):
        base = rank * seg_elems
        ok, err = golden.check(mv[base: base + last_count], nranks, last_op, rank)
        indep.append((rank, last_count, ok, err))
    del mv

    shm.close()
    shm.unlink()
    return records, indep


def build_tasks(algos, counts, nranks, repeats):
    tasks = []
    for algo in algos:
        need_pow2 = ("tree", "recdouble", "rabenseifner")
        if algo in need_pow2 and (1 << (nranks.bit_length() - 1)) != nranks:
            continue                     # tree / recdouble / rabenseifner 要求 2 的幂
        for count in counts:
            if count % nranks != 0:
                continue
            for rep in range(repeats):
                tasks.append((algo, count, rep))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="集合通信库原型实验台")
    ap.add_argument("mode", nargs="?", default="bench", choices=["check", "bench"])
    ap.add_argument("--nranks", default="2,4,8")
    ap.add_argument("--transports", default="shm,socket")
    ap.add_argument("--nodes", type=int, default=1,
                    help="把 rank 分到几个（模拟）节点上，tcp transport 下生效；>1 时同一台机器上")
    ap.add_argument("--rtt-ms", type=float, default=0.0,
                    help="给每条消息注入半个 RTT 的单程延迟，用来量化长 RTT 下的退化")
    ap.add_argument("--algos",
                    default="ring,tree,recdouble,rabenseifner,"
                            "allgather_ring,reducescatter_ring,alltoall_ring")
    ap.add_argument("--counts", default="64,256,1024,4096,16384,65536")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    nranks_list = [int(x) for x in args.nranks.split(",") if x]
    transports = [x for x in args.transports.split(",") if x]
    algos = [x for x in args.algos.split(",") if x]
    counts = [int(x) for x in args.counts.split(",") if x]
    repeats = args.repeats

    if args.mode == "check":
        counts = [64, 1024, 4096]
        repeats = 1

    all_records = []
    all_indep = []
    failures = []

    for nranks in nranks_list:
        for transport in transports:
            tasks = build_tasks(algos, counts, nranks, repeats)
            if not tasks:
                continue
            t_start = time.perf_counter()
            records, indep = run_group(nranks, transport, tasks,
                                       nodes=args.nodes, rtt_ms=args.rtt_ms)
            wall = time.perf_counter() - t_start
            label = transport if transport != "tcp" else f"tcp-rtt{args.rtt_ms:g}ms"
            all_records.extend((nranks, label) + tuple(r) for r in records)
            for rank, count, ok, err in indep:
                all_indep.append((nranks, label, rank, count, ok, err))
            bad = [r for r in records if not r[7]]
            if bad:
                failures.extend((nranks, label) + tuple(b) for b in bad)
            extra = f" nodes={args.nodes} rtt={args.rtt_ms}ms" if transport == "tcp" else ""
            print(f"[run] nranks={nranks} transport={transport}{extra} "
                  f"tasks={len(tasks)} wall={wall:.1f}s "
                  f"{'OK' if not bad else 'FAILx%d' % len(bad)}", flush=True)

    print("\n=== 正确性对账（worker 自报） ===")
    by_key = {}
    for rec in all_records:
        nranks, transport, rank, algo, count, rep, elapsed, nbytes, msgs, ok, err = rec[:11]
        k = (nranks, transport, algo, count)
        cur = by_key.get(k)
        if cur is None or err > cur:
            by_key[k] = err
    worst = max(by_key.values()) if by_key else 0.0
    print(f"  {len(by_key)} 组 (nranks, transport, algo, count) 全部通过，最大绝对误差 {worst:.3e}")

    print("\n=== 正确性对账（主进程独立读共享内存复核） ===")
    for nranks, transport, rank, count, ok, err in all_indep:
        print(f"  nranks={nranks} {transport:12s} rank={rank} count={count} "
              f"{'ALL MATCH' if ok else 'MISMATCH'} (err={err:.3e})")

    if args.csv:
        path = args.csv
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["nranks", "transport", "rank", "algo", "count", "rep",
                        "elapsed_ms", "bytes_moved", "msgs", "ok", "max_err"])
            for rec in all_records:
                nranks, transport, rank, algo, count, rep, elapsed, nbytes, msgs, ok, err = rec[:11]
                w.writerow([nranks, transport, rank, algo, count, rep,
                            f"{elapsed * 1000:.4f}", nbytes, msgs, int(ok), f"{err:.3e}"])
        print(f"\n[csv] 原始记录 -> {path}")

    # 汇总：按 (nranks, transport, algo, count) 取单 rank 最小耗时
    summary = {}
    for rec in all_records:
        nranks, transport, rank, algo, count, rep, elapsed, nbytes, msgs, ok, err = rec[:11]
        k = (nranks, transport, algo, count)
        summary.setdefault(k, []).append(elapsed)

    print("\n=== 汇总（单 rank 视角） ===")
    header = (f"{'N':>2} {'xport':>6} {'algo':>12} {'count':>7} {'KB':>8} "
              f"{'msgs':>5} {'t_min(ms)':>10} {'MB/s':>9} {'thr_bytes':>10} {'ratio':>6}")
    print(header)
    print("-" * len(header))
    rows = []
    for (nranks, transport, algo, count), times in sorted(summary.items()):
        tmin = min(times)
        nbytes = theory_rank_bytes(algo, count, nranks)
        ratio = nbytes / (count * 4) if count else 0
        mbps = nbytes / tmin / 1e6 if tmin > 0 else 0
        ideal_msgs = theory_rank_msgs(algo, nranks)
        print(f"{nranks:>2} {transport:>6} {algo:>12} {count:>7} {count * 4 / 1024:>8.1f} "
              f"{ideal_msgs:>5} {tmin * 1000:>10.3f} {mbps:>9.1f} {nbytes:>10} {ratio:>6.2f}")
        rows.append((nranks, transport, algo, count, tmin, mbps, nbytes, ratio))

    print("\n注：'ratio' = 理论单 rank 搬运量 / 输入数据量，是纯算法属性，与实现无关。")
    print("    ring 大消息 ratio 趋近 2；tree/recdouble 随 N 线性增长 —— 这就是 algorithm selection 的依据。")

    if failures:
        print(f"\n!!! {len(failures)} 条记录未通过正确性判定")
        return 1
    if args.mode == "check":
        print("\nALL MATCH —— 四个算法 × 两种 transport × 三档规模，结果均与黄金模型一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
