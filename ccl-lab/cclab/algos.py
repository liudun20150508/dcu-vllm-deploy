"""四个集合通信算法（AllReduce）。

每个算法只调 transport 的四个原语，不知道底下是共享内存还是 socket。
比照面是 NCCL 的算法选择：
    小消息 → tree / 低延迟算法（步数少、每步全量，延迟 O(log N)）
    大消息 → ring（每步只搬 S/N，带宽 O(2(N-1)/N · S)）
    中消息 → recursive doubling 一类的折中
"什么时候切" 由 bench 出来的交叉点决定，不是拍脑袋。
"""

from __future__ import annotations


def allreduce_ring(t, count: int, nranks: int, rank: int) -> None:
    """Ring AllReduce：reduce-scatter + all-gather，各 N-1 步。

    每步只搬 count/N 个元素 —— 带宽最优，但步数 O(N)，延迟随规模线性增长。
    这就是 NCCL 大消息默认走 Ring 的原因。
    """
    if count % nranks != 0:
        raise ValueError("ring 要求 count 能被 nranks 整除")
    chunk = count // nranks
    right = (rank + 1) % nranks
    left = (rank - 1) % nranks

    # 1) reduce-scatter：第 step 步，我发 chunk (r-step)，收 chunk (r-step-1) 并累加
    for step in range(nranks - 1):
        send_idx = (rank - step) % nranks
        recv_idx = (rank - step - 1) % nranks
        t.send(right, send_idx * chunk, chunk)
        t.recv_add(left, chunk, recv_idx * chunk)
    # 收尾后本 rank 持有的是 chunk (rank+1) % nranks 的完整归约结果

    # 2) all-gather：把各自手里的完整 chunk 沿着环传一圈
    for step in range(nranks - 1):
        send_idx = (rank - step + 1) % nranks
        recv_idx = (rank - step) % nranks
        t.send(right, send_idx * chunk, chunk)
        t.recv_copy(left, chunk, recv_idx * chunk)


def _tree_pairs(nranks: int, rank: int, mask: int):
    """返回 (角色, partner)，角色 ∈ {'none','recv','send'}。

    只有 rank % (2*mask) 为 0（收方）或 mask（发方）的节点参与这一层。
    """
    if (rank & mask) == 0:
        if (rank & (mask - 1)) == 0:
            return "recv", rank | mask
        return "none", -1
    if (rank & (mask - 1)) == 0:
        return "send", rank & ~mask
    return "none", -1


def allreduce_tree(t, count: int, nranks: int, rank: int) -> None:
    """二叉树 AllReduce：规约 O(log N) 步 + 广播 O(log N) 步。

    步数少（延迟低），但每步搬全量数据（带宽差）—— 小消息场景的赢家。
    """
    masks = [1 << i for i in range(nranks.bit_length() - 1)]
    if (1 << len(masks)) != nranks:
        raise ValueError("tree 要求 nranks 是 2 的幂")

    # 规约：叶子 → 根
    for mask in masks:
        role, partner = _tree_pairs(nranks, rank, mask)
        if role == "send":
            t.send(partner, 0, count)
        elif role == "recv":
            t.recv_add(partner, count, 0)

    # 广播：根 → 叶子
    for mask in reversed(masks):
        role, partner = _tree_pairs(nranks, rank, mask)
        if role == "send":
            t.recv_copy(partner, count, 0)
        elif role == "recv":
            t.send(partner, 0, count)


def allreduce_recursive_doubling(t, count: int, nranks: int, rank: int) -> None:
    """Recursive Doubling：log2(N) 步，每步与 rank^mask 交换全量并累加。

    步数最少（延迟最低），但每步全量，总搬运量 N·log2(N)·S。
    小消息 + 大 N 时延迟优势明显，大消息时带宽代价高。
    """
    if (1 << (nranks.bit_length() - 1)) != nranks:
        raise ValueError("recursive doubling 要求 nranks 是 2 的幂")
    mask = 1
    while mask < nranks:
        partner = rank ^ mask
        t.send(partner, 0, count)
        t.recv_add(partner, count, 0)
        mask <<= 1


def allreduce_rabenseifner(t, count: int, nranks: int, rank: int) -> None:
    """Rabenseifner：Recursive Halving 做 reduce-scatter + Recursive Doubling 做 all-gather。

    兼得两边的好处：步数 2·log2(N)（延迟好），每步数据量逐层减半
    （总搬运量与 Ring 同阶）—— 这就是 NCCL 里 "Rabenseifner 变体" 的位置。
    """
    if (1 << (nranks.bit_length() - 1)) != nranks:
        raise ValueError("rabenseifner 要求 nranks 是 2 的幂")
    if count % nranks != 0:
        raise ValueError("rabenseifner 要求 count 能被 nranks 整除")

    # --- Recursive Halving：reduce-scatter ---
    mask = nranks >> 1
    off = 0
    rem = count
    while mask >= 1:
        partner = rank ^ mask
        half = rem // 2
        if (rank & mask) == 0:
            # 我保留低半，把高半送走，收对方的低半累加进来
            t.send(partner, off + half, half)
            t.recv_add(partner, half, off)
            rem = half
        else:
            # 我保留高半，把低半送走，收对方的高半累加进来
            t.send(partner, off, half)
            t.recv_add(partner, half, off + half)
            off += half
            rem = half
        mask >>= 1

    # --- Recursive Doubling：all-gather ---
    blk = count // nranks
    mask = 1
    off = rank * blk
    rem = blk
    while mask < nranks:
        partner = rank ^ mask
        if (rank & mask) == 0:
            t.send(partner, off, rem)
            t.recv_copy(partner, rem, off + rem)
            rem *= 2
        else:
            t.send(partner, off, rem)
            t.recv_copy(partner, rem, off - rem)
            off -= rem
            rem *= 2
        mask <<= 1



# ============================================================
# 另外三个集合操作（JD 职责 3 明文列举：AllGather / ReduceScatter / AllToAll）
# 统一约定：每个 rank 的数据区都是 count 个 float32，块大小 chunk = count // nranks
# ============================================================


def allgather_ring(t, count: int, nranks: int, rank: int) -> None:
    """Ring AllGather：每个 rank 贡献 1/N 的数据，收齐后人人持有全部 N 份。

    步数 N-1，每步搬 chunk —— 总搬运量 (N-1)/N · S，和 Ring AllReduce 的带宽特性一致。
    """
    if count % nranks != 0:
        raise ValueError("allgather 要求 count 能被 nranks 整除")
    chunk = count // nranks
    right = (rank + 1) % nranks
    left = (rank - 1) % nranks
    # 初始：rank r 的贡献在块 r（data 的 [rank*chunk, (rank+1)*chunk)）
    for step in range(nranks - 1):
        send_idx = (rank - step) % nranks
        recv_idx = (rank - step - 1) % nranks
        t.send(right, send_idx * chunk, chunk)
        t.recv_copy(left, chunk, recv_idx * chunk)


def reducescatter_ring(t, count: int, nranks: int, rank: int) -> None:
    """Ring ReduceScatter：输入是 N 份数据，每个 rank 只留下自己负责那一份的和。

    这就是 Ring AllReduce 的 reduce-scatter 阶段（N-1 步）。
    注意：Ring 走完后本 rank 持有的是「块 (rank+1) % N」的和，而 MPI 语义规定
    rank r 拿到第 r 块 —— 两者只差一个固定映射，不影响通信量。这里在收尾把它
    搬到 buffer 开头，好让接口语义干净（这一步不产生通信）。
    """
    if count % nranks != 0:
        raise ValueError("reducescatter 要求 count 能被 nranks 整除")
    chunk = count // nranks
    right = (rank + 1) % nranks
    left = (rank - 1) % nranks
    for step in range(nranks - 1):
        send_idx = (rank - step) % nranks
        recv_idx = (rank - step - 1) % nranks
        t.send(right, send_idx * chunk, chunk)
        t.recv_add(left, chunk, recv_idx * chunk)
    # 收尾：把本 rank 负责的块挪到 [0, chunk)，对齐 MPI_Reduce_scatter 语义
    src = ((rank + 1) % nranks) * chunk
    if src != 0:
        d = t.data()
        d[0:chunk] = d[src:src + chunk]


def alltoall_ring(t, count: int, nranks: int, rank: int) -> None:
    """Ring AllToAll：第 j 块发给 rank j，同时收下别人发给我的那一块。

    步数 N-1，每步搬 chunk —— 这是 MoE 里 all-to-all 的最简形态
    （真实 MoE 的 all-to-all 还要配路由与变长切分）。

    ⚠️ 这里有个非平凡的坑，值得记下来：**Ring 旋转中「接收落点」和
    「后续发送的源」是同一批位置**，所以单靠一个快照解决不了。

        step k: 发块 (r+k)，落块 (r-k)
        而块 (r-k) 在第 (N-k) 步还要被当作发送源

    于是有两次覆盖，必须两头都护住：
      1) 发送前，从 snap 把要发的块恢复回来；
      2) **接收后立刻把结果搬进 out**，否则下一步"恢复发送块"会把刚收到的结果抹掉。
    第一版只做了 (1)，N=2 恰好蒙对（只有一步），N>=3 立刻算错。

    快照与回填都用 memoryview 切片（memmove 级）而不是逐元素循环 ——
    否则本地拷贝会盖过通信本身的开销，测出来的时间就没有意义了。
    真实实现用双缓冲 / 乒乓缓冲，这里是应用层缓冲。
    """
    if count % nranks != 0:
        raise ValueError("alltoall 要求 count 能被 nranks 整除")
    chunk = count // nranks
    d = t.data()
    snap = bytes(d)                                     # 一次 C 级拷贝做快照
    snap_mv = memoryview(snap).cast("f")
    out = bytearray(snap)                               # 自己的第 rank 块本来就对
    out_mv = memoryview(out).cast("f")
    for step in range(1, nranks):
        dst = (rank + step) % nranks
        src = (rank - step) % nranks
        lo, hi = dst * chunk, (dst + 1) * chunk
        d[lo:hi] = snap_mv[lo:hi]                       # ① 恢复要发的那一块
        t.send(dst, lo, chunk)                          # 我的第 dst 块，正是给 dst 的
        t.recv_copy(src, chunk, src * chunk)            # 收 src 给我的那块
        lo, hi = src * chunk, (src + 1) * chunk
        out_mv[lo:hi] = d[lo:hi]                        # ② 立刻搬走，防后续覆盖
    d[:] = out_mv[:]



ALGOS = {
    "ring": allreduce_ring,
    "tree": allreduce_tree,
    "recdouble": allreduce_recursive_doubling,
    "rabenseifner": allreduce_rabenseifner,
    # 另外三个集合操作（每个只有 Ring 一种实现，作为该操作的基准线）
    "allgather_ring": allgather_ring,
    "reducescatter_ring": reducescatter_ring,
    "alltoall_ring": alltoall_ring,
}


def op_of(algo: str) -> str:
    """算法名 -> 集合操作名（用来选黄金模型）。"""
    for op in ("allgather", "reducescatter", "alltoall"):
        if algo.startswith(op):
            return op
    return "allreduce"

# 每个算法的适用规模约束（用于 bench 自动过滤）
ALGO_MIN_NRANKS = {"tree": 2, "recdouble": 2, "rabenseifner": 2}
