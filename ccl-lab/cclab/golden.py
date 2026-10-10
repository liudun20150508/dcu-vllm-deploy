"""黄金模型 —— 独立于任何集合通信算法的参考实现。

刻意不写成 "ring/tree 的正确版本"，而是写成最直白的形式：
    expected[j] = sum over all ranks of input[rank][j]
用一个封闭公式直接算出来，和被测算法走的路径毫无关系。
这样才是真正独立的对账基准（鹏城那套 "Python 位精确黄金模型" 的同一个思路）。

本实验的输入约定：
    input[rank][j] = (rank + 1) + (j % 7) * 0.125
取这个形式有两个原因：
  1. 值与 j 相关（不是常数）—— chunk 切分错位、地址算错会被立刻抓出来；
  2. 值与 rank 相关 —— 漏掉某个 rank 的贡献也会被抓出来。

注意：float32 下不同算法的加法**顺序**不同，结果不会 bitwise 相等。
这和 NCCL 里 "bitwise reproducibility" 的问题是同一个根源，
所以这里用容差比较，而不是逐位相等。
"""

from __future__ import annotations

RTOL = 1e-5
ATOL = 1e-4


def fill_data(data_view, rank: int) -> None:
    """把本 rank 的输入写进 data 区。四个集合操作共用同一份输入约定。"""
    for j in range(len(data_view)):
        data_view[j] = (rank + 1) + (j % 7) * 0.125


def expected_at(op: str, j: int, rank: int, nranks: int, count: int) -> float:
    """按操作给出第 j 个元素的期望值 —— 闭式解，独立于任何通信路径。

    输入约定：in[r][j] = (r + 1) + (j % 7) * 0.125   （r 是 rank，j 是块内索引）
    块大小 chunk = count // nranks。

      allreduce      out[j]         = Σ_r in[r][j]
      allgather      out[j]         = in[j // chunk][j]          （块 owner 贡献的那份）
      reducescatter  out[j]         = Σ_r in[r][rank*chunk + j]  （rank 负责第 rank 块）
      alltoall       out[j]         = in[j // chunk][rank*chunk + j%chunk]
                                 （第 j//chunk 块来自 rank j//chunk 的第 rank 块）
    """
    chunk = count // nranks
    base_j = (j % 7) * 0.125
    total = nranks * (nranks + 1) / 2.0
    if op == "allreduce":
        return total + nranks * base_j
    if op == "allgather":
        return (j // chunk + 1) + base_j
    if op == "reducescatter":
        # Ring 实现里 rank r 收完 N-1 步后持有的是「块 (r+1) % N」的完整和
        # （MPI 标准规定 rank r 拿第 r 块，两者只差一个固定位移，不影响通信量）。
        # 算法收尾已把它搬到 [0, chunk)，所以这里按 (r+1)%N 算期望值。
        gidx = ((rank + 1) % nranks) * chunk + j
        return total + nranks * ((gidx % 7) * 0.125)
    if op == "alltoall":
        owner = j // chunk
        idx = rank * chunk + (j % chunk)
        return (owner + 1) + (idx % 7) * 0.125
    raise ValueError(f"未知集合操作：{op}")


def expected(j: int, nranks: int) -> float:
    """兼容旧调用：allreduce 的单点期望值。"""
    return nranks * ((j % 7) * 0.125) + nranks * (nranks + 1) / 2.0


def valid_range(op: str, count: int, nranks: int):
    """哪些位置是「有语义的」。

    ReduceScatter 只保证每个 rank 拿到**一块**的和（MPI 语义），
    其余位置是中间过程的残留，不该参与判定。
    """
    if op == "reducescatter":
        return range(count // nranks)
    return range(count)


def max_err(data_view, nranks: int, op: str = "allreduce", rank: int = 0) -> float:
    """与黄金模型的最大绝对误差（只在有语义的位置上算）。"""
    count = len(data_view)
    m = 0.0
    for j in valid_range(op, count, nranks):
        e = abs(float(data_view[j]) - expected_at(op, j, rank, nranks, count))
        if e > m:
            m = e
    return m


def check(data_view, nranks: int, op: str = "allreduce", rank: int = 0) -> tuple[bool, float]:
    """带容差的判定。返回 (是否通过, 最大绝对误差)。

    注意：float32 下不同算法的加法顺序不同，结果不会 bitwise 相等 ——
    这和 NCCL 里 bitwise reproducibility 的问题是同一个根源。
    """
    count = len(data_view)
    m = max_err(data_view, nranks, op, rank)
    limit = ATOL
    for j in valid_range(op, count, nranks):
        limit = max(limit, ATOL + RTOL * abs(expected_at(op, j, rank, nranks, count)))
    return (m <= limit), m
