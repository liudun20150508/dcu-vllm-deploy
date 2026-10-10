"""ccl-lab：集合通信库原型实验台。

模块划分（对应 UCX / NCCL 的实际分层）：
    transport.py  传输抽象        ≈ UCX UCT / NCCL NET plugin
    algos.py      集合通信算法    ≈ NCCL 的 Ring / Tree / Rabenseifner
    golden.py     黄金模型        ≈ 参考实现（numpy allreduce / NCCL 单进程基准）
    run.py        实验编排 + 性能测试与分析（≈ 自建 benchmark + tuning 表）
    worker.py     rank 进程

跑法见 README.md。
"""

__all__ = ["transport", "algos", "golden", "run", "worker"]
