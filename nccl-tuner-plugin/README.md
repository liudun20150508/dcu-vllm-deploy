# nccl-tuner-plugin —— NCCL 自动调优插件骨架（C++17）

> **这份代码回答 JD 职责 6 里的一句话**：
> 「建立通信性能测试与分析体系，开发性能测试、数据分析、问题定位及**自动化调优工具**」
>
> **NCCL 给"自动化调优"留的官方挂载点，就是 tuner plugin。**

## 一句话理解它

```
NCCL 核心  算出 collCostTable[算法][协议] = 预估耗时
                      │
                      ▼
              tuner plugin  ← 你在这里决策（本文件）
                      │
                      ▼
        改写这次集合通信用哪个算法/协议/几条通道
```

**调优不是在外面挂脚本改环境变量，而是在库内部按规则决策。**

## 和 NET plugin 的分工（两个插件别混）

| | NET plugin | **tuner plugin** |
|---|---|---|
| 解决什么 | 通信库**认不认识**这张网卡 | 认得了之后**走哪条最合适** |
| 关心的 | QP 建链、MR 注册、收发 | 算法(Ring/Tree/…)、协议(LL/LL128/Simple)、通道数 |
| 不实现会怎样 | **跑不起来** | 跑得**不够快**（NCCL 用自己的默认表） |
| 本仓库对应 | `../nccl-net-plugin/` | **这里** |

## 三个回调（NCCL tuner v5）

| 回调 | 什么时候被调 | 干什么 |
|---|---|---|
| `init` | NCCL 启动一次 | 拿拓扑：`nRanks / nNodes / rank / nRanksPerNode / compCap / collNetSupport / nvlsSupport` |
| **`getCollInfo`** | **每次集合通信启动前** | **核心**：拿到 `collCostTable`，写回 `algorithm / protocol`，可改 `nChannels` |
| `finalize` | 退出 | 释放 |

### `getCollInfo` 的入参里三个关键点

- **`collCostTable[numAlgo][numProto]`**：**NCCL 已经算好的代价表**。
  `< 0` 表示"这个组合不可用"（比如没有 SHARP 就选不了 CollNet）。
- **`regression > 0`**：**NCCL 正在做自校准，此时绝对不能改选择** ——
  改了会让它的回归数据失真。**这是最容易写错的一处**，测试里专门覆盖了。
- ⚠️ **`collCostTable` 签名是 `float**`，但真实内存是连续二维数组** ——
  NCCL 内部把它强转成 `float**`（见 `tuner_v2.cc`：
  `float (*table)[NCCL_NUM_PROTOCOLS] = (float(*)[NCCL_NUM_PROTOCOLS])collCostTable`）。
  **必须按扁平下标 `a*numProto+p` 访问**，不能当"指针数组"写 `table[a][p]`，
  否则会把两个 float 的位模式当指针 → 空指针/野指针 → 段错误。
  **这正是一次真 RCCL 上 `getCollInfo` SIGSEGV 的根因**，本插件已修（见 `mycard_tuner.cpp` 的 `cost_flat`）。

## 本骨架的选择逻辑（三步）

```
① 基线：从 collCostTable 里取代价最小的 (算法, 协议)
      —— 等于"不动偏置"，先保证不比默认差

② 偏置：按消息大小和规模，应用自己的规则
      规则来自我在 ccl-lab（自写的集合通信原型）里量化出的结论：
        · 小消息：**步数**主导   → 倾向 Tree（log N 步 vs Ring 的 2(N-1) 步）
        · 大消息：**每步搬运量**主导 → 倾向 Ring（双向流水）
        · 交叉点**随规模右移**：rank 越多，Tree 的优势区间越宽
      代码里用 `compute_bias()` 表达，阈值随 nRanks 缩放。

③ 通道数：**跨机 + 大消息**时抬高下限（骨架里是 8）
      —— 把大象流摊到多条通道上，避免 ECMP 哈希撞同一条链路
```

**每一步都打日志**，方便和 `NCCL_DEBUG=INFO` 的输出对上。

## 怎么挂上去

```bash
# ① 用真头文件编（必须！否则结构体布局不一致会段错误）
make so NCCL_INC=/path/to/nccl/src/include/plugin

# ② 让 NCCL 加载
export NCCL_TUNER_PLUGIN=$PWD/libnccl-tuner-mycard.so
export NCCL_DEBUG=INFO          # 看 [tuner] 行的选择结果

# ③ 跑个 benchmark 对照
#    对照组：不设 NCCL_TUNER_PLUGIN
```

**⚠️ 本机验证用的是自带的 `nccl_tuner.h`（stub）**，
**挂到真 NCCL 上必须换成 NCCL 源码里的真头文件** —— 文件头已注明。

## 本机验证

```bash
make            # 编译 + 跑测试 + 编 .so
make test       # 只跑测试
```

测试覆盖 5 种情形：

| # | 场景 | 期望 |
|---|---|---|
| 1 | 8KB 小消息 | 偏置到 **Tree**（步数主导） |
| 2 | 64MB 大消息、跨机 | 偏置到 **Ring** + **channels ≥ 8** |
| 3 | **`regression > 0`** | **不应用偏置**（保护 NCCL 自校准） |
| 4 | 单机 64MB | 偏置到 Ring，但**不抬通道** |
| 5 | `finalize` | 状态清空 |

编译用 `-Wall -Wextra -Werror -O2 -fPIC`，**零警告**。

## 这份代码的定位（别当成生产实现）

- **是教学骨架**：接口填满、每个判断的依据写清楚、有测试。
- **不是生产调优器**：`compute_bias()` 里的阈值是公式，真实系统应该用**实测定标出来的表**。
- **它证明的是"我知道自动化调优该挂在哪、按什么决策"** ——
  而不是"我调出了一个更好的 tuner"。
