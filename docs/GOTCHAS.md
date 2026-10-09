# GOTCHAS — 踩坑记录（脱敏）

> 按"基线 → 定位 → 根因 → 修复 → 验证"五步整理的实战坑位。每条都带
> **修复后的环境变量/参数**，适配者照着改即可。坑位编号与部署脚本注释一一对应。

## 网络 / 分布式

| # | 坑 | 根因 | 修复 |
|---|---|---|---|
| 1 | ZMQ `Cannot assign requested address` | 多机 c10d/TCPStore 与 vllm ZMQ 走了管理网地址 | 解析 fabric 地址（优先 `<host>i`，失败按网段规律推导）；管理网一律不用 |
| 2 | 端口冲突 | master/dp 端口写死 | 启动前 `ss -tln` 探测空闲端口再定 |
| 10 | RCCL 只能通单口 | 多网卡只有 `mlx5_0`（或 `shca_0`）能建 QP，其余口 `ibv_modify_qp EINVAL` | 钉死好口：`NCCL_IB_HCA=mlx5_0`（自研卡则 `NCCL_NET_PLUGIN=librccl-net-shca.so`） |

## 环境 / 工具链

| # | 坑 | 根因 | 修复 |
|---|---|---|---|
| 6 | `HSACOError: File operation failed` | sbatch 继承登录节点 TMPDIR，计算节点无该目录 | `export TMPDIR=/tmp TMP=/tmp TEMP=/tmp` |
| 7 | triton 编译 HSACO 失败 | DTK 自带 clang-17 不认识 HCU `-mllvm` 参数 | 用 aillvm clang-18：`TRITON_HIP_CLANG_PATH=.../aillvm/bin/clang-18` |
| 8 | `Intrinsic has incorrect argument type` | aillvm buffer rsrc intrinsic 签名不一致 | `export AMDGCN_USE_BUFFER_OPS=0` |
| 21 | triton launcher `.c` 编译炸 | 实为 C++（HIP 头），gcc 按 C 编译 | `export CC=$DTKROOT/bin/hipcc` |
| 24 | triton cache 并发写踩 | 多 worker 共享 `$HOME/.triton`，网络盘 flock 语义坏 | `export TRITON_CACHE_DIR=/tmp/triton-cache-<nodeid>` |

## vLLM / 算子

| # | 坑 | 根因 | 修复 |
|---|---|---|---|
| 4 | `Unsupported mxfp4_backend: EMULATION` | Mxfp4MoEMethod 硬拒 emulation | `--moe-backend auto`（ROCm/DCU 上选 AITER_MXFP4_BF16） |
| 9 | `expected BF16 but found Float8_e4m3fn` | fast-woa 分支拿 fp8 `wo_a` 直接做 bf16 einsum | `export VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA=0` |
| 11 | `ImportError: set_deepep_auto_use_low_latency` | vllm_hcu 装 wheel 漏跑 post_install，磁盘补丁未落盘 | `python -m vllm_hcu.post_install` |
| 12/14 | `Duplicate op name` / `operator registered multiple times` | 官方 deepseek_v4 与 vllm_hcu 实现撞车 | `patches/sitecustomize.py`（MODULE_EXCHANGE_MAP 补两条交换） |
| 13 | 走 custom allreduce 时 `amdsmi_init` NameError | das185 环境缺 amdsmi 模块 | `--disable-custom-all-reduce`（先跑通，性能后议） |
| 17 | AITER 拒 gfx906 | 老卡无 gfx938 code object | 换 FP8 后端走 vllm 内置 TRITON（`DUPFIX_FORCE_FP8=1`，仅老卡） |
| 22 | MHC tilelang `k_dim must be 4/16/32/64` | `T.gemm(k_pack=2, use_tf32=True)` 布局不支持 | `export VLLM_HCU_USE_CUSTOM_OPS=0` |
| 25 | KV cache 断言 `max(swa_page) <= max(full_mla_page)` | SWA 页硬编码 64 令牌，与 C4A KV block 冲突 | `--block-size 256` |
| 26 | 16G 卡 profile 后 KV cache 不足 | 0.85 util 下仅剩 1.1 GiB | `--gpu-memory-utilization 0.95` + 降 `MAXLEN` |
| 27 | worker 死、生成请求挂起 | flash_mla（CUDA 编译）不存在，调 raise 桩 | `export VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK=1` |

## 安装（gfx938 单卡专属）

| # | 坑 | 根因 | 修复 |
|---|---|---|---|
| A | venv 缺 `Python.h` | 系统 python3.10-minimal 无 dev 头 | 手动装 libpython3.10-dev |
| B | vllm_hcu 源码补丁未生效 | `--no-deps` 不触发 post-install hook | 手动 `python -m vllm_hcu.post_install` |
| C | sitecustomize `_Finder` 误判 | virtualenv 已有 `_virtualenv._Finder` | 改 `type(f) is _Finder` 精确判断 |
| D | tilelang 误判 CUDA target | 残留 `CUDA_HOME=/usr/local/cuda` | 启动时 `unset CUDA_HOME CUDA_PATH` |
| E | aiter fused_moe 签名不匹配 | vllm 传了 aiter 没有的 kwarg | `das185_patch.py` 补丁4 加 kwarg |
| F | gfx938 MXFP4 门槛 | `_supports_quant_scheme` 只认 `on_gfx950()` | `das185_patch.py` 补丁3 放开 gfx938 |
| G | dummy 模型 KV 反量化 OOM | 权重太小，KV cache 预算被塞满 | `gpu_memory_utilization=0.15` 限制 KV 预算 |

## 两代软件栈对照（别混用）

| 线 | DTK | torch | vllm_hcu | 适用架构 |
|---|---|---|---|---|
| das185（本框架） | 26.04 | 2.10.0+das.opt1 | 0.21.0+das.dtk2604 | gfx938 |
| das（旧） | 25.04 | 2.5.1+das.opt1 | 0.11.0+das | gfx906 |

## 遗留的集群特定项（换集群要改）

- 网段推导规律：管理网 `10.6.x.x → 12.6.x.x`（`v4pro_4n32c_serve.sh` 里的
  `sed 's/^10\./12./'`）。**这是目标集群的网络规划，换集群按实际网段改。**
- GLIBC shim：vllm 的 `.so` 链了 glibc 2.32 符号，老集群 glibc 2.31 需
  `LD_PRELOAD` 自编 shim（`run_serve_node.sh` 里已注释说明）。
