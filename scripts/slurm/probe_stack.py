# 全链路探针（kshdnormal 版）：import → 补丁 → 配置 → triton 编译出 hsaco
# 参考 zksg-all 验收/patches/README.md 的坑 1/3/4 修复点逐项验证
import os, sys, glob

fails = []

def ok(msg):
    print(f"  [OK] {msg}", flush=True)

def bad(msg):
    print(f"  [FAIL] {msg}", flush=True)
    fails.append(msg)

print("== 0. 基础环境 ==", flush=True)
import platform
print("  python", sys.version.split()[0], "| glibc", platform.libc_ver()[1], flush=True)
print("  TRITON_HIP_CLANG_PATH =", os.environ.get("TRITON_HIP_CLANG_PATH"), flush=True)
print("  AMDGCN_USE_BUFFER_OPS =", os.environ.get("AMDGCN_USE_BUFFER_OPS"), flush=True)

print("== 1. torch / vllm / vllm_hcu ==", flush=True)
try:
    import torch
    ok(f"torch {torch.__version__} | device_count={torch.cuda.device_count()} | "
       f"dev0={torch.cuda.get_device_name(0) if torch.cuda.device_count() else 'N/A'}")
except Exception as e:
    bad(f"torch import: {type(e).__name__}: {e}")
    print("torch 不可用，后续项无意义，提前结束", flush=True)
    sys.exit(1)

for mod, name in [("vllm", "vllm"), ("vllm_hcu", "vllm_hcu")]:
    try:
        m = __import__(mod)
        ok(f"{name} {getattr(m, '__version__', '?')}")
    except Exception as e:
        bad(f"{name} import: {type(e).__name__}: {e}")

print("== 2. 厂商 tilelang（坑3）==", flush=True)
try:
    import tilelang
    from tilelang.layout import make_hcu_swizzled_layout  # noqa
    ok(f"tilelang {tilelang.__version__} + make_hcu_swizzled_layout")
except Exception as e:
    bad(f"tilelang: {type(e).__name__}: {e}")

print("== 3. aiter shuffle 补丁（坑1/aiter） ==", flush=True)
try:
    from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4  # noqa
    ok("shuffle_weight_a16w4 / shuffle_scale_a16w4 可导入")
except Exception as e:
    bad(f"aiter shuffle: {type(e).__name__}: {e}")

print("== 4. 模型配置可读 ==", flush=True)
try:
    from transformers import AutoConfig
    for m in ["DeepSeek-V4-Flash-0731", "DeepSeek-V4-Pro-0813"]:
        p = os.path.expanduser(f"~/models/{m}")
        c = AutoConfig.from_pretrained(p)
        ok(f"{m}: {c.model_type} {c.architectures} layers={c.num_hidden_layers}")
except Exception as e:
    bad(f"AutoConfig: {type(e).__name__}: {e}")

print("== 5. triton 编译+运行（坑4：aillvm/clang-18 链路）==", flush=True)
cache_before = set(glob.glob(os.path.expanduser("~/.triton/cache/**/*.hsaco"), recursive=True))
try:
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(n // 256,)](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    assert torch.allclose(out, x + y, atol=1e-5), "结果数值不对"
    ok(f"triton {triton.__version__} kernel 编译+运行通过 out[:3]={out[:3].tolist()}")
except Exception as e:
    bad(f"triton: {type(e).__name__}: {e}")

cache_after = set(glob.glob(os.path.expanduser("~/.triton/cache/**/*.hsaco"), recursive=True))
new_hsaco = cache_after - cache_before
if new_hsaco:
    ok(f"hsaco 已生成: {[os.path.basename(h) for h in new_hsaco]}")
else:
    bad("triton cache 中没有新 hsaco（编译未走通到 codegen）")

print("== 6. 简单 HCU 矩阵乘（rocBLAS 路径）==", flush=True)
try:
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    ok(f"matmul OK c[0,0]={c[0,0].item():.4f}")
except Exception as e:
    bad(f"matmul: {type(e).__name__}: {e}")

print("", flush=True)
if fails:
    print(f"### 探针结果: {len(fails)} 项失败 ###", flush=True)
    for f in fails:
        print("  - " + f, flush=True)
    sys.exit(1)
print("### 探针结果: 全部通过 ###", flush=True)
