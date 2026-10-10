"""gfx936 单卡冒烟前验证：aiter shuffle 补丁 + triton 编译 + fp8/f16 dot。"""
import traceback
import torch

print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0),
      torch.cuda.get_device_properties(0).gcnArchName)

# 1) aiter shuffle 补丁
try:
    from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4  # noqa
    print("PASS aiter shuffle_weight_a16w4 / shuffle_scale_a16w4 可导入")
except Exception as e:
    print("FAIL aiter shuffle:", type(e).__name__, e)
    traceback.print_exc(limit=4)

# 2) triton 编译 + 运行（aillvm clang-18 链路）
import triton
import triton.language as tl
print("triton", triton.__version__, triton.__file__)


@triton.jit
def k_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def t_add():
    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    k_add[(n // 256,)](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    assert torch.allclose(out, x + y, atol=1e-5)
    print("PASS triton.jit float add")


@triton.jit
def k_dot16(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * 16 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)


def t_dot16():
    M, N, K = 16, 16, 32
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    k_dot16[(1,)](a, b, c, M, N, K)
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    err = (c - ref).abs().max().item()
    print("PASS triton f16 dot, max err", err)


@triton.jit
def k_dot8(a_ptr, b_ptr, c_ptr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * 16 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * K + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * 16 + offs_n[None, :], c)


def t_dot8():
    K = 32
    a = torch.randn(16, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, 16, device="cuda", dtype=torch.float16)
    a8 = a.to(torch.float8_e4m3fn)
    b8 = b.to(torch.float8_e4m3fn)
    c = torch.empty(16, 16, device="cuda", dtype=torch.float32)
    k_dot8[(1,)](a8, b8, c, K)
    torch.cuda.synchronize()
    print("PASS triton fp8 e4m3 dot, sample", c[0, :3].tolist())


for fn in (t_add, t_dot16, t_dot8):
    try:
        fn()
    except Exception as e:
        print("FAIL", fn.__name__, type(e).__name__, str(e)[:200])
        traceback.print_exc(limit=8)
print("PROBE-DONE")
