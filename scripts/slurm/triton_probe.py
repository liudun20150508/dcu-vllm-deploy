# 坑21 探针 v2：libtriton 只认 gfx926/928/936/938，gfx906 被拒（unsupported target）。
# 骗术：make_llir 期间把 options.arch 换成 gfx926 过 C++ 检查，
# make_amdgcn（外部 clang）仍用真 arch=gfx906 出 ISA。
# 重点：f16 dot / fp8 dot 两个 MoE 核心算子能不能编译+跑。
import traceback
import torch

print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0),
      torch.cuda.get_device_properties(0).gcnArchName)
import triton
print("triton", triton.__version__, triton.__file__)

REAL_ARCH = "gfx906"
FAKE_ARCH = "gfx906"  # 运行时按候选表试探，找到第一个 C++ 认的
CANDIDATES = ["gfx938", "gfx936", "gfx928", "gfx926"]

import triton.backends.amd.compiler as amdc
HIPBackend = amdc.HIPBackend
_orig_make_llir = HIPBackend.make_llir

@staticmethod
def patched_make_llir(src, metadata, options):
    # make_llir 是 staticmethod：换 options.arch 只影响本方法内（C++ 检查+LLVM 侧），
    # make_amdgcn 用的是外层 options 对象，会被我们恢复，继续按 gfx906 出 ISA。
    if options.arch == REAL_ARCH and FAKE_ARCH != REAL_ARCH:
        object.__setattr__(options, "arch", FAKE_ARCH)  # frozen dataclass
        try:
            return _orig_make_llir(src, metadata, options)
        finally:
            object.__setattr__(options, "arch", REAL_ARCH)
    return _orig_make_llir(src, metadata, options)

HIPBackend.make_llir = patched_make_llir
print("make_llir arch-spoof installed:", REAL_ARCH, "-> (候选)",
      CANDIDATES)

import triton.language as tl


def try1(label, fn):
    try:
        fn()
        print("PASS:", label)
    except Exception:
        print("FAIL:", label)
        traceback.print_exc(limit=6)
        print("----")


# 1) triton.jit float pointwise
@triton.jit
def k_add(x_ptr, y_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 64 + tl.arange(0, 64)
    x = tl.load(x_ptr + offs, mask=offs < N)
    tl.store(y_ptr + offs, x + 1.0, mask=offs < N)


def t1():
    x = torch.randn(128, device="cuda")
    y = torch.empty_like(x)
    k_add[(2,)](x, y, 128)
    torch.cuda.synchronize()
    assert torch.allclose(y, x + 1.0)


# 2) triton.jit int64 bitwise（复刻 vocab 掩码核的算子集）
@triton.jit
def k_mask(x_ptr, y_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 64 + tl.arange(0, 64)
    m = offs < N
    x = tl.load(x_ptr + offs, mask=m, other=0)
    r = (x & 7) | ((x < 10) & (x > 2))
    tl.store(y_ptr + offs, r, mask=m)


def t2():
    x = torch.randint(0, 100, (128,), device="cuda", dtype=torch.int64)
    y = torch.empty_like(x)
    k_mask[(2,)](x, y, 128)
    torch.cuda.synchronize()
    print("  int64 kernel sample:", y[:4].tolist())


# 3) f16 dot（MoE 反量化路线的核心算子）
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


def t3():
    M, N, K = 16, 16, 32
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    k_dot16[(1,)](a, b, c, M, N, K)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float())
    print("  f16 dot max err:", (c - ref).abs().max().item())


# 4) fp8 e4m3 dot（vllm TritonExperts 的原生路径）
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


def t4():
    K = 32
    a = torch.randn(16, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, 16, device="cuda", dtype=torch.float16)
    a8 = a.to(torch.float8_e4m3fn)
    b8 = b.to(torch.float8_e4m3fn)
    c = torch.empty(16, 16, device="cuda", dtype=torch.float32)
    k_dot8[(1,)](a8, b8, c, K)
    torch.cuda.synchronize()
    print("  fp8 dot ran, sample:", c[0, :3].tolist())


# 5) inductor torch.compile int64 bitwise（vocab 掩码的 vllm 路径）
def t5():
    def f(x):
        return (x & 7) | ((x < 10) & (x > 2))
    cf = torch.compile(f, backend="inductor")
    x = torch.randint(0, 100, (128,), device="cuda", dtype=torch.int64)
    y = cf(x)
    torch.cuda.synchronize()
    print("  inductor int64 sample:", y[:4].tolist())


# 找 C++ 认的假架构：依次试探，第一个能过 float 核的即为可用
import contextlib, io

def silent(fn):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        fn()
    return buf.getvalue()

chosen = None
for cand in CANDIDATES:
    FAKE_ARCH = cand
    try:
        silent(t1)
        chosen = cand
        break
    except Exception as e:
        print("candidate", cand, "rejected:", str(e).splitlines()[0][:80])
if chosen:
    print("CHOSEN-FAKE-ARCH:", chosen)
    FAKE_ARCH = chosen
    try1("triton.jit float add", t1)
    try1("triton.jit int64 bitwise", t2)
    try1("triton.jit f16 dot", t3)
    try1("triton.jit fp8 e4m3 dot", t4)
    try1("inductor int64 bitwise", t5)
else:
    print("NO-CANDIDATE-WORKS")
print("PROBE-DONE")
