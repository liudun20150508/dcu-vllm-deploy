#!/usr/bin/env python3
"""给 das185 环境打 gfx936 适配补丁（对应之前 906/928 适配的坑位）。"""
import sys
from pathlib import Path

SITE = Path("/root/das185/env/lib/python3.10/site-packages")

# ---- 补丁1：aiter shuffle 缺 shuffle_weight_a16w4 / shuffle_scale_a16w4 ----
shuffle_py = SITE / "aiter/ops/shuffle.py"
patch_file = Path("/root/private_data/中科曙光/zksg-all/中科曙光/验收/patches/aiter_ops_shuffle_a16w4.patch")
if "def shuffle_weight_a16w4" not in shuffle_py.read_text():
    added = []
    for line in patch_file.read_text().splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    body = "\n".join(added)
    assert "def shuffle_weight_a16w4" in body and "def shuffle_scale_a16w4" in body
    with shuffle_py.open("a") as f:
        f.write("\n\n" + body + "\n")
    print("[patch1] aiter shuffle 补丁已应用")
else:
    print("[patch1] aiter shuffle 已存在，跳过")

# ---- 补丁2：triton hcu compiler.py 跳过 aillvm 不认识的 flag ----
compiler_py = SITE / "triton/backends/hcu/compiler.py"
txt = compiler_py.read_text()
orig = txt
txt = txt.replace('if not os.environ.get("PMD_PATH"):',
                  'if not os.environ.get("PMD_PATH") and False:')
txt = txt.replace("if options.enable_fp_fusion:",
                  "if options.enable_fp_fusion and False:")
if txt != orig:
    compiler_py.write_text(txt)
    print("[patch2] triton compiler flags 补丁已应用")
else:
    print("[patch2] triton compiler flags 已存在或未匹配")

# ---- 补丁3：gfx936/gfx938 的 MXFP4 MoE 门槛（AiterExperts._supports_quant_scheme）----
aiter_moe = SITE / "vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py"
txt = aiter_moe.read_text()
orig = txt
old = """        if weight_key == kMxfp4Static:
            from vllm.platforms.rocm import on_gfx950

            if not on_gfx950():
                return False"""
new = """        if weight_key == kMxfp4Static:
            from vllm.platforms.rocm import on_gfx950

            # [中科曙光 gfx936 适配] aiter 自带 gfx936/gfx938 的 fp4_w4a16 CK 算子，
            # 原门槛只认 gfx950，导致海光 BW(gfx936) 被拒。
            _arch = torch.cuda.get_device_properties("cuda").gcnArchName
            if not (on_gfx950() or any(a in _arch for a in ("gfx936", "gfx938"))):
                return False"""
if old in txt:
    txt = txt.replace(old, new)
    aiter_moe.write_text(txt)
    print("[patch3] gfx936 MXFP4 门槛补丁已应用")
elif "gfx936" in txt:
    print("[patch3] gfx936 补丁已存在，跳过")
else:
    print("[patch3] WARN: 未找到 MXFP4 门槛代码段，需人工确认")

# ---- 补丁4：aiter fused_moe 签名补 vllm 260611 传的 keyword 参数 ----
# vllm 0.21.0(260611) 的 _aiter_ops.py 调 aiter.fused_moe 时传了
# num_local_tokens/dtype/hidden_pad/intermediate_pad/bias1/bias2，
# 海光 aiter 0.1.5(260813) 的 fused_moe 没有这些参数 → TypeError。
# 单卡(EP=1)冒烟下这些参数为 None/0，可安全忽略。
fused_moe_py = SITE / "aiter/fused_moe.py"
txt = fused_moe_py.read_text()
old = """    # following for tuning
    block_size_M=None,
):"""
new = """    # following for tuning
    block_size_M=None,
    # [中科曙光 gfx936 适配] vllm 260611 的 _aiter_ops 会传这些 kwarg，
    # 海光 aiter 260813 不支持；单卡 EP=1 下为 None/0，忽略即可。
    num_local_tokens=None,
    dtype=None,
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
):"""
if old in txt:
    txt = txt.replace(old, new)
    fused_moe_py.write_text(txt)
    print("[patch4] aiter fused_moe 签名补丁已应用")
else:
    print("[patch4] WARN: 未找到 fused_moe 签名段，需人工确认")

print("done")
