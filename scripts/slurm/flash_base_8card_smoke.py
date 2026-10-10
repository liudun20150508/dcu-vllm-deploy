"""8 卡 gfx936 真实权重冒烟：DeepSeek-V4-Flash-Base（275G fp8）TP=8 加载 + 前向。

与 flash0731_8card_smoke.py 同结构，区别：Base 是 expert_dtype=fp8（走 AITER fp8 MoE，
dummy 冒烟已验证该链路），权重 275G。
"""
import os
import torch
from vllm import LLM, SamplingParams


def main():
    MODEL = "/root/models/DeepSeek-V4-Flash-Base"
    print(f"model={MODEL} exists={os.path.isdir(MODEL)}", flush=True)
    print(f"torch {torch.__version__} dev={torch.cuda.device_count()}", flush=True)

    llm = LLM(
        model=MODEL,
        served_model_name="DeepSeek-V4-Flash-Base",
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        moe_backend="triton",
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        block_size=256,
        disable_custom_all_reduce=True,
        trust_remote_code=False,
    )
    print("LLM 构造成功", flush=True)

    sp = SamplingParams(max_tokens=32, temperature=0.0)
    out = llm.generate(["用一句话解释什么是RDMA。"], sampling_params=sp)
    for o in out:
        print("output:", o.outputs[0].text[:300], flush=True)
    print("FLASHBASE-8CARD-DONE", flush=True)


if __name__ == "__main__":
    main()
