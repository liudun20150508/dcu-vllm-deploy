"""8 卡 gfx936 真实权重冒烟：DeepSeek-V4-Flash-0731（156G fp4）TP=8 加载 + 前向。

基于 gfx936 单卡 dummy 冒烟（gfx936_dummy_smoke.py）+ run_serve_node.sh 的坑位经验，
区别：真实 safetensors 权重 + expert_dtype=fp4（走 AITER CK MoE fp4_w4a16，补丁3已放开
gfx936/gfx938 门槛）+ tensor_parallel_size=8。
"""
import os
import torch
from vllm import LLM, SamplingParams


def main():
    MODEL = "/root/models/DeepSeek-V4-Flash-0731"
    print(f"model={MODEL} exists={os.path.isdir(MODEL)}", flush=True)
    print(f"torch {torch.__version__} dev={torch.cuda.device_count()}",
          flush=True)

    llm = LLM(
        model=MODEL,
        served_model_name="DeepSeek-V4-Flash-0731",
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        moe_backend="auto",
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
    print("FLASH0731-8CARD-DONE", flush=True)


if __name__ == "__main__":
    main()
