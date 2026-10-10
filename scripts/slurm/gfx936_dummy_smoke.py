"""gfx936 单卡 dummy 冒烟：vllm_hcu 加载 DeepSeek-V4-Flash-0731 结构（dummy 权重）+ 前向。

复用真实 config.json，缩小 num_hidden_layers 控制显存；权重用 --load-format dummy
（vllm 内置随机权重，不读 safetensors），验证 gfx936 上模型结构加载 + 前向链路。
"""
import os
import json
import shutil


def main():
    SRC = "/root/private_data/models/DeepSeek-V4-Flash-0731"
    DUMMY = "/root/private_data/models/dummy-v4-flash"
    os.makedirs(DUMMY, exist_ok=True)

    # 1) 复制 config + tokenizer，缩小规模
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    orig_layers = cfg["num_hidden_layers"]
    cfg["num_hidden_layers"] = 1
    # 单卡 64GB 冒烟：缩小 MoE 专家规模，否则 dummy 权重本身占 ~48GB 导致 decode OOM。
    cfg["n_routed_experts"] = 16
    cfg["moe_intermediate_size"] = 512
    # dummy 冒烟用 fp8 专家（避开 fp4/MXFP4 的 CK MoE dtype 门槛），
    # 只验证 gfx936 加载 + 前向链路；fp4 权重另行验证。
    cfg["expert_dtype"] = "fp8"
    json.dump(cfg, open(os.path.join(DUMMY, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(SRC, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(DUMMY, f))
    print(f"dummy config: layers {orig_layers} -> {cfg['num_hidden_layers']}")

    # 2) vllm 加载（dummy 权重）
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=DUMMY,
        load_format="dummy",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=128,
        gpu_memory_utilization=0.15,
        trust_remote_code=False,
        disable_custom_all_reduce=True,
        kv_cache_dtype="fp8",
        block_size=256,
    )
    print("LLM 构造成功")

    # 3) 前向
    sp = SamplingParams(max_tokens=16, temperature=0.0)
    out = llm.generate(["人工智能是"], sampling_params=sp)
    for o in out:
        print("output:", o.outputs[0].text[:200])
    print("DUMMY-SMOKE-DONE")


if __name__ == "__main__":
    main()
