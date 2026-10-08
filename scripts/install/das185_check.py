# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
import os
WS = os.environ.get("WS", "/path/to/dcu-vllm-deploy")   # 仓库根
MODELS = os.environ.get("MODELS", os.path.expanduser("~/models"))  # 模型权重目录
DAS185 = os.environ.get("DAS185", os.path.expanduser("~/das185"))  # das185 环境
WORK = os.environ.get("WORK", os.path.expanduser("~/work"))        # 大容量工作盘
# ============================================================
"""das185 环境探针：torch 认卡 + 关键模块 import。"""
import sys
print("python", sys.version.split()[0])
import torch
import numpy
print("torch", torch.__version__, "numpy", numpy.__version__)
print("hip", torch.version.hip, "avail", torch.cuda.is_available(),
      "dev", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

for name in ("vllm", "vllm_hcu", "aiter", "triton", "tilelang", "flash_mla"):
    try:
        m = __import__(name)
        print("OK  ", name, getattr(m, "__version__", "?"))
    except Exception as e:
        print("FAIL", name, type(e).__name__, str(e)[:140])
