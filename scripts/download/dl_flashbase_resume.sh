#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
# 只补 DeepSeek-V4-Flash-Base 缺失的 model-00033-of-00046.safetensors
# 逻辑与 ~/dl_v4flash_base.sh 一致：先查官方树拿期望尺寸，删残留，再下载并校验。
set -u
PY=/usr/local/python3.12/bin/python3
[ -x "$PY" ] || PY=python3

M=DeepSeek-V4-Flash-Base
P=model-00033-of-00046.safetensors
D=$HOME/models/$M
B=https://hf-mirror.com/deepseek-ai/$M/resolve/main
API=https://hf-mirror.com/api/models/deepseek-ai/$M/tree/main

SIZE=$(curl -sk "$API" | "$PY" -c "
import sys, json
for f in json.load(sys.stdin):
    if f.get('path') == '$P':
        print(f['size'])
        break
")
if [ -z "$SIZE" ]; then
  echo "FAIL: 拿不到 $P 的期望大小（API 失败或文件不存在）"
  exit 1
fi
echo "期望大小: $SIZE"

cur=$(stat -c%s "$D/$P" 2>/dev/null || echo 0)
if [ "$cur" = "$SIZE" ]; then
  echo "已完整，无需下载: $P ($cur)"
  exit 0
fi
[ "$cur" != "0" ] && rm -f "$D/$P"   # 删掉 1.6MB 残留

curl -skL --retry 5 --retry-delay 5 -o "$D/$P" "$B/$P"
new=$(stat -c%s "$D/$P" 2>/dev/null || echo 0)
if [ "$new" = "$SIZE" ]; then
  echo "OK: $P ($new)"
  du -sh "$D"
else
  echo "FAIL: 实=$new 期望=$SIZE → 已删残留，请重跑本脚本"
  rm -f "$D/$P"
  exit 1
fi
