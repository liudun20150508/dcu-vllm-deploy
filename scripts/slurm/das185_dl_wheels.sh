#!/bin/bash
# 下载 das185 (DAS1.8.5 / torch2.10 / cp310) 全家桶 wheel + dtk_llvm.run
# 来源：光合社区 download.sourcefind.cn:65024/directlink/4
# 断点续传 + 尺寸校验（curl -C -），可重复运行补漏。目标 /root/das185/wheels。
set -u
BASE=https://download.sourcefind.cn:65024/directlink/4
D=/root/das185/wheels
mkdir -p "$D"
LOG=$D/dl.log
echo "[$(date '+%F %T')] === 开始下载 das185 全家桶 ===" | tee -a "$LOG"

FILES=(
'pytorch/DAS1.8.5/torch-2.10.0+das.opt1.dtk2604.2608131652.g28498c-cp310-cp310-manylinux_2_28_x86_64.whl'
'vllm/DAS1.8.5/vllm-0.21.0+das.dtk2604.torch2100.2606111143.g8c979d-cp310-cp310-linux_x86_64.whl'
'vllm_hcu/DAS1.8.5/vllm_hcu-0.21.0+das.dtk2604.torch2100.2608132222.g386d02-cp310-cp310-linux_x86_64.whl'
'aiter/DAS1.8.5/aiter-0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
'triton/DAS1.8.5/triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
'tilelang/DAS1.8.5/tilelang-0.1.9+das185.dtk2604.torch2100.2608121523.gf631c9-cp38-abi3-linux_x86_64.whl'
'flash_mla/DAS1.8.5/flash_mla-1.2.0+das185.dtk2604.torch2100.2608131631.gc33d94-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl'
'vision/DAS1.8.5/torchvision-0.25.0+das185.dtk2604.torch2100.2608130858.gdcd044-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
'lightop/DAS1.8.5/lightop-0.6.0+das.dtk2604.torch2100.2608121706.g09a4f3-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
'deepgemm/DAS1.8.5/deepgemm-2.1.0+das185.dtk2604.torch2100.2608121826.g2b4d4e-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl'
'dtk_llvm/dtk_llvm.run'
)

for f in "${FILES[@]}"; do
  name=$(basename "$f")
  dest="$D/$name"
  size=$(curl -skI "$BASE/$f" 2>/dev/null | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{print $2}' | tail -1)
  cur=$(stat -c%s "$dest" 2>/dev/null || echo 0)
  if [ -n "$size" ] && [ "$cur" = "$size" ]; then
    echo "[$(date '+%F %T')] SKIP $name (已完整 $cur)" | tee -a "$LOG"
    continue
  fi
  echo "[$(date '+%F %T')] 下载 $name (期望 $size) ..." | tee -a "$LOG"
  curl -skfL -C - --retry 5 --retry-delay 5 -o "$dest" "$BASE/$f"
  rc=$?
  new=$(stat -c%s "$dest" 2>/dev/null || echo 0)
  if [ $rc -eq 0 ] && { [ -z "$size" ] || [ "$new" = "$size" ]; }; then
    echo "    OK ($new)" | tee -a "$LOG"
  else
    echo "    FAIL rc=$rc 实=$new 期望=$size" | tee -a "$LOG"
  fi
done

echo "[$(date '+%F %T')] === 下载结束 ===" | tee -a "$LOG"
ls -la "$D" | tee -a "$LOG"
