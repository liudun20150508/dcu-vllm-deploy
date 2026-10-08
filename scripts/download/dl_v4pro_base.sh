#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
# 后台下载 DeepSeek-V4-Pro-Base 权重 (约 1.46 TB, 64 个分片, FP8) —— 走 hf-mirror
#   逻辑同 dl_v4flash_base.sh：尺寸一致跳过；不符删了重下；先下到 .part 再改名。
#   与 Flash-Base 不同：并行 PAR 路（默认 4），否则 1.46TB 串行要 18h+。
#   可重跑补漏：已完整的文件会 SKIP。
set -u
M=DeepSeek-V4-Pro-Base
D=$HOME/models/$M
B=https://hf-mirror.com/deepseek-ai/$M/resolve/main
LOG=$HOME/models/dl_v4pro_base.log
PAR=${PAR:-4}
mkdir -p "$D"

echo "[$(date '+%F %T')] === 开始下载 $M (并行 $PAR) ===" >> "$LOG"

curl -sk "https://hf-mirror.com/api/models/deepseek-ai/$M/tree/main" | /usr/local/python3.12/bin/python3 -c "
import sys,json
for f in json.load(sys.stdin):
    if f.get('type')=='file':
        print(f['path'], f['size'])
" > /tmp/v4probase_sizes.$$.txt

TOTAL=$(wc -l < /tmp/v4probase_sizes.$$.txt)
echo "[$(date '+%F %T')] 清单: 共 $TOTAL 个文件" >> "$LOG"

export D B
xargs -P "$PAR" -n 2 bash -c '
  path="$1"; size="$2"
  dest="$D/$path"; part="$dest.part"
  cur=$(stat -c%s "$dest" 2>/dev/null || echo 0)
  if [ "$cur" = "$size" ]; then
    echo "[$(date "+%F %T")] SKIP $path (已完整)"
    exit 0
  fi
  [ "$cur" != "0" ] && rm -f "$dest"
  rm -f "$part"
  if curl -skfL --retry 5 --retry-delay 5 -o "$part" "$B/$path"; then
    new=$(stat -c%s "$part" 2>/dev/null || echo 0)
    if [ "$new" = "$size" ]; then
      mv "$part" "$dest"
      echo "[$(date "+%F %T")] OK   $path ($new)"
      exit 0
    fi
  fi
  rm -f "$part"
  echo "[$(date "+%F %T")] FAIL $path (需重跑本脚本补)"
  exit 1
' _ < /tmp/v4probase_sizes.$$.txt >> "$LOG" 2>&1

rm -f /tmp/v4probase_sizes.$$.txt

# 最终全量校验
missing=0
curl -sk "https://hf-mirror.com/api/models/deepseek-ai/$M/tree/main" | /usr/local/python3.12/bin/python3 -c "
import sys,json
for f in json.load(sys.stdin):
    if f.get('type')=='file':
        print(f['path'], f['size'])
" > /tmp/v4probase_check.$$.txt
while read -r path size; do
  cur=$(stat -c%s "$D/$path" 2>/dev/null || echo 0)
  if [ "$cur" != "$size" ]; then
    echo "[$(date '+%F %T')] 缺失/不符: $path 实=$cur 期望=$size" >> "$LOG"
    missing=1
  fi
done < /tmp/v4probase_check.$$.txt
rm -f /tmp/v4probase_check.$$.txt

echo "[$(date '+%F %T')] === 结束 $M ===" >> "$LOG"
du -sh "$D" >> "$LOG"
if [ "$missing" = "0" ]; then
  echo "[$(date '+%F %T')] ✅ 全部文件校验通过" >> "$LOG"
else
  echo "[$(date '+%F %T')] ⚠️ 有缺失，重跑本脚本即可补" >> "$LOG"
fi
