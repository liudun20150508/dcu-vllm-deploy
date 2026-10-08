#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
# 下载 DeepSeek-V4-Flash-0731 + DeepSeek-V4-Flash-Base 真实权重 → $MODELS（本地 overlay 大盘）
#   逻辑同 dl_v4pro_base.sh：尺寸一致跳过；不符删了重下；先下到 .part 再改名；
#   每个模型内部并行 PAR 路（默认 4）。可重跑补漏：已完整的文件会 SKIP。
#   ★ 目标必须是本地 overlay（1.2T），持久盘只有 50G 放不下 153G/295G。
set -u
API=https://hf-mirror.com/api/models/deepseek-ai
BASE=https://hf-mirror.com/deepseek-ai
ROOT=$MODELS
PAR=${PAR:-4}
PY=$(command -v python3 || command -v python)
mkdir -p "$ROOT"

dl_model() {
  local M="$1"
  local D="$ROOT/$M"
  local LOG="$ROOT/dl_${M}.log"
  local TMP_SIZES="/tmp/dl_sizes_$$.txt"
  mkdir -p "$D"

  echo "[$(date '+%F %T')] === 开始 $M → $D (并行 $PAR) ===" | tee -a "$LOG"

  curl -sk "$API/$M/tree/main" | "$PY" -c "
import sys,json
d=json.load(sys.stdin)
if isinstance(d,dict):
    sys.stderr.write('API错误: %r\n' % d); sys.exit(1)
for f in d:
    if f.get('type')=='file':
        print(f['path'], f['size'])
" > "$TMP_SIZES" || { echo "[$(date '+%F %T')] FAIL 拿不到 $M 文件清单" | tee -a "$LOG"; return 1; }

  local TOTAL
  TOTAL=$(wc -l < "$TMP_SIZES")
  echo "[$(date '+%F %T')] 清单: 共 $TOTAL 个文件" | tee -a "$LOG"

  export D B="$BASE/$M/resolve/main"
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
    echo "[$(date "+%F %T")] FAIL $path (需重跑补漏)"
    exit 1
  ' _ < "$TMP_SIZES" >> "$LOG" 2>&1

  # 最终全量校验
  local missing=0
  curl -sk "$API/$M/tree/main" | "$PY" -c "
import sys,json
for f in json.load(sys.stdin):
    if f.get('type')=='file':
        print(f['path'], f['size'])
" > "$TMP_SIZES"
  while read -r path size; do
    cur=$(stat -c%s "$D/$path" 2>/dev/null || echo 0)
    if [ "$cur" != "$size" ]; then
      echo "[$(date '+%F %T')] 缺失/不符: $path 实=$cur 期望=$size" >> "$LOG"
      missing=1
    fi
  done < "$TMP_SIZES"
  rm -f "$TMP_SIZES"

  echo "[$(date '+%F %T')] === 结束 $M ===" >> "$LOG"
  du -sh "$D" >> "$LOG"
  if [ "$missing" = "0" ]; then
    echo "[$(date '+%F %T')] ✅ $M 全部文件校验通过" | tee -a "$LOG"
  else
    echo "[$(date '+%F %T')] ⚠️ $M 有缺失，重跑本脚本即可补漏" | tee -a "$LOG"
  fi
}

for M in DeepSeek-V4-Flash-0731 DeepSeek-V4-Flash-Base; do
  dl_model "$M"
done
echo "[$(date '+%F %T')] ALL DONE"
