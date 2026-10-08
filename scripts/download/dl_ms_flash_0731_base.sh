#!/bin/bash
# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
WS=${WS:-/path/to/dcu-vllm-deploy}        # 仓库根
MODELS=${MODELS:-$HOME/models}            # 模型权重目录
DAS185=${DAS185:-$HOME/das185}            # das185 环境
WORK=${WORK:-$HOME/work}                  # 大容量工作盘
# ============================================================
# 从 ModelScope 下载 DeepSeek-V4-Flash-0731 + DeepSeek-V4-Flash-Base → $MODELS（overlay 大盘）
#   hf-mirror 被限流到 ~70KB/s，ModelScope 实测 ~10.7MB/s，改用本脚本。
#   逻辑同 dl_flash_0731_base.sh：尺寸一致 SKIP；不符删了重下；先下到 .part 再改名；
#   每个模型内部并行 PAR 路（默认 4）。可重跑补漏。
set -u
NS=deepseek-ai
API="https://modelscope.cn/api/v1/models"
ROOT=$MODELS
PAR=${PAR:-4}
PY=$(command -v python3 || command -v python)
mkdir -p "$ROOT"

dl_model() {
  local M="$1"
  local D="$ROOT/$M"
  local LOG="$ROOT/dl_ms_${M}.log"
  local TMP="/tmp/ms_sizes_$$.txt"
  mkdir -p "$D"

  echo "[$(date '+%F %T')] === 开始 $M → $D (并行 $PAR) ===" | tee -a "$LOG"

  curl -skL --connect-timeout 20 --max-time 60 "$API/$NS/$M/repo/files?Revision=master&Root=/" | "$PY" -c '
import sys, json
d = json.load(sys.stdin)
if not isinstance(d, dict) or "Files" not in d.get("Data", {}):
    sys.stderr.write("API错误: %r\n" % d); sys.exit(1)
for f in d["Data"]["Files"]:
    if f.get("Type") == "blob":
        print(f["Path"], f["Size"])
' > "$TMP" || { echo "[$(date '+%F %T')] FAIL 拿不到 $M 文件清单" | tee -a "$LOG"; return 1; }

  local TOTAL
  TOTAL=$(wc -l < "$TMP")
  echo "[$(date '+%F %T')] 清单: 共 $TOTAL 个文件" | tee -a "$LOG"

  export D URL="$API/$NS/$M/repo"
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
    if curl -skL --retry 5 --retry-delay 3 --connect-timeout 20 -o "$part" "$URL?Revision=master&FilePath=$path"; then
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
  ' _ < "$TMP" >> "$LOG" 2>&1

  # 最终全量校验
  local missing=0
  while read -r path size; do
    cur=$(stat -c%s "$D/$path" 2>/dev/null || echo 0)
    if [ "$cur" != "$size" ]; then
      echo "[$(date '+%F %T')] 缺失/不符: $path 实=$cur 期望=$size" >> "$LOG"
      missing=1
    fi
  done < "$TMP"
  rm -f "$TMP"

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
