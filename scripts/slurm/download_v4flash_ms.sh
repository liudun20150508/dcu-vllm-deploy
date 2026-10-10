#!/bin/bash
# 从 ModelScope 补全 DeepSeek-V4-Flash（preview）权重
# 只下载本地缺失/大小不符的文件，已完整的自动跳过（可断点续传）
set -u
BASE="https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash/resolve/master"
DEST="$HOME/models/DeepSeek-V4-Flash"
API="https://www.modelscope.cn/api/v1/models/deepseek-ai/DeepSeek-V4-Flash/repo/files?Revision=master&Recursive=true"
LIST=/tmp/ms_files.json
MANIFEST=/tmp/v4flash_missing.tsv
LOG="$DEST/.download.log"
mkdir -p "$DEST"

echo "=== 开始 $(date '+%F %T') ===" >> "$LOG"

# 1. 拉最新文件清单
curl -sSL "$API" -o "$LIST" || { echo "拉清单失败" >> "$LOG"; exit 1; }

# 2. 生成缺失文件清单：size<TAB>path
python3 - "$LIST" "$DEST" <<'PY' > "$MANIFEST"
import sys, json, os
d = json.load(open(sys.argv[1]))
dest = sys.argv[2]
files = d.get('Data', {}).get('Files', [])
out = []
for f in files:
    p = f.get('Path', '')
    s = f.get('Size', 0) or 0
    if s <= 0 or not p or p.endswith('/'):
        continue
    lp = os.path.join(dest, p)
    if os.path.exists(lp) and os.path.getsize(lp) == s:
        continue
    out.append((s, p))
out.sort(key=lambda x: (0 if 'safetensors' not in x[1] else 1, x[1]))
for s, p in out:
    print("%d\t%s" % (s, p))
PY

total=$(wc -l < "$MANIFEST")
echo "待下载文件数: $total" >> "$LOG"
if [ "$total" -eq 0 ]; then
  echo "无缺失文件，已完整" >> "$LOG"
  exit 0
fi

# 3. 逐个 curl 下载（-C - 断点续传，--retry 重试）
i=0
while IFS=$'\t' read -r size path; do
  [ -z "$path" ] && continue
  i=$((i+1))
  mkdir -p "$DEST/$(dirname "$path")"
  gb=$(python3 -c "print('%.2f' % ($size/1024/1024/1024))")
  echo "[$i/$total] $path ($gb GB) $(date '+%T')" >> "$LOG"
  curl -sSL --retry 3 --retry-delay 3 -C - -o "$DEST/$path" "$BASE/$path" >> "$LOG" 2>&1
  got=$(stat -c%s "$DEST/$path" 2>/dev/null || echo 0)
  if [ "$got" = "$size" ]; then
    echo "    OK $path" >> "$LOG"
  else
    echo "    !!! 大小不符 $path 期望=$size 实得=$got" >> "$LOG"
  fi
done < "$MANIFEST"

echo "=== 结束 $(date '+%F %T') ===" >> "$LOG"
