#!/bin/bash
# ============================================================================
# build_glibc_shim.sh — 编译 glibc 2.32 符号 shim（老集群 glibc 2.31 需要）
#
# 背景：das185 的 vllm .so 链接了 __libc_single_threaded@@GLIBC_2.32，
# 老集群 glibc 2.31 没有这个符号。编译出的 libglibc232shim.so 用 LD_PRELOAD 注入。
#
# 用法：
#   bash scripts/install/build_glibc_shim.sh [输出目录]
#   默认输出到 $HOME/glibc-shim/libglibc232shim.so（各 serve 脚本里 SHIM 的默认路径）
#
# 前提：gcc（任意版本都可，只编一个符号导出 .so）
# ============================================================================
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$HOME/glibc-shim}"
SRC="$WS/patches/glibc232shim.c"

mkdir -p "$OUT"
echo ">> 编译 glibc shim: $SRC → $OUT/libglibc232shim.so"
gcc -shared -fPIC -O2 -o "$OUT/libglibc232shim.so" "$SRC"

echo ">> 验证导出符号："
nm -D "$OUT/libglibc232shim.so" | grep -E "__libc_single_threaded" || {
  echo "  ✗ 未找到符号，编译可能失败"; exit 1; }

echo ">> 完成。使用方式："
echo "   export LD_PRELOAD=$OUT/libglibc232shim.so"
echo "   （serve 脚本 run_serve_node.sh / run_v4pro_node.sh 已内置该行，SHIM 路径默认 \$HOME/glibc-shim）"
