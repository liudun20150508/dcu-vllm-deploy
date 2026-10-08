#!/bin/bash
# ============================================================================
# check_hw.sh — 硬件探针：确认 DCU 架构 / 显存 / CU / warpSize / 卡数 / 网卡
#
# 用途：验证「前置条件 1（DCU 硬件）」是否满足。单机可跑，无需 Slurm。
# 用法：
#   source scripts/install/env.sh && bash scripts/verify/check_hw.sh
# 输出解读：
#   gcnArchName = gfx936  → 本框架主测目标，✅
#   gcnArchName = gfx906  → 老架构，部分算子不支持（见 SUPPORTED_HARDWARE.md）
#   gcnArchName = gfx928  → 未实测，需人工补充结果
# ============================================================================
set -u

echo "########## DCU 硬件探针 $(date) ##########"

echo ""
echo "===== [1] 架构（rocminfo）====="
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo 2>/dev/null | grep -iE "gfx|Compute Unit|Name:" | head -20
else
  echo "  rocminfo 不存在，尝试 hy-smi / rocm-smi"
  command -v hy-smi >/dev/null 2>&1 && hy-smi 2>&1 | head -20
  command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showproductname 2>&1 | head -10
fi

echo ""
echo "===== [2] 显存（权威口径：sysfs）====="
# 每个 card 一个 mem_info_vram_total，单位字节
found=0
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
  [ -f "$f" ] || continue
  bytes=$(cat "$f" 2>/dev/null)
  gib=$(awk -v b="$bytes" 'BEGIN{printf "%.1f", b/1024/1024/1024}')
  echo "  $(basename "$(dirname "$(dirname "$f")")")  $gib GiB  ($bytes)"
  found=1
done
[ "$found" = 0 ] && echo "  sysfs 无 mem_info_vram_total，用 rocm-smi --showmeminfo vram 代替"

echo ""
echo "===== [3] CU 数 / warpSize / 卡数（torch 口径）====="
PY="${PYTHON_BIN:-python3}"
"$PY" - <<'EOF' 2>&1 | sed 's/^/  /'
try:
    import torch
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() = False（DTK 未加载或装的是 AMD 官方 wheel）")
        raise SystemExit
    n = torch.cuda.device_count()
    print("device_count =", n)
    p = torch.cuda.get_device_properties(0)
    print("gcnArchName =", p.gcnArchName)
    print("name        =", p.name)
    print("total_mem   = %.1f GiB" % (p.total_memory / 1024**3))
    print("multiProcessor (CU) =", getattr(p, "multi_processor_count", "?"))
    # warpSize 由 torch 版本决定是否暴露
    print("warpSize 见 torch._C / 默认 wave64（海光/AMD）")
except Exception as e:
    print("torch probe failed:", repr(e))
EOF

echo ""
echo "===== [4] 每节点卡数（Slurm 集群才有）====="
if command -v sinfo >/dev/null 2>&1; then
  sinfo -o "%P %D %c %m %G" 2>/dev/null | head -10 || echo "  sinfo 不可用"
else
  echo "  非 Slurm 环境，跳过"
fi

echo ""
echo "===== [5] 网卡（shca 自研 vs Mellanox mlx5）====="
if command -v ibstat >/dev/null 2>&1; then
  ibstat 2>/dev/null | grep -iE "^CA |Rate|Link layer|State:" | head -30
elif command -v ibv_devices >/dev/null 2>&1; then
  ibv_devices 2>&1 | head -20
else
  echo "  无 ibstat/ibv_devices"
fi

echo ""
echo "===== [6] DTK 版本 ====="
echo "  DTKROOT = ${DTKROOT:-<未设置，先 source scripts/install/env.sh>}"
[ -n "${DTKROOT:-}" ] && ls -d "$DTKROOT" 2>/dev/null

echo ""
echo "########## 探针结束 ##########"
echo "对照 SUPPORTED_HARDWARE.md：gfx936=✅主测目标 / gfx906=老架构部分支持 / gfx928=未实测"
