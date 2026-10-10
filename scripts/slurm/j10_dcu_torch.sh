#!/bin/bash
#SBATCH -J acc-j10-dcu
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH -t 00:15:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j10-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j10-%j.err

# J10：让海光版 torch 认出 DCU —— 诊断 + 逐版本试
set -u
module purge
source /etc/profile.d/modules.sh
echo "########## J10 on $(hostname) $(date) ##########"
PY=/usr/local/python3.12/bin/python3
echo "PY = $PY"; $PY --version

echo "===== [1] 设备节点 / 驱动 ====="
ls -l /dev/kfd 2>&1
ls /dev/dri/ 2>&1
lsmod 2>/dev/null | grep -i -E "amdgpu|hydcu|hycu" || echo "(lsmod 无匹配)"
echo "-- amdgpu.ids 在哪 --"
for f in /opt/hyhal/share/amdgpu.ids /usr/share/libdrm/amdgpu.ids \
         /public/software/compiler/dtk-26.04/share/amdgpu.ids \
         /public/software/compiler/dtk-25.04.4/share/amdgpu.ids; do
  printf "%-58s %s\n" "$f" "$([ -f "$f" ] && echo YES || echo no)"
done
timeout 40 find /opt/hyhal /public/software/compiler -maxdepth 4 -name "amdgpu.ids" 2>/dev/null | head

echo
echo "===== [2] 逐个 DTK 版本试 is_available ====="
for dtk in compiler/dtk/25.04.4 compiler/dtk/25.04 compiler/dtk/26.04; do
  module purge >/dev/null 2>&1
  source /etc/profile.d/modules.sh
  if ! module load $dtk >/dev/null 2>&1; then echo "### $dtk 加载失败，跳过"; continue; fi
  echo "### $dtk"
  echo "  DTKROOT=$DTKROOT  ROCM_PATH=${ROCM_PATH:-}  HIP_PATH=${HIP_PATH:-}"
  echo "  libamdhip64 soname:"
  ls $DTKROOT/hip/lib/libamdhip64.so* 2>/dev/null | head -3 | sed 's/^/    /'
  ls $DTKROOT/lib/libhsa-runtime64.so* 2>/dev/null | head -3 | sed 's/^/    /'
  export HSA_FORCE_FINE_GRAIN_PCIE=1
  $PY - <<'EOF' 2>&1 | sed 's/^/  /'
import os, torch
print("torch", torch.__version__, "hip", torch.version.hip)
print("LD_LIBRARY_PATH 里的 dtk:", [p for p in os.environ.get("LD_LIBRARY_PATH","").split(":") if "dtk" in p][:3])
try:
    print("is_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("EXC:", type(e).__name__, e)
EOF
done

echo
echo "===== [3] 直接测 HIP 运行时（绕开 torch）====="
module purge >/dev/null 2>&1; source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
echo "-- rocminfo --"; timeout 30 rocminfo 2>&1 | head -12
echo "-- 用 python3.12 直接调 libamdhip64 --"
$PY - <<'EOF' 2>&1 | tail -12
import ctypes, ctypes.util, os
p = ctypes.util.find_library("amdhip64")
print("find_library:", p)
for cand in [os.environ.get("ROCM_PATH","")+"/hip/lib/libamdhip64.so",
             os.environ.get("HIP_PATH","")+"/lib/libamdhip64.so"]:
    if cand and os.path.exists(cand):
        p = cand; print("use:", p); break
try:
    lib = ctypes.CDLL(p)
    n = ctypes.c_int(0)
    r = lib.hipGetDeviceCount(ctypes.byref(n))
    print("hipGetDeviceCount rc =", r, "count =", n.value)
    err = lib.hipGetErrorString(r); print("err string:", ctypes.c_char_p(err).value)
except Exception as e:
    print("EXC:", type(e).__name__, e)
EOF
echo "########## J10 done $(date) ##########"
