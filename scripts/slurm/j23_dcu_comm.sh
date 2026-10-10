#!/bin/bash
#SBATCH -J acc-j23-dcucomm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH -t 00:20:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j23-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j23-%j.err

# J23（优先 3 + 2 前置）：DCU 通信能力 —— 卡间 P2P / 显存注册 / dma-buf 导出
set -u
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
export HSA_FORCE_FINE_GRAIN_PCIE=1
PY=/usr/local/python3.12/bin/python3
echo "########## J23 on $(hostname) $(date) ##########"

echo "===== [1] DCU 卡间 P2P（peer access）====="
$PY - <<'EOF' 2>&1 | grep -v -E "^\s*$|Warning|warn" | tail -25
import torch
n = torch.cuda.device_count()
print(f"  DCU 数: {n}  ({torch.cuda.get_device_name(0)})")
ok = []
for i in range(n):
    for j in range(n):
        if i != j and torch.cuda.can_device_access_peer(i, j):
            ok.append((i, j))
print(f"  可 P2P 的卡对: {ok if ok else '无（全部经 host 中转）'}")
EOF

echo
echo "===== [2] 卡间拷贝带宽：P2P vs 经 host ====="
$PY - <<'EOF' 2>&1 | grep -a -E "^  (P2P|H2D|D2H|host)|Error" | head -10
import torch, time
def bw(fn, mb=256, iters=10):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return mb / 1024 / dt
n = torch.cuda.device_count()
if n >= 2:
    a = torch.empty(256*1024*1024//4, device="cuda:0")
    b = torch.empty_like(a, device="cuda:1")
    if torch.cuda.can_device_access_peer(0, 1):
        print(f"  P2P   D0->D1 : {bw(lambda: b.copy_(a)):.2f} GB/s")
    else:
        print(f"  P2P   不可用 —— 走 host 中转")
    print(f"  host  D0->CPU->D1 : {bw(lambda: torch.empty_like(a).copy_(a)):.2f} GB/s (单侧)")
else:
    print("  DCU < 2 张，跳过卡间拷贝")
EOF

echo
echo "===== [3] 显存能不能直接给 RDMA 注册（GPUDirect 可行性）====="
$PY - <<'EOF' 2>&1 | tail -25
import ctypes, ctypes.util, torch, os
torch.cuda.init()
dev_ptr = torch.empty(4*1024*1024, dtype=torch.uint8, device="cuda").data_ptr()
print(f"  DCU 显存地址: 0x{dev_ptr:x}")

# --- libibverbs ---
path = ctypes.util.find_library("ibverbs") or "libibverbs.so.1"
v = ctypes.CDLL(path)
v.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
v.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
v.ibv_open_device.restype = ctypes.c_void_p
v.ibv_open_device.argtypes = [ctypes.c_void_p]
v.ibv_alloc_pd.restype = ctypes.c_void_p
v.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
v.ibv_reg_mr.restype = ctypes.c_void_p
v.ibv_reg_mr.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

cnt = ctypes.c_int(0)
lst = v.ibv_get_device_list(ctypes.byref(cnt))
print(f"  verbs 设备数: {cnt.value}")
dev = lst[0]; ctx = v.ibv_open_device(dev); pd = v.ibv_alloc_pd(ctx)
print(f"  pd = {'OK' if pd else 'FAIL'}")

IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_WRITE, IBV_ACCESS_REMOTE_READ = 1, 2, 4
acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
mr = v.ibv_reg_mr(pd, ctypes.c_void_p(dev_ptr), 4*1024*1024, acc)
if mr:
    print("  ★ 显存直接 ibv_reg_mr：成功 → 该网卡可对 DCU 显存做 RDMA（GPUDirect 通路存在）")
else:
    err = ctypes.get_errno()
    print(f"  ✗ 显存直接 ibv_reg_mr：失败 (errno={err}) —— 需走 dma-buf 路径")

# --- dma-buf 导出 API 是否存在 ---
try:
    hip = ctypes.CDLL("libamdhip64.so")
    for sym in ("hipMemGetHandleForAddressRange", "hipMemExportToShareableHandle", "hipMemGetHandleForAddress"):
        print(f"  hip {sym}: {'存在' if hasattr(hip, sym) else '不存在'}")
except Exception as e:
    print("  libamdhip64 加载失败:", e)
EOF

echo
echo "===== [4] 系统侧的 P2P 通路（PCIe 拓扑）====="
echo "-- DCU 的 PCIe 地址 --"
for i in 0 1; do
  p=$(ls -d /sys/class/drm/card$i/device 2>/dev/null)
  [ -n "$p" ] && echo "  card$i -> $(readlink -f $p)"
done
echo "-- shca 网卡的 PCIe 地址 --"
ls -d /sys/class/infiniband/shca_*/device 2>/dev/null | head -4 | while read d; do echo "  $(readlink -f $d)"; done
echo "-- IOMMU 状态（P2P 能否直通）--"
dmesg 2>/dev/null | grep -i -E "iommu|DMAR" | head -3 || echo "  (dmesg 不可读)"
cat /sys/class/iommu/*/type 2>/dev/null | head -2 || true
lspci -t 2>/dev/null | head -12 || echo "  (无 lspci)"
echo "########## J23 done $(date) ##########"
