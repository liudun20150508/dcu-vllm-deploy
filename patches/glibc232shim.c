/* glibc232shim.c —— 提供 glibc 2.32 新增的 __libc_single_threaded 符号
 *
 * 【为什么需要】
 *   das185 的 vllm 二进制（_rocm_C / _moe_C / _C.abi3.so）链接了 glibc 2.32
 *   才有的 __libc_single_threaded@@GLIBC_2.32，而部分老集群（glibc 2.31）
 *   没有这个符号 → import 时 undefined symbol。这个 shim 用 LD_PRELOAD 提供它。
 *
 * 【编译】
 *   gcc -shared -fPIC -O2 -o libglibc232shim.so glibc232shim.c
 *   详见 scripts/install/build_glibc_shim.sh
 *
 * 【使用】
 *   export LD_PRELOAD=/path/to/libglibc232shim.so
 *
 * 注意：__libc_single_threaded 语义是「进程是否单线程」，glibc 用它跳过加锁。
 * 固定导出 1（单线程）在真实推理的 worker 进程里是安全的——每个 vllm worker
 * 是独立进程、主线程启动；若你在多线程里 fork，请改成 0 或链接真实 glibc 2.32。
 */
int __libc_single_threaded = 1;

/* 让 .so 导出带 GLIBC_2.32 版本标签的符号（vllm 二进制按版本化符号查找）。
 * .symver 是 GNU as 指令，仅 Linux gcc 支持；macOS 等平台跳过（本 shim 只在集群上编）。 */
#if defined(__linux__) && defined(__GNUC__)
__asm__(".symver __libc_single_threaded,__libc_single_threaded@@GLIBC_2.32");
#endif
