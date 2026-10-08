# ==== 路径配置（脱敏：改成你的实际路径，或用环境变量覆盖）====
import os
WS = os.environ.get("WS", "/path/to/dcu-vllm-deploy")   # 仓库根
MODELS = os.environ.get("MODELS", os.path.expanduser("~/models"))  # 模型权重目录
DAS185 = os.environ.get("DAS185", os.path.expanduser("~/das185"))  # das185 环境
WORK = os.environ.get("WORK", os.path.expanduser("~/work"))        # 大容量工作盘
# ============================================================
# 坑12/14 修复（作业级注入，不改磁盘上的第三方代码）
#
# 根因：vllm 0.21.0 官方自带 DeepSeek-V4 实现，与 vllm_hcu 的实现撞车，同一进程内
# 两边都被 import。冲突分两层：
#   层1 custom_op.py:72   PluggableLayer.register 同名 op → AssertionError: Duplicate op name
#   层2 torch_utils.py:928 direct_register_custom_op("deepseek_v4_attention")
#                          → RuntimeError: operator registered multiple times
# 官方模块是被这条链拉进来的（作业 123548244 抓栈实测）：
#   config/model.py:997 _verify_quantization
#   → quantization/__init__.py:119 get_quantization_config
#     from vllm.model_executor.models.deepseek_v4 import DeepseekV4FP8Config
#   → vllm/model_executor/models/deepseek_v4.py:21 → 官方 attention
#
# 治本：给 vllm_hcu 的 MODULE_EXCHANGE_MAP 补两条交换（它本来就干"官方名→hcu 实现"
# 这件事，只是漏了 deepseek_v4），官方 v4 模块名直接解析到 hcu 实现 → 只注册一次，
# 两层冲突同时消失。hcu 侧确实有 DeepseekV4FP8Config（models/deepseek_v4.py:125）。
# 兜底：仍包装 PluggableLayer.register，若真撞上则让 hcu 覆盖官方（DCU 必须跑 hcu）。
#
# 坑15 实验开关：DUPFIX_FORCE_AITER=1 时，把 vllm._aiter_ops 的
# is_aiter_found_and_supported() 恒改为 True（gfx906 老卡硬跑 AITER MoE/量化）。
# hcu 的门槛是 on_gfx93x()（gfx936/gfx938），gfx906 上恒 False → MXFP4 无后端；
# 环境里装的是海光定制 aiter（0.1.5+das185.dtk2604），可能真支持 gfx906、只是门槛不认。
# 默认关（不设变量行为与原来完全一致），只在需要时作业脚本里 export 打开。
import os
import sys

TARGET_OP = "vllm.model_executor.custom_op"
TARGET_PU = "vllm_hcu.patch_utils"
TARGET_TU = "vllm.utils.torch_utils"
TARGET_AO = "vllm._aiter_ops"
TARGET_ST = "safetensors.torch"
TARGET_RC = "vllm.platforms.rocm"
TARGET_HP = "vllm_hcu.platforms.hcu"
TARGET_TB = "triton.backends.amd.compiler"
TARGET_TM = "vllm.model_executor.layers.fused_moe.experts.triton_moe"
MARK_OP = "_dupfix_op_installed"
MARK_PU = "_dupfix_pu_installed"
MARK_TU = "_dupfix_tu_installed"
MARK_AO = "_dupfix_ao_installed"
MARK_ST = "_dupfix_st_installed"
MARK_RC = "_dupfix_rc_installed"
HCU = "vllm_hcu"
OFFICIAL_V4_ATTN = "vllm.model_executor.layers.deepseek_v4_attention"
ENV_FORCE_AITER = "DUPFIX_FORCE_AITER"
ENV_FORCE_FP8 = "DUPFIX_FORCE_FP8"

# 官方名 -> hcu 实现（补 vllm_hcu 漏掉的 deepseek_v4 两条）
EXTRA_EXCHANGE = {
    "vllm.model_executor.models.deepseek_v4": "vllm_hcu.models.deepseek_v4",
    "vllm.model_executor.layers.deepseek_v4_attention":
        "vllm_hcu.model_executor.layers.deepseek_v4_attention",
}


def _log(msg):
    try:
        sys.stderr.write("[DUP-FIX] " + msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _is_hcu(cls):
    return (getattr(cls, "__module__", "") or "").startswith(HCU)


def _hcu_loaded():
    for k in sys.modules:
        if k == HCU or k.startswith(HCU + "."):
            return True
    return False


def _install_op(mod):
    """兜底：op_registry 冲突时让 hcu 胜出。"""
    PL = getattr(mod, "PluggableLayer", None)
    if PL is None or getattr(PL, MARK_OP, False):
        return
    raw = PL.__dict__.get("register")
    if raw is None:
        return
    orig = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
    registry = mod.__dict__.get("op_registry", {})

    def register(cls, name, *a, **k):
        dec = orig(cls, name, *a, **k)

        def wrapper(op_cls):
            prev = registry.get(name)
            if prev is None:
                return dec(op_cls)
            if _is_hcu(op_cls) and not _is_hcu(prev):
                op_cls.name = name
                registry[name] = op_cls
                _log("OVERRIDE-OP name=%s %s.%s -> %s.%s (pid=%d)"
                     % (name, prev.__module__, prev.__name__,
                        op_cls.__module__, op_cls.__name__, os.getpid()))
                return op_cls
            _log("SKIP-OP name=%s keep=%s.%s ignore=%s.%s (pid=%d)"
                 % (name, prev.__module__, prev.__name__,
                    op_cls.__module__, op_cls.__name__, os.getpid()))
            return op_cls

        return wrapper

    PL.register = classmethod(register)
    setattr(PL, MARK_OP, True)
    _log("op hook installed pid=%d on %s" % (os.getpid(), mod.__name__))


def _install_pu(mod):
    """不再用 MODULE_EXCHANGE_MAP：实测补 v4 两条会触发循环导入
    （module_exchanges_all_ok 在模块 import 中途 pop + 重新 import hcu 的 v4
    attention → 'partially initialized module ... circular import'，随后
    sys.modules 状态被破坏 → KeyError）。改为在 torch op 层拦截官方注册。"""
    if getattr(mod, MARK_PU, False):
        return
    setattr(mod, MARK_PU, True)
    _log("patch_utils seen (no exchange injected) pid=%d" % os.getpid())


def _install_tu(mod):
    """第二层：torch custom op 同名重复 define 时改为幂等（跳过 define，仍覆盖 impl）。

    仅加 EXCHANGE 还不够：vllm_hcu 的 module_exchanges_all_ok 会
    `sys.modules.pop(impl)` 后重新 import，导致 hcu 的 attention 模块体执行两次，
    torch library 不允许同名 op 重复 define。两边 schema 完全相同，跳过 define
    后用同一份 hcu 实现覆盖 impl 是安全的。
    """
    if getattr(mod, MARK_TU, False):
        return
    orig = getattr(mod, "direct_register_custom_op", None)
    if orig is None:
        return
    infer = getattr(mod, "infer_schema", None)
    if infer is None:
        try:
            from torch.library import infer_schema as infer  # type: ignore
        except Exception:
            infer = None

    def patched(op_name, op_func, mutates_args=None, fake_impl=None,
                target_lib=None, dispatch_key=None, tags=()):
        # 官方 v4 attention 是被 _verify_quantization 的 lazy import 拉进来的，
        # 它注册的 torch op 与 hcu 完全同名同 schema。torch 不允许 python 注册的
        # kernel 被后来者覆盖，所以不能"先官方后 hcu"——直接让官方这次注册落空，
        # 只保留 hcu 的实现（hcu 平台下官方实现本来也不可用）。
        fn_mod = getattr(op_func, "__module__", "") or ""
        if fn_mod.startswith(OFFICIAL_V4_ATTN) and _hcu_loaded():
            _log("SKIP-OFFICIAL op=%s (from %s, hcu 实现会注册, pid=%d)"
                 % (op_name, fn_mod, os.getpid()))
            return
        if mutates_args is None:
            mutates_args = []
        if dispatch_key is None:
            from vllm.platforms import current_platform
            dispatch_key = current_platform.dispatch_key
        my_lib = target_lib if target_lib is not None else getattr(mod, "vllm_lib", None)
        if my_lib is None or infer is None:
            return orig(op_name, op_func, mutates_args, fake_impl,
                        target_lib, dispatch_key, tags)
        schema_str = infer(op_func, mutates_args=mutates_args)
        try:
            my_lib.define(op_name + schema_str, tags=tags)
        except RuntimeError as e:
            msg = str(e)
            if ("same name and overload" in msg or "Duplicate registration" in msg
                    or "multiple times" in msg):
                _log("REDEFINE-SKIP op=%s (already defined, pid=%d)" % (op_name, os.getpid()))
            else:
                raise
        try:
            my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
        except Exception as e:
            _log("impl override failed op=%s: %r" % (op_name, e))
        if fake_impl is not None:
            try:
                my_lib._register_fake(op_name, fake_impl)
            except Exception as e:
                _log("fake register failed op=%s: %r" % (op_name, e))

    mod.direct_register_custom_op = patched
    setattr(mod, MARK_TU, True)
    _log("torch_utils hook installed pid=%d" % os.getpid())


def _wrap_support(cls, label):
    """打印 MXFP4 后端 is_supported_config 的拒绝原因（官方只 logger.debug 出来）。"""
    if cls is None or getattr(cls, "_support_wrapped", False):
        return
    raw = cls.__dict__.get("is_supported_config")
    if raw is None:
        return
    orig = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw

    def patched(*a, **k):
        try:
            ok, reason = orig(*a, **k)
        except Exception as e:
            _log("SUPPORT %s raised %r (pid=%d)" % (label, e, os.getpid()))
            raise
        _log("SUPPORT %s supported=%s reason=%s (pid=%d)"
             % (label, ok, reason, os.getpid()))
        return ok, reason

    cls.is_supported_config = staticmethod(patched)
    setattr(cls, "_support_wrapped", True)


def _install_aiter(mod):
    """AiterExperts 两道处理：
    ① is_supported_config 是继承自基类的(不在 cls.__dict__ 里,旧 _wrap_support
       会落空),这里显式把打日志包装装到类上 → 每次支持检查的真实拒绝原因可见;
    ② DUPFIX_FORCE_AITER=1 时绕过 _supports_quant_scheme 里的 gfx950 门槛
       (MXFP4 权重的第二道架构门:vllm_hcu 没 patch 它,gfx906/gfx936 都恒 False)。
    """
    cls = getattr(mod, "AiterExperts", None)
    if cls is not None and not getattr(cls, "_support_wrapped", False):
        raw = cls.is_supported_config  # 属性查找,可拿到继承方法
        if raw is not None:
            orig = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw

            def patched(*a, **k):
                try:
                    ok, reason = orig(*a, **k)
                except Exception as e:
                    _log("SUPPORT AiterExperts raised %r (pid=%d)" % (e, os.getpid()))
                    raise
                _log("SUPPORT AiterExperts supported=%s reason=%s (pid=%d)"
                     % (ok, reason, os.getpid()))
                return ok, reason

            cls.is_supported_config = staticmethod(patched)
            setattr(cls, "_support_wrapped", True)
            _log("AiterExperts support-log wrapper installed pid=%d" % os.getpid())

    if cls is not None and _env_on(ENV_FORCE_AITER) and not getattr(cls, "_sqs_forced", False):
        sqs = cls.__dict__.get("_supports_quant_scheme")
        if sqs is not None:
            orig_sqs = sqs.__func__ if isinstance(sqs, (staticmethod, classmethod)) else sqs

            def sqs_forced(weight_key, activation_key):
                try:
                    import vllm.platforms.rocm as rocm_mod
                except Exception:
                    return orig_sqs(weight_key, activation_key)
                saved = getattr(rocm_mod, "on_gfx950", None)
                rocm_mod.on_gfx950 = lambda: True  # 第二道门旁路(仅本次调用内生效)
                try:
                    return orig_sqs(weight_key, activation_key)
                finally:
                    if saved is not None:
                        rocm_mod.on_gfx950 = saved
                    else:
                        try:
                            delattr(rocm_mod, "on_gfx950")
                        except Exception:
                            pass

            cls._supports_quant_scheme = staticmethod(sqs_forced)
            setattr(cls, "_sqs_forced", True)
            _log("AITER-GATE2 (gfx950 quant-scheme) bypass installed pid=%d" % os.getpid())


def _install_triton_kernels(mod):
    for nm in ("UnfusedOAITritonExperts", "OAITritonMxfp4ExpertsMonolithic",
               "OAITritonExperts"):
        _wrap_support(getattr(mod, nm, None), nm)


def _env_on(name):
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _install_safetensors_fix(mod):
    """坑16:safetensors 0.8.0 的 _TYPES 表漏了 "F8_E8M0" 字符串键
    (_SIZE 里有 _float8_e8m0、torch 也有 float8_e8m0fnu dtype,就是映射表没挂)。
    DeepSeek-V4 的 ue8m0 缩放全是 F8_E8M0 → safe_open.get_tensor 崩
    "could not determine the shape of object type 'torch.storage.UntypedStorage'"。
    纯 bug 修复,不加开关,默认生效。"""
    if getattr(mod, MARK_ST, False):
        return
    types = getattr(mod, "_TYPES", None)
    if isinstance(types, dict) and "F8_E8M0" not in types:
        e8 = None
        try:
            import torch
            e8 = getattr(torch, "float8_e8m0fnu", None)
        except Exception:
            pass
        if e8 is not None:
            types["F8_E8M0"] = e8
            _log("safetensors _TYPES += F8_E8M0 (pid=%d)" % os.getpid())
    setattr(mod, MARK_ST, True)


def _install_aiter_gate(mod):
    """坑15：DUPFIX_FORCE_AITER=1 时把 is_aiter_found_and_supported() 恒改 True。

    消费方（rocm_aiter_moe/expert is_supported_config、quark mxfp4、
    oracle/unquantized、rocm.py 后端优先级）都是 from-import 或晚绑定，
    在 _aiter_ops exec 完成这一刻替换模块属性即可全覆盖。
    真实判断结果只记一次（该函数按 token 路径也会被调到，防日志洪水）。
    """
    if not _env_on(ENV_FORCE_AITER):
        return
    orig = getattr(mod, "is_aiter_found_and_supported", None)
    if orig is None or getattr(mod, MARK_AO, False):
        return
    state = {"logged": False}

    def forced():
        if not state["logged"]:
            state["logged"] = True
            try:
                real = bool(orig())
            except Exception as e:
                real = "raised %r" % (e,)
            _log("AITER-GATE force=ON real=%s -> True (pid=%d)"
                 % (real, os.getpid()))
        return True

    mod.is_aiter_found_and_supported = forced
    setattr(mod, MARK_AO, True)
    _log("aiter gate force-override installed pid=%d" % os.getpid())


def _install_rocm_fp8(mod):
    """坑17：DUPFIX_FORCE_FP8=1 时把 RocmPlatform.supports_fp8() 恒改 True。

    消费面实测很窄：关键一处是 oracle/fp8.py 里 TritonExperts._supports_quant_scheme
    （fp8 块量化组合只在 supports_fp8() 为真时进白名单）。gfx906 不在 on_gfx9 的
    [gfx90a/gfx942/gfx950] 白名单里 → FP8 MoE 的 TRITON 后端被拒。
    TRITON 融合核本质是把 fp8 权重反量化到 bf16 再算，老卡可跑，属纯架构白名单问题。
    与 DUPFIX_FORCE_AITER 是两条独立路（AITER 核 vs vllm 内置 TRITON 核），
    可分别强开对比哪条在 gfx906 上真能跑。真实判断结果只记一次，防日志洪水。
    默认关（不设变量行为与原来完全一致）。
    """
    if not _env_on(ENV_FORCE_FP8):
        return
    if getattr(mod, MARK_RC, False):
        return
    rp = getattr(mod, "RocmPlatform", None)
    if rp is None:
        return
    raw = getattr(rp, "supports_fp8", None)
    if raw is None:
        return
    orig = raw.__func__ if isinstance(raw, classmethod) else raw
    state = {"logged": False}

    def forced(cls):
        if not state["logged"]:
            state["logged"] = True
            try:
                real = bool(orig(cls))
            except Exception as e:
                real = "raised %r" % (e,)
            _log("FP8-GATE force=ON real=%s -> True (pid=%d)" % (real, os.getpid()))
        return True

    rp.supports_fp8 = classmethod(forced)
    setattr(mod, MARK_RC, True)
    _log("rocm supports_fp8 force-override installed pid=%d" % os.getpid())


def _install_hcu_fp8(mod):
    """坑17 修正（2026-10-04）：DCU 上 current_platform 是 HCUPlatform
    （vllm_hcu/platforms/hcu.py:216 class HCUPlatform(Platform)，不继承 RocmPlatform），
    且自带 supports_fp8 classmethod = on_gfx938()（hcu.py:714），在 gfx906 恒 False。
    上面 _install_rocm_fp8 只改 RocmPlatform 对本平台无效 —— TritonExperts.
    _supports_quant_scheme 走 current_platform.supports_fp8()，查的是 HCUPlatform。
    这里直接换 HCUPlatform 类上的 classmethod（取 __dict__ 防继承干扰）。
    """
    if not _env_on(ENV_FORCE_FP8):
        return
    if getattr(mod, MARK_RC, False):
        return
    hcu_platform = getattr(mod, "HCUPlatform", None)
    if hcu_platform is None:
        return
    raw = hcu_platform.__dict__.get("supports_fp8")
    if raw is None:
        return
    orig = raw.__func__ if isinstance(raw, classmethod) else raw
    state = {"logged": False}

    def forced(cls):
        if not state["logged"]:
            state["logged"] = True
            try:
                real = bool(orig(cls))
            except Exception as e:
                real = "raised %r" % (e,)
            _log("FP8-GATE(hcu) force=ON real=%s -> True (pid=%d)"
                 % (real, os.getpid()))
        return True

    hcu_platform.supports_fp8 = classmethod(forced)
    setattr(mod, MARK_RC, True)
    _log("hcu supports_fp8 force-override installed pid=%d" % os.getpid())


def _install_triton_arch_spoof(mod):
    """坑21（10-04）：das185 triton 的 libtriton C++ 只认 gfx926/928/936/938，
    gfx906 在 make_llir 的 ConvertTritonAMDGPUToLLVM 被拒（unsupported target）。
    探针 123623914 实测定案：make_llir 期间把 options.arch 换成 gfx938 过 C++ 检查
    （frozen dataclass 用 object.__setattr__），外部 clang 仍按真 gfx906 出 ISA，
    float/int64/f16-dot/fp8-e4m3-dot/inductor 五类核全部 PASS 且数值正确。
    只对 gfx906 生效（其他 arch 原样走），vllm TritonExperts 与 torch.compile
    的 inductor 都走这个 backend，一处 patch 全覆盖。
    """
    HB = getattr(mod, "HIPBackend", None)
    if HB is None or getattr(HB, "_dupfix_arch_spoofed", False):
        return
    raw = HB.__dict__.get("make_llir")
    if raw is None:
        return
    orig = raw.__func__ if isinstance(raw, staticmethod) else raw

    @staticmethod
    def patched(src, metadata, options):
        if getattr(options, "arch", None) == "gfx906":
            object.__setattr__(options, "arch", "gfx938")
            try:
                return orig(src, metadata, options)
            finally:
                object.__setattr__(options, "arch", "gfx906")
        return orig(src, metadata, options)

    HB.make_llir = patched
    setattr(HB, "_dupfix_arch_spoofed", True)
    _log("triton make_llir arch-spoof(gfx906->gfx938) installed pid=%d" % os.getpid())


def _install_triton_support_log(mod):
    """坑17 诊断：TritonExperts.is_supported_config 继承自基类（不在 cls.__dict__ 里），
    且拒绝原因只走 logger.debug_once（INFO 看不到）。把打日志包装显式装到类上，
    与 AiterExperts 同款 —— 每轮实验都能看到 TRITON 后端被哪道门拒。纯日志，无开关。
    """
    cls = getattr(mod, "TritonExperts", None)
    if cls is None or getattr(cls, "_support_wrapped", False):
        return
    raw = cls.is_supported_config  # 属性查找,可拿到继承方法
    if raw is None:
        return
    orig = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw

    def patched(*a, **k):
        try:
            ok, reason = orig(*a, **k)
        except Exception as e:
            _log("SUPPORT TritonExperts raised %r (pid=%d)" % (e, os.getpid()))
            raise
        _log("SUPPORT TritonExperts supported=%s reason=%s (pid=%d)"
             % (ok, reason, os.getpid()))
        return ok, reason

    cls.is_supported_config = staticmethod(patched)
    setattr(cls, "_support_wrapped", True)
    _log("TritonExperts support-log wrapper installed pid=%d" % os.getpid())


HANDLERS = {
    TARGET_OP: _install_op,
    TARGET_PU: _install_pu,
    TARGET_TU: _install_tu,
    TARGET_AO: _install_aiter_gate,
    TARGET_ST: _install_safetensors_fix,
    TARGET_RC: _install_rocm_fp8,
    TARGET_HP: _install_hcu_fp8,
    TARGET_TB: _install_triton_arch_spoof,
    TARGET_TM: _install_triton_support_log,
    "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe": _install_aiter,
    "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe":
        _install_triton_kernels,
}


def _setup():
    from importlib.abc import Loader, MetaPathFinder

    class _WrapLoader(Loader):
        def __init__(self, inner, handler):
            self.inner = inner
            self.handler = handler

        def create_module(self, spec):
            cm = getattr(self.inner, "create_module", None)
            return cm(spec) if cm is not None else None

        def exec_module(self, module):
            self.inner.exec_module(module)
            try:
                self.handler(module)
            except Exception as e:
                _log("install failed: %r" % (e,))

    class _Finder(MetaPathFinder):
        def __init__(self):
            self.done = set()

        def find_spec(self, fullname, path=None, target=None):
            handler = HANDLERS.get(fullname)
            if handler is None or fullname in self.done:
                return None
            for f in list(sys.meta_path):
                if f is self:
                    continue
                fs = getattr(f, "find_spec", None)
                if fs is None:
                    continue
                try:
                    spec = fs(fullname, path, target)
                except Exception:
                    continue
                if spec is not None and getattr(spec, "loader", None) is not None:
                    spec.loader = _WrapLoader(spec.loader, handler)
                    self.done.add(fullname)
                    return spec
            return None

        def invalidate_caches(self):
            pass

    for name, handler in HANDLERS.items():
        already = sys.modules.get(name)
        if already is not None:
            try:
                handler(already)
            except Exception as e:
                _log("install failed: %r" % (e,))
    for f in sys.meta_path:
        # 用 is 精确匹配本模块的 _Finder；virtualenv 会在 meta_path 里放
        # _virtualenv._Finder（类名同为 _Finder），若按 __name__ 判断会误跳过。
        if type(f) is _Finder:
            return
    sys.meta_path.insert(0, _Finder())


try:
    _setup()
except Exception as e:
    _log("setup failed: %r" % (e,))
