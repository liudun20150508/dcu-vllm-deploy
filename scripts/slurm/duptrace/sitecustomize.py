# 坑12 探针：抓 `Duplicate op name: deepseek_v4_multi_head_latent_attention` 的第二次注册来源。
# 只在 PYTHONPATH 含本目录的作业里生效（site.py 启动时自动 import sitecustomize），
# 子进程（registry.inspect_model_cls 的 _run_in_subprocess）同样会继承 PYTHONPATH 并触发。
#
# 做法：用 meta_path finder 在 vllm.model_executor.custom_op 被 exec 完成的瞬间
# 包装 PluggableLayer.register，每次注册打印：pid/ppid/argv、新类、已被占用的类、
# 以及去掉 importlib 噪声后的完整调用栈。duplicate 时 assert 会抛，但栈先打出来了。
import os
import sys

TARGET = "vllm.model_executor.custom_op"
MARK = "_duptrace_installed"


def _log(msg):
    try:
        sys.stderr.write("[DUP-TRACE] " + msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _env_tag():
    keys = ("RANK", "LOCAL_RANK", "NODE_RANK", "SLURM_NODEID",
            "VLLM_WORKER_MULTIPROC_METHOD", "TP", "PP")
    return " ".join("%s=%s" % (k, os.environ.get(k)) for k in keys if os.environ.get(k))


def _clean_stack():
    import traceback
    out = []
    for fr in traceback.extract_stack()[:-1]:
        fn = fr.filename or ""
        if "<frozen importlib" in fn or "importlib/" in fn or fn.endswith("importlib/__init__.py"):
            continue
        out.append("    %s:%d in %s" % (fn, fr.lineno, fr.name))
    return "\n".join(out)


def _install(mod):
    PL = getattr(mod, "PluggableLayer", None)
    if PL is None or getattr(PL, MARK, False):
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
            tag = "DUPLICATE" if prev is not None else "first"
            prev_s = "%s.%s" % (prev.__module__, prev.__name__) if prev is not None else "None"
            _log("%s name=%s pid=%d ppid=%d argv=%s %s"
                 % (tag, name, os.getpid(), os.getppid(), sys.argv[:3], _env_tag()))
            _log("    new=%s.%s  prev=%s" % (op_cls.__module__, op_cls.__name__, prev_s))
            _log("  stack:\n" + _clean_stack())
            return dec(op_cls)

        return wrapper

    PL.register = classmethod(register)
    setattr(PL, MARK, True)
    _log("hook installed pid=%d ppid=%d on %s" % (os.getpid(), os.getppid(), mod.__name__))


def _setup():
    from importlib.abc import Loader, MetaPathFinder

    class _WrapLoader(Loader):
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            cm = getattr(self.inner, "create_module", None)
            return cm(spec) if cm is not None else None

        def exec_module(self, module):
            self.inner.exec_module(module)
            try:
                _install(module)
            except Exception as e:  # 探针绝不能把业务搞崩
                _log("install failed: %r" % (e,))

    class _Finder(MetaPathFinder):
        def __init__(self):
            self.done = False

        def find_spec(self, fullname, path=None, target=None):
            if self.done or fullname != TARGET:
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
                    spec.loader = _WrapLoader(spec.loader)
                    self.done = True
                    return spec
            return None

        def invalidate_caches(self):
            pass

    already = sys.modules.get(TARGET)
    if already is not None:
        _install(already)
        return
    for f in sys.meta_path:
        if type(f).__name__ == "_Finder":
            return
    sys.meta_path.insert(0, _Finder())


try:
    _setup()
except Exception as e:
    _log("setup failed: %r" % (e,))
