"""C and archived NumPy implementations of the weighted-KM RMST kernel."""

import ctypes
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "native" / "weighted_km.c"
REFERENCE = HERE / "legacy_locked" / "survival.py"
P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INDEX = ctypes.POINTER(ctypes.c_ssize_t)


@lru_cache(maxsize=1)
def reference():
    spec = importlib.util.spec_from_file_location("_bud_archived_survival", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiler_command():
    override = os.environ.get("BUD_CC")
    choices = (override,) if override else ("gcc", "clang", "cc", "cl")
    for name in choices:
        executable = shutil.which(name)
        if executable:
            return Path(executable).resolve()
    raise RuntimeError("The C backend needs GCC, Clang or MSVC. Set BUD_CC to the compiler path "
                       "or select --backend python.")


@lru_cache(maxsize=8)
def load_native_library(source):
    compiler = compiler_command()
    msvc = compiler.stem.lower() in {"cl", "clang-cl"}
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    if msvc:
        flags = ["/nologo", "/LD", "/O2", "/fp:strict"]
    else:
        link = "-dynamiclib" if sys.platform == "darwin" else "-shared"
        flags = ["-std=c11", "-O3", "-fno-fast-math", "-ffp-contract=off", link]
        if os.name != "nt":
            flags.append("-fPIC")
        if compiler.stem.lower().endswith("gcc"):
            flags.append("-fexcess-precision=standard")
            if os.name == "nt" and (compiler.parent / "as.exe").is_file():
                flags.append(f"-B{compiler.parent}/")
    configuration = (str(compiler), compiler.stat().st_mtime_ns, sysconfig.get_platform(), flags)
    digest = hashlib.sha256(source.read_bytes() + repr(configuration).encode()).hexdigest()
    cache = Path(tempfile.gettempdir()) / "bud_native_v1" / digest
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / f"{source.stem}{suffix}"
    if not binary.exists():
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            target = Path(directory) / binary.name
            output = [f"/Fe:{target}"] if msvc else ["-o", str(target)]
            command = [str(compiler), *flags, str(source), *output]
            result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace", check=False)
            if result.returncode or not target.exists():
                raise RuntimeError(f"Could not build the C backend with {compiler.name}. "
                                   f"Use BUD_CC or --backend python.\n{result.stdout}")
            try:
                target.replace(binary)
            except PermissionError:
                # A different process may have published and loaded the same DLL.
                if not binary.is_file():
                    raise
    try:
        library = ctypes.CDLL(str(binary))
    except OSError as exc:
        message = "Use --backend python or have the native library approved under your local policy."
        if getattr(exc, "winerror", None) == 4551:
            raise RuntimeError(f"Windows application control blocked {binary.name}. {message}") from exc
        raise RuntimeError(f"Could not load {binary.name}: {exc}. {message}") from exc
    return library, dict(compiler=compiler.name, flags=flags, platform=sysconfig.get_platform(),
                         c_source_sha256=sha(source), library_sha256=sha(binary))


@lru_cache(maxsize=1)
def native():
    library, configuration = load_native_library(SOURCE)
    kernel = library.bud_km_survival
    count = ctypes.c_size_t
    kernel.argtypes = [P_DOUBLE, count, count, P_INDEX, count, P_INDEX, P_INDEX,
                       P_INDEX, P_DOUBLE, P_DOUBLE]
    kernel.restype = ctypes.c_int
    return kernel, configuration


def _tied_events(weights, order, sorted_event, first, use, counts):
    """Preserve the archived NumPy reduction order inside tied jumps."""
    starts = first[use]
    failures = np.flatnonzero(sorted_event)
    first_failure = np.ascontiguousarray(order[failures[np.searchsorted(failures, starts)]], dtype=np.intp)
    multiple = counts > 1
    slots = np.full(len(starts), -1, dtype=np.intp)
    if not multiple.any():
        return first_failure, slots, np.empty(0, dtype=np.float64)
    stops = np.r_[first[1:], len(order)][use]
    starts_tied, stops_tied = starts[multiple], stops[multiple]
    positions = np.concatenate([np.arange(a, b) for a, b in zip(starts_tied, stops_tied)])
    offsets = np.r_[0, np.cumsum(stops_tied-starts_tied)[:-1]]
    products = weights[:, order[positions]] * sorted_event[positions][None, :]
    totals = np.asfortranarray(np.add.reduceat(products, offsets, axis=1))
    slots[multiple] = np.arange(len(starts_tied))
    return first_failure, slots, totals


class KMBackend:
    """Select the numerical backend without changing seeds, estimators or designs."""

    def __init__(self, name="c"):
        self.name = name
        self._reference = reference()
        self._configuration = {}
        if name == "c":
            self._kernel, self._configuration = native()
        elif name != "python":
            raise ValueError("backend must be 'c' or 'python'")

    def metadata(self):
        return {"name": self.name, "numpy": np.__version__, "reference_sha256": sha(REFERENCE),
                "adapter_sha256": sha(Path(__file__)), **self._configuration}

    def km_rmst(self, time, event, weights, tau):
        ref = self._reference
        if self.name == "python" or np.ndim(weights) == 1:
            return ref.km_rmst(time, event, weights, tau)
        t, e = ref._check_observations(time, event, tau)
        w = np.asarray(weights, dtype=np.float64, order="C")
        if w.ndim != 2 or w.shape[1] != len(t):
            raise ValueError("weights must have shape (n,) or (B,n)")
        order, sorted_event, first, use, times, counts = ref._event_layout(t, e, tau)
        if len(times) == 0 or len(w) == 0:
            return ref.km_rmst(t, e, w, tau)
        starts = np.ascontiguousarray(first[use], dtype=np.intp)
        failure, slots, tied_totals = _tied_events(w, order, sorted_event, first, use, counts)
        order = np.ascontiguousarray(order, dtype=np.intp)
        survival = np.empty((len(w), len(starts)), dtype=np.float64, order="F")
        error = self._kernel(w.ctypes.data_as(P_DOUBLE), len(w), len(t),
                             order.ctypes.data_as(P_INDEX), len(starts), starts.ctypes.data_as(P_INDEX),
                             failure.ctypes.data_as(P_INDEX), slots.ctypes.data_as(P_INDEX),
                             tied_totals.ctypes.data_as(P_DOUBLE), survival.ctypes.data_as(P_DOUBLE))
        if error:
            raise ValueError("weights must be finite, nonnegative, and have positive row totals")
        widths = np.diff(np.r_[0.0, times, float(tau)])
        # Use the same NumPy/BLAS reduction and memory layout in both backends.
        return widths[0] + survival @ widths[1:]

    def bootstrap_rmst(self, time, event, weights, meta, rng, B, tau):
        wstar = self._reference.bootstrap_weights(weights, meta, rng, B)
        return self.km_rmst(time, event, wstar, tau)
