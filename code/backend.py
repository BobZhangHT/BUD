"""Native kernels and NumPy reference paths for the BUD experiments."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
KM_SOURCE = HERE / "native" / "weighted_km.c"
SURVIVAL = HERE / "legacy_locked" / "survival.py"
P_DOUBLE = ctypes.POINTER(ctypes.c_double)
P_INDEX = ctypes.POINTER(ctypes.c_ssize_t)
SIZE = ctypes.c_size_t
P_SIZE = ctypes.POINTER(SIZE)


@lru_cache(maxsize=1)
def _reference_survival():
    """Load the locked survival module without shadowing the real-data imports."""
    spec = importlib.util.spec_from_file_location("_bud_archived_survival", SURVIVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiler():
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
    """Build once per source and compiler configuration, then load the cached binary."""
    compiler = _compiler()
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


# Weighted Kaplan?Meier and RMST


@lru_cache(maxsize=1)
def _km_kernel():
    library, configuration = load_native_library(KM_SOURCE)
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
    # Pre-sum failures for tied jumps so that the C kernel uses NumPy's
    # reduction order. Single-failure jumps need only the failure's index.
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
        self._reference = _reference_survival()
        self._configuration = {}
        if name == "c":
            self._kernel, self._configuration = _km_kernel()
        elif name != "python":
            raise ValueError("backend must be 'c' or 'python'")

    def metadata(self):
        return {"name": self.name, "numpy": np.__version__, "reference_sha256": sha(SURVIVAL),
                "adapter_sha256": sha(Path(__file__)), **self._configuration}

    def km_rmst(self, time, event, weights, tau):
        """Integrate the weighted product limit up to tau for one or many weight rows."""
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
        """Reuse the reference bootstrap draws with the selected RMST kernel."""
        wstar = self._reference.bootstrap_weights(weights, meta, rng, B)
        return self.km_rmst(time, event, wstar, tau)


# Lattice designs and capacity constraints


def _bind(library, symbol, args, result):
    func = getattr(library, symbol)
    func.argtypes, func.restype = args, result
    return func


def _size_ptr(a):
    return a.ctypes.data_as(P_SIZE)


def _double_ptr(a):
    return a.ctypes.data_as(P_DOUBLE)


class DesignBackend:
    """Accelerate GLP screening, anchor transport, and exact cell assignment."""

    def __init__(self, glp_reference, quantile_reference):
        library, self.metadata = load_native_library(HERE / "native/design.c")
        self._original_glp = glp_reference
        self._original_quantiles = quantile_reference
        self._generators = _bind(library, "bud_glp_generators",
                                 [SIZE, SIZE, P_SIZE, P_SIZE, P_SIZE], SIZE)
        self._points = _bind(library, "bud_glp_points", [SIZE, SIZE, P_SIZE, P_DOUBLE], None)
        self._screen = _bind(library, "bud_glp_screen",
                             [SIZE, SIZE, SIZE, P_SIZE, P_DOUBLE], ctypes.c_int)
        self._expand = _bind(library, "bud_expand_cost", [SIZE, SIZE, P_DOUBLE, P_SIZE, P_DOUBLE], None)
        self._quantiles = _bind(library, "bud_inverse_quantiles",
                                [SIZE, SIZE, SIZE, P_DOUBLE, P_DOUBLE, P_DOUBLE], None)

    def lattice_points(self, h, generator):
        g = np.ascontiguousarray(generator, dtype=np.uintp)
        points = np.empty((h, len(g)), dtype=np.float64)
        self._points(h, len(g), _size_ptr(g), _double_ptr(points))
        return points

    def generators(self, h, q):
        units = np.empty(h, dtype=np.uintp)
        generators = np.empty((h, q), dtype=np.uintp)
        unit_count = SIZE()
        count = self._generators(h, q, _size_ptr(units), ctypes.byref(unit_count), _size_ptr(generators))
        if count:
            return generators[:count]
        # Preserve NumPy's archived fallback stream and the original 32 draws.
        units = units[:unit_count.value].astype(np.intp)
        rng = np.random.default_rng(781211+997*h+q)
        fallback = [rng.choice(units, size=q, replace=q > len(units)) for _ in range(32)]
        return np.ascontiguousarray(fallback, dtype=np.uintp)

    def screen_scores(self, h, generators):
        count, q = generators.shape
        scores = np.empty(count, dtype=np.float64)
        if self._screen(h, q, count, _size_ptr(generators), _double_ptr(scores)):
            raise MemoryError("Could not allocate GLP screening workspace")
        return scores

    @lru_cache(maxsize=128)
    def glp_uniform_design(self, h, q):
        h, q = int(h), int(q)
        if not 1 < h <= 2048 or not 0 < q <= 64:
            return self._original_glp(h, q)
        # Every power generator has a first component of one.
        if q == 1:
            answer = self.lattice_points(h, [1])
        else:
            generators = self.generators(h, q)
            scores = self.screen_scores(h, generators)
            if not np.isfinite(scores).all():
                return self._original_glp(h, q)
            # All univariate factors are positive and at most 15/8; summing
            # H^2 such products is the largest roundoff term. The conservative
            # envelope covers both summation orders and the screening formula.
            scale = math.pow(15.0/8.0, q)
            error = 128*np.finfo(float).eps*(h*h+q+1)*scale
            # If two error intervals can overlap, resolve in exact reference
            # order. This also preserves the first-minimum tie rule.
            candidates = np.flatnonzero(scores <= scores.min()+2*error)
            if len(candidates) == 1:
                answer = self.lattice_points(h, generators[candidates[0]])
            else:
                best = math.inf
                for idx in candidates:
                    points = self.lattice_points(h, generators[idx])
                    value = qmc.discrepancy(points, method="MD")
                    if value < best:
                        best, answer = value, points
        answer.setflags(write=False)
        return answer

    def inverse_quantile_anchors(self, scores, design):
        scores, design = np.asarray(scores), np.asarray(design)
        if (scores.ndim != 2 or design.ndim != 2 or not len(scores) or scores.shape[1] != design.shape[1]
                or not np.isfinite(scores).all() or not np.isfinite(design).all()
                or np.any(design < 0) or np.any(design > 1)):
            return self._original_quantiles(scores, design)
        design = np.ascontiguousarray(design, dtype=np.float64)
        ordered = np.ascontiguousarray(np.sort(scores.astype(np.float64, copy=False), axis=0))
        anchors = np.empty_like(design)
        self._quantiles(len(ordered), *design.shape, _double_ptr(ordered), _double_ptr(design), _double_ptr(anchors))
        return anchors

    def expanded_cost(self, distances, capacities):
        d = np.ascontiguousarray(distances, dtype=np.float64)
        caps = np.ascontiguousarray(capacities, dtype=np.uintp)
        if caps.ndim != 1 or len(caps) != d.shape[1] or int(caps.sum()) != len(d):
            raise ValueError("Assignment capacities must match the distance matrix")
        dense = np.empty((len(d), len(d)), dtype=np.float64)
        self._expand(*d.shape, _double_ptr(d), _size_ptr(caps), _double_ptr(dense))
        return dense

    def capacitated_assignment(self, coordinates, anchors, capacities):
        distances = cdist(coordinates, anchors, metric="sqeuclidean")
        # The expanded assignment has one slot per seat in each cell.
        slots = np.repeat(np.arange(len(anchors)), capacities)
        row, col = linear_sum_assignment(self.expanded_cost(distances, capacities))
        labels = np.empty(len(coordinates), dtype=int)
        labels[row] = slots[col]
        assert np.array_equal(np.bincount(labels, minlength=len(anchors)), capacities)
        return labels, float(distances[np.arange(len(coordinates)), labels].mean())

    def balanced_assignment(self, coordinates, anchors):
        n, h = len(coordinates), len(anchors)
        if n % h:
            raise ValueError("Exact BUD requires N divisible by the number of strata.")
        return self.capacitated_assignment(coordinates, anchors, np.full(h, n//h, dtype=np.intp))


# Phase-II sampling of two records per cell


def _pair_allocation(allocation, unique):
    """Return an all-two allocation, or ``None`` when reference must decide."""
    if isinstance(allocation, dict):
        return None
    if np.ndim(allocation) == 0:
        try:
            return np.full(len(unique), allocation) if allocation == 2 else None
        except (TypeError, ValueError):
            return None
    try:
        alloc = np.asarray(allocation)
    except Exception:
        return None
    if alloc.shape != (len(unique),):
        return None
    try:
        return alloc if np.all(alloc == 2) else None
    except (TypeError, ValueError):
        return None


def bind_sampler(library, reference_draw_sample):
    """Bind ``bud_paired_sample`` from *library* as a drop-in sampler.

    ``reference_draw_sample`` is called for general allocations, malformed
    inputs, and boundary ties, preserving its validation and NumPy partition
    semantics.  The native path never advances ``rng`` itself.
    """
    if not callable(reference_draw_sample):
        raise TypeError("reference_draw_sample must be callable")
    kernel = None
    if library is not None:
        kernel = library.bud_paired_sample
        kernel.argtypes = [P_DOUBLE, P_SIZE, P_SIZE, ctypes.c_size_t, P_SIZE]
        kernel.restype = ctypes.c_int

    def draw_sample(labels, allocation, rng, priorities=None):
        # Leave all unusual validation cases to the archived routine.  In
        # particular, this prevents an adapter-only exception before its RNG
        # draw behaviour is known.
        try:
            lab = np.asarray(labels)
            if lab.ndim != 1 or not len(lab):
                return reference_draw_sample(labels, allocation, rng, priorities=priorities)
            if isinstance(allocation, dict) or not np.all(np.asarray(allocation) == 2):
                return reference_draw_sample(labels, allocation, rng, priorities=priorities)
            unique, inv, sizes = np.unique(lab, return_inverse=True, return_counts=True)
            alloc = _pair_allocation(allocation, unique)
            if alloc is None or np.any(sizes < 2):
                return reference_draw_sample(labels, allocation, rng, priorities=priorities)
        except Exception:
            return reference_draw_sample(labels, allocation, rng, priorities=priorities)

        if priorities is None:
            # Exactly the reference sampler's sole RNG operation.
            pri = rng.random(len(lab))
        else:
            pri = np.asarray(priorities, dtype=float)
        if pri.shape != lab.shape or np.any(~np.isfinite(pri)):
            # Pass the generated priorities, if any, to prevent a second draw.
            return reference_draw_sample(labels, allocation, rng, priorities=pri)

        edges = np.ascontiguousarray(np.r_[0, np.cumsum(sizes)], dtype=np.uintp)
        if kernel is None:
            # System policies can disallow a locally built library. The same
            # selection is vectorized with NumPy's compiled lexicographic sort.
            ordered = np.lexsort((pri, inv))
            head = edges[:-1]
            check = head[sizes > 2]
            if np.any(pri[ordered[check+1]] == pri[ordered[check+2]]):
                return reference_draw_sample(labels, allocation, rng, priorities=pri)
            a, b = ordered[head], ordered[head+1]
            chosen = np.column_stack((np.minimum(a,b), np.maximum(a,b))).ravel()
        else:
            groups = np.ascontiguousarray(np.argsort(inv, kind="stable"), dtype=np.uintp)
            native_priorities = np.ascontiguousarray(pri, dtype=np.float64)
            chosen = np.empty(2 * len(unique), dtype=np.uintp)
            status = kernel(native_priorities.ctypes.data_as(P_DOUBLE),
                            groups.ctypes.data_as(P_SIZE), edges.ctypes.data_as(P_SIZE),
                            len(unique), chosen.ctypes.data_as(P_SIZE))
            if status != 0:
                return reference_draw_sample(labels, allocation, rng, priorities=pri)

        selected = chosen.astype(np.intp, copy=False)
        weights = np.repeat(sizes.astype(float) / 2.0, 2)
        meta = []
        offset = 0
        for label, size in zip(unique, sizes):
            item = label.item() if hasattr(label, "item") else label
            meta.append({"label": item, "N": int(size), "s": 2,
                         "start": offset, "stop": offset + 2,
                         "f": float(2 / size)})
            offset += 2
        return selected, weights, meta

    return draw_sample


# Installation and provenance


def install(runner):
    """Bind native helpers to the locked runner in its isolated process."""
    design = sys.modules["designs"]
    original = (design.glp_uniform_design, design.inverse_quantile_anchors)
    adapter = DesignBackend(*original)
    design.glp_uniform_design = adapter.glp_uniform_design
    design.inverse_quantile_anchors = adapter.inverse_quantile_anchors
    design.balanced_assignment = adapter.balanced_assignment
    for name in ("refinement", "anchor_ablation", "final_anchor_ablation", "real_application"):
        module = sys.modules.get(name)
        if module is None:
            continue
        for helper in ("glp_uniform_design", "inverse_quantile_anchors", "balanced_assignment", "capacitated_assignment"):
            if hasattr(module, helper):
                setattr(module, helper, getattr(adapter, helper))
    library, _ = paired_library()
    if hasattr(runner, "draw_sample"):
        runner.draw_sample = bind_sampler(library, runner.draw_sample)


@lru_cache(maxsize=1)
def paired_library():
    try:
        library, metadata = load_native_library(HERE / "native/paired_sample.c")
    except RuntimeError as exc:
        if getattr(exc.__cause__, "winerror", None) != 4551:
            raise
        return None, {"backend": "numpy", "reason": "Windows application control blocked the sampler DLL"}
    return library, {**metadata, "backend": "c"}


def native_metadata():
    return {"design": load_native_library(HERE / "native/design.c")[1], "paired_sample": paired_library()[1]}
