"""Native acceleration for the locked lattice and capacity assignments."""

import ctypes
from functools import lru_cache
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from km_backend import P_DOUBLE, load_native_library

HERE = Path(__file__).resolve().parent
SIZE = ctypes.c_size_t
P_SIZE = ctypes.POINTER(SIZE)


def _bind(library, symbol, args, result):
    func = getattr(library, symbol)
    func.argtypes, func.restype = args, result
    return func


def _size_ptr(a):
    return a.ctypes.data_as(P_SIZE)


def _double_ptr(a):
    return a.ctypes.data_as(P_DOUBLE)


class DesignBackend:
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


def install(runner):
    """Install the native helpers in one experiment process."""
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
    from paired_sample_backend import bind_sampler
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
