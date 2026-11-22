"""Native exact fast path for independent paired phase-II sampling.

``bind_sampler`` does not compile or load a library.  Its parent owns that
policy and passes an already loaded shared library plus the archived sampler.
The returned callable is intentionally conservative: it uses C only for
all-pair allocations with finite, tie-free priorities, and otherwise calls the
reference implementation unchanged.
"""

from __future__ import annotations

import ctypes

import numpy as np


_P_DOUBLE = ctypes.POINTER(ctypes.c_double)
_P_SIZE = ctypes.POINTER(ctypes.c_size_t)


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
        kernel.argtypes = [_P_DOUBLE, _P_SIZE, _P_SIZE, ctypes.c_size_t, _P_SIZE]
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
            status = kernel(native_priorities.ctypes.data_as(_P_DOUBLE),
                            groups.ctypes.data_as(_P_SIZE), edges.ctypes.data_as(_P_SIZE),
                            len(unique), chosen.ctypes.data_as(_P_SIZE))
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
