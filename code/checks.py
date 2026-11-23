"""Verify paper outputs and numerical agreement between the C and NumPy backends."""

import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import json
import os
from math import isclose
from statistics import median
from pathlib import Path
import platform
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_results.json"
# Tight enough to catch a changed draw or reported rounded value, while tolerating
# harmless floating-point differences between supported compiler/BLAS builds.
RTOL = 2e-9
ATOL = 1e-11


def extract(root, check):
    path = root / check["path"]
    if path.suffix == ".pdf":
        # PDFs carry build-dependent metadata. Their underlying numerical inputs
        # are verified separately, so here we check that the output was rendered.
        with path.open("rb") as file:
            if file.read(5) != b"%PDF-":
                raise ValueError(f"Invalid PDF header in {path}")
        return None
    if path.suffix == ".tex":
        return sha256(path.read_text(encoding="utf8").strip().encode("utf8")).hexdigest()
    if path.suffix == ".csv":
        with path.open(encoding="utf8", newline="") as file:
            rows = csv.DictReader(file)
            return sorted([row[column] for column in check["columns"]] for row in rows
                          if all(row[key] in values for key, values in check.get("where", {}).items()))
    values = json.loads(path.read_bytes())
    selected = []
    for fields in check["fields"]:
        value = values
        for field in fields:
            value = value[field]
        # Seeds are represented as hex in the reference so comparison is exact.
        selected.append(hex(value) if fields == ["seed"] else value)
    return selected


def compare(actual, expected, where):
    if type(expected) is dict:
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError(f"{where}: changed object keys")
        for key, reference in expected.items():
            compare(actual[key], reference, f"{where}.{key}")
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError(f"{where}: expected {len(expected)} items")
        for index, (candidate, reference) in enumerate(zip(actual, expected)):
            compare(candidate, reference, f"{where}[{index}]")
    elif type(expected) is bool or type(actual) is bool:
        if actual is not expected:
            raise ValueError(f"{where}: expected {expected!r}, found {actual!r}")
    elif actual != expected:
        # CSV readers return strings. Compare numbers as numbers, but keep
        # labels, exact booleans, and missing-value markers as literal values.
        try:
            if isclose(float(actual), float(expected), rel_tol=RTOL, abs_tol=ATOL):
                return
        except (TypeError, ValueError):
            pass
        raise ValueError(f"{where}: expected {expected!r}, found {actual!r}")


def verify(target, outputs):
    reference = json.loads(REFERENCE.read_bytes(), parse_constant=lambda value: value)
    targets = reference if target == "all" else {target: reference[target]}
    failures = []
    for name, checks in targets.items():
        for check in checks:
            try:
                compare(extract(outputs, check), check.get("expected"), check["path"])
            except (OSError, KeyError, ValueError) as error:
                failures.append(f"{name}: {error}")
        print(f"{name}: checked {len(checks)} output files")
    if failures:
        raise SystemExit("Verification failed:\n" + "\n".join(failures))
    print("All selected paper outputs matched the reference.")


def doctor(backend):
    from bud_experiment import _new_backend

    print(f"Python {platform.python_version()} | {platform.platform()}", flush=True)
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib"):
        print(f"{package} {version(package)}", flush=True)
    configuration = _new_backend(backend).metadata()
    if backend == "c":
        from backend import native_metadata
        configuration.update(native_metadata())
    print(json.dumps(configuration, indent=2))


# Numerical kernels and optional performance checks

TIMING_COLUMNS = {"seconds", "mean_seconds", "total_seconds", "evaluation_seconds", "seconds_design_per_arm",
                  "pca_seconds", "pc1_total_seconds", "bud_total_seconds"}


def bit_equal(left, right):
    import numpy as np

    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


def check_kernel(py, c):
    import numpy as np

    rng = np.random.default_rng(227303)
    cases = 400
    for i in range(cases):
        n = int(rng.integers(1, 500))
        reps = int(rng.integers(1, 120))
        tau = float(rng.integers(1, n + 2))
        if i % 3 == 0:
            time = rng.uniform(0, n + 1, n)
        else:
            time = rng.integers(0, max(1, n // (1 if i % 3 == 1 else 15)), n).astype(float)
        event = rng.random(n) < .5
        weights = np.exp(rng.normal(0, 6, (reps, n)))
        weights[rng.random(weights.shape) < .4] = 0
        weights[:, 0] = 1
        if i % 4 == 0:
            weights = np.asfortranarray(weights)
        if i % 9 == 0:
            time[-1] = np.inf
        bit_equal(py.km_rmst(time, event, weights, tau), c.km_rmst(time, event, weights, tau))
    # Events at the origin and tied censoring, a zero-weight tail, an event
    # exactly at the horizon, and more than one failure within a jump.
    time = np.array([0, 0, 1, 1, 1, 1, 2, 3, 4, 4.])
    event = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    weights = np.array([[0, .1, .2, .3, 0, .1, .4, 0, 0, 0],
                        [0, .1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 3, 0, 1, 1, 1, 1, 1, 2, 1]])
    for outcome in (event, np.zeros_like(event)):
        bit_equal(py.km_rmst(time, outcome, weights, 4), c.km_rmst(time, outcome, weights, 4))
        bit_equal(py.km_rmst(time, outcome, weights[0], 4), c.km_rmst(time, outcome, weights[0], 4))
    for invalid in (np.zeros((2, len(time))), -np.ones((2, len(time))),
                    np.full((2, len(time)), np.nan), np.ones((2, len(time)-1))):
        for backend in (py, c):
            try:
                backend.km_rmst(time, event, invalid, 4)
            except ValueError:
                continue
            raise AssertionError(f"{backend.name} accepted invalid weights")
    return cases + 4


def benchmark_kernel(py, c):
    import numpy as np

    rng = np.random.default_rng(673425)
    summary = []
    for tie in (False, True):
        for n in (40, 120, 400, 800):
            time = rng.integers(0, max(10, n//3), n) if tie else rng.uniform(0, 600, n)
            event = rng.random(n) < .5
            weights = rng.uniform(.3, 2, (1000, n))
            bit_equal(py.km_rmst(time, event, weights, 600), c.km_rmst(time, event, weights, 600))
            trials = {p.name: [] for p in (py, c)}
            for i in range(12):
                for backend in ((py, c) if i % 2 == 0 else (c, py)):
                    start = perf_counter()
                    backend.km_rmst(time, event, weights, 600)
                    trials[backend.name].append(1000*(perf_counter()-start))
            python_ms, c_ms = [median(trials[k]) for k in ("python", "c")]
            summary.append(dict(size=n, tied_times=tie, resamples=1000,
                                python_ms=python_ms, c_ms=c_ms, speedup=python_ms/c_ms))
    return summary


def compare_results(paths):
    import pandas as pd

    left, right = [Path(x) for x in paths]
    matched = []
    candidates = ("raw", "real_raw", "main_three_methods", "diagnostics", "cohort_metrics",
                  "summary", "primary_summary", "real_summary", "paired", "paired_mse", "real_paired", "paired_evidence",
                  "first_draw_estimates", "first_draw_selected")
    for name in candidates:
        file_pair = []
        for folder in (left, right):
            file_pair.append(next((folder/f"{name}{ext}" for ext in (".csv", ".csv.gz")
                                   if (folder/f"{name}{ext}").is_file()), None))
        if not all(file_pair):
            continue
        a, b = [pd.read_csv(x, float_precision="round_trip") for x in file_pair]
        timing = [x for x in a.columns if x in TIMING_COLUMNS]
        a, b = (frame.drop(columns=timing) for frame in (a, b))
        keys = [x for x in ("dataset", "cohort", "family", "scene", "n_group", "m", "replicate",
                            "draw", "group", "method", "variant", "comparator", "benchmark", "baseline",
                            "source_row") if x in a and x in b]
        if keys:
            a, b = (frame.sort_values(keys).reset_index(drop=True) for frame in (a, b))
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        for col in a.select_dtypes(include=["floating"]):
            bit_equal(a[col].to_numpy(), b[col].to_numpy())
        matched.append(dict(result=name, rows=len(a), max_absolute_error=0.0, elapsed_time_columns_ignored=timing))
    if not any(d["result"] in {"raw", "real_raw"} for d in matched):
        raise RuntimeError("An end-to-end comparison requires matching raw or real_raw results")
    partitions = {p.name for p in left.glob("*_partitions.csv")}
    if partitions != {p.name for p in right.glob("*_partitions.csv")}:
        raise AssertionError("Saved partition files differ between backends")
    for filename in sorted(partitions):
        if (left / filename).read_bytes() != (right / filename).read_bytes():
            raise AssertionError(f"Saved partition differs: {filename}")
    if partitions:
        matched.append(dict(result="partitions", files=sorted(partitions), byte_identical=True))
    return matched


def benchmark_design():
    """Time uncached designs and verify the exact selected lattice and anchors."""
    import numpy as np

    from backend import DesignBackend, native_metadata

    sys.path.insert(0, str(ROOT / "code/legacy_locked"))
    from designs import glp_uniform_design, inverse_quantile_anchors

    native = DesignBackend(glp_uniform_design, inverse_quantile_anchors)
    rng = np.random.default_rng(673425)
    cases = [(1, 8), (2, 5), (3, 6), (20, 4), (40, 1), (40, 5), (60, 5),
             (60, 13), (100, 5), (120, 8), (200, 5), (400, 6)]
    timings = []
    signed_zero_differences = 0
    for h, q in cases:
        methods = {"python": glp_uniform_design, "c": native.glp_uniform_design}
        durations = {name: [] for name in methods}
        for trial in range(5):
            points = {}
            for name in (("python", "c") if trial % 2 == 0 else ("c", "python")):
                generate = methods[name]
                generate.cache_clear()
                start = perf_counter()
                points[name] = generate(h, q)
                durations[name].append(perf_counter() - start)
            bit_equal(points["python"], points["c"])
        scores = np.round(rng.normal(size=(801, q)), 1)
        reference = inverse_quantile_anchors(scores, points["python"])
        accelerated = native.inverse_quantile_anchors(scores, points["c"])
        # Sorting and partial sorting can order equal signed zeros differently.
        np.testing.assert_array_equal(reference, accelerated)
        signed_zero_differences += int(np.count_nonzero(np.signbit(reference) != np.signbit(accelerated)))
        python_ms = 1000 * median(durations["python"])
        c_ms = 1000 * median(durations["c"])
        timings.append(dict(strata=h, pc_dimensions=q, python_ms=python_ms, c_ms=c_ms,
                            speedup=python_ms/c_ms))
    distances = rng.uniform(size=(686, 120))
    capacities = np.full(120, 686//120, dtype=np.intp)
    capacities[:686 % 120] += 1
    bit_equal(np.repeat(distances, capacities, axis=1), native.expanded_cost(distances, capacities))
    return dict(native=native_metadata(), exact_lattices_and_anchors=len(cases),
                anchor_signed_zero_differences=signed_zero_differences, exact_capacity_costs=True,
                cold_glp_timings=timings)


def parity(args):
    """Check edge cases and optionally time kernels or compare two raw runs."""
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    from backend import KMBackend

    py, c = KMBackend("python"), KMBackend()
    report = dict(kernel_cases=check_kernel(py, c), kernel_max_absolute_error=0.0, backend=c.metadata())
    if args.benchmark:
        report["weighted_km_timings"] = benchmark_kernel(py, c)
    if args.design_benchmark:
        report["design"] = benchmark_design()
    if args.compare:
        report["result_tables"] = compare_results(args.compare)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    env = modes.add_parser("doctor", help="Check dependencies and load the selected backend")
    env.add_argument("--backend", choices=("c", "python"), required=True)
    paper = modes.add_parser("verify", help="Match outputs against the numerical reference")
    paper.add_argument("--target", choices=("all", "simulation", "gbsg", "supplement"), required=True)
    paper.add_argument("--outputs", type=Path, required=True)
    kernels = modes.add_parser("parity", help="Check numerical agreement of the two backends")
    kernels.add_argument("--compare", nargs=2, metavar=("PYTHON_RESULTS", "C_RESULTS"))
    kernels.add_argument("--benchmark", action="store_true")
    kernels.add_argument("--design-benchmark", action="store_true")
    kernels.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "parity":
        parity(args)
    elif args.mode == "doctor":
        doctor(args.backend)
    else:
        verify(args.target, args.outputs)


if __name__ == "__main__":
    main()
