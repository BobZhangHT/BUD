#!/usr/bin/env python3
"""Reproduce supplementary checks, sensitivity analyses, and anchor ablations.

Run the normal release target with ``--backend``, ``--workers``, and optional
``--quick``. Use ``anchors`` to resume the chunked S12 anchor study.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

# Importing the engine first establishes single-threaded numerical libraries
# before NumPy or SciPy is loaded, including in direct anchor-resume workers.
import bud_experiment

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
LEGACY = CODE / "legacy_locked"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
METHODS = ["SRS", "PC1-X", "BUD-fullspace-onepass"]
SENTINELS = {"val_pc1": "new-confirmation", "val_multivariate": "new-confirmation",
             "val_null": "new-confirmation", "saddle3d": "legacy"}
_WORKER_MODULE = None
# SHA-256 of the previous anchor driver with LF and CRLF checkouts.
ANCHOR_DRIVER_HASHES = {
    "92bced30917ffb23bb2bda6ebb549fb0db758cb807d534112fc317119700c868",
    "549435dd9e230413225d0d1ad5bfbb0c1c49f50cc17ce997d12a7be6a0481c86",
}


def sha(path):
    return bud_experiment._sha256(path)


def _run(*arguments: str) -> None:
    """Run one release script in a clean interpreter from the project root."""
    command = [sys.executable, "-B", *arguments]
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} is required but was not found: {path}")


def _import_locked(name):
    """Import an archived bare module after placing its directory first."""
    legacy = str(LEGACY)
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    return bud_experiment._import_runner(name)


def initialize_worker(runner, backend_name):
    """Patch the archived runner after Windows spawns each worker process."""
    global _WORKER_MODULE
    _WORKER_MODULE = _import_locked(runner)
    backend = bud_experiment._new_backend(backend_name)
    bud_experiment._patch_backend(
        _WORKER_MODULE, backend, bootstrap=runner == "run_confirmation")


def anchor_chunk(tasks):
    """Evaluate one resumable group of anchor-ablation cohorts."""
    if _WORKER_MODULE is None:
        raise RuntimeError("anchor worker was not initialized")
    return [row for task in tasks for row in _WORKER_MODULE.one_cohort(task)]


def simulation_task(task):
    """Evaluate one independent sensitivity cohort in an isolated worker."""
    if _WORKER_MODULE is None:
        raise RuntimeError("simulation worker was not initialized")
    return _WORKER_MODULE.worker(task)


def backend_record(name):
    metadata = bud_experiment._new_backend(name).metadata()
    if name == "c":
        from backend import native_metadata
        metadata.update(native_metadata())
    return metadata


def write_refresh_provenance(out, backend, workers):
    """Record executable sources; archived protocols retain their own hashes."""
    files = [Path(__file__), Path(bud_experiment.__file__), CODE / "backend.py"]
    record = {"backend": backend_record(backend),
              "sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
              "workers": workers}
    (out / "refresh_provenance.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")


def anchor_protocol(study):
    families = {"legacy": list(study.LEGACY_SCENES), "new": list(study.VALIDATION)}
    spec = {"seed": 975314, "n_group": 960, "m_group": 120, "cohorts": 1000,
            "families": families, "methods": list(study.METHODS),
            "bootstrap_applicable": False, "chunk_size": 10,
            "driver_sha256": sha(Path(__file__)),
            "original_study_sha256": sha(Path(study.__file__))}
    spec["locked_sources"] = {
        p.relative_to(ROOT).as_posix(): sha(p) for p in LEGACY.rglob("*.py")}
    return spec


def validate_anchor_protocol(saved, expected):
    """Validate a resumable protocol, allowing only its published driver alias.

    The historical archive recorded the previous driver in this one field.
    All scientific fields and every locked-source digest must still match exactly.
    """
    saved = dict(saved)
    expected = dict(expected)
    saved_driver = str(saved.pop("driver_sha256", "")).lower()
    current_driver = str(expected.pop("driver_sha256")).lower()
    if saved != expected:
        raise RuntimeError("Saved anchor protocol differs")
    if saved_driver != current_driver and saved_driver not in ANCHOR_DRIVER_HASHES:
        raise RuntimeError("Saved anchor protocol has an unrecognized driver hash")


def run_anchors(root, workers, backend):
    """Resume or start the chunked 1,000-cohort anchor ablation."""
    study = _import_locked("final_anchor_ablation")
    out = root / "anchor_ablation"
    out.mkdir(parents=True, exist_ok=True)
    spec = anchor_protocol(study)
    protocol = out / "protocol.json"
    if protocol.exists():
        # Read for validation only.  Keeping the existing bytes intact retains
        # the published provenance record during a resumed calculation.
        validate_anchor_protocol(json.loads(protocol.read_bytes()), spec)
    else:
        protocol.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    jobs = {}
    for family, scenes in spec["families"].items():
        for scene in scenes:
            for initial in range(0, spec["cohorts"], spec["chunk_size"]):
                cohorts = range(initial, initial + spec["chunk_size"])
                path = out / f"chunk_{family}_{scene}_{initial:04d}.csv.gz"
                jobs[path] = [(family, scene, cohort, spec) for cohort in cohorts]
    pending = {path: tasks for path, tasks in jobs.items() if not path.exists()}
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             initargs=("final_anchor_ablation", backend)) as pool:
        futures = {pool.submit(anchor_chunk, tasks): path for path, tasks in pending.items()}
        completed = len(jobs) - len(pending)
        for future in as_completed(futures):
            path = futures[future]
            rows = future.result()
            if len(rows) != spec["chunk_size"] * 6:
                raise RuntimeError(f"Invalid anchor chunk {path.name}")
            temporary = path.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False, compression="gzip")
            temporary.replace(path)
            completed += 1
            if completed % 10 == 0:
                print(f"Anchor chunks {completed}/{len(jobs)}, "
                      f"{perf_counter() - started:.1f}s", flush=True)

    data = pd.concat([pd.read_csv(path) for path in jobs], ignore_index=True)
    keys = ["family", "scene", "cohort", "group", "variant"]
    expected = len(jobs) * spec["chunk_size"] * 6
    if (len(data) != expected or data.duplicated(keys).any()
            or not data.groupby(keys[:2]).size().eq(6000).all()):
        raise RuntimeError("Invalid completed anchor archive")
    if not (data.full_after <= data.full_before + 1e-9).all():
        raise RuntimeError("Anchor scatter increased")
    data = data.sort_values(keys)
    data.to_csv(out / "diagnostics.csv.gz", index=False)
    study.summarize(data, out)
    write_refresh_provenance(out, backend, workers)


def run_sensitivity(root, reference_raw, workers, backend):
    """Run the prespecified S12 budgets while reusing archived m=120 draws."""
    out = root / "simulation_budget_sensitivity"
    out.mkdir(parents=True, exist_ok=False)
    input_path = reference_raw.relative_to(ROOT) if reference_raw.is_relative_to(ROOT) else reference_raw
    paths = [Path(__file__), *LEGACY.rglob("*.py")]
    protocol = {"base_seed": 33891027, "n_group": 960, "budgets": [60, 120, 240],
                "cohorts": 50, "draws": 20, "B": 1000, "sentinels": SENTINELS,
                "methods": METHODS, "original_input": input_path.as_posix(),
                "sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    more = []
    base_spec = {"seed": 33891027, "n_group": 960, "draws": 20,
                 "B": 1000, "methods": METHODS}
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             initargs=("run_confirmation", backend)) as pool:
        futures = {}
        for budget in (60, 240):
            spec = dict(**base_spec, m_group=budget)
            for scene, family in SENTINELS.items():
                for cohort in range(50):
                    future = pool.submit(simulation_task, (family, scene, cohort, spec))
                    futures[future] = (budget, scene, cohort)
        for done, future in enumerate(as_completed(futures), 1):
            budget, scene, cohort = futures[future]
            rows, _diagnostics = future.result()
            frame = pd.DataFrame(rows).assign(m=budget)
            more.append(frame)
            frame.to_csv(out / f"raw_{budget}_{scene}_{cohort:03d}.csv.gz", index=False)
            if done % 50 == 0:
                print(f"Budget sensitivity {done}/{len(futures)}", flush=True)

    reference = pd.read_csv(reference_raw)
    reference = reference[reference.scene.isin(SENTINELS)].assign(m=120)
    full = pd.concat([reference, *more], ignore_index=True)
    full = full.sort_values(["scene", "m", "cohort", "draw", "method"])
    full.to_csv(out / "raw.csv.gz", index=False)
    raw = full[full.method.isin(METHODS)].copy()
    assert len(raw) == 4 * 3 * 3 * 1000
    assert not raw.duplicated(["scene", "m", "method", "cohort", "draw"]).any()
    assert raw.groupby(["scene", "m", "method"]).size().eq(1000).all()
    raw["sq_error"] = raw.error ** 2
    raw["empty_tail"] = raw.sample_risk_tau_min.eq(0)
    blocks = raw.groupby(["scene", "m", "method", "cohort"], as_index=False).agg(
        mse=("sq_error", "mean"), coverage=("coverage", "mean"), width=("width", "mean"),
        empty_tail=("empty_tail", "sum"))
    summary = blocks.groupby(["scene", "m", "method"], as_index=False).agg(
        mse=("mse", "mean"), coverage=("coverage", "mean"), width=("width", "mean"),
        coverage_mcse=("coverage", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        empty_tail_count=("empty_tail", "sum"))
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    margin = t.ppf(.975, 49) * summary.coverage_mcse
    summary["coverage_mc_low"] = (summary.coverage - margin).clip(lower=0)
    summary["coverage_mc_high"] = (summary.coverage + margin).clip(upper=1)
    summary.to_csv(out / "summary.csv", index=False)
    blocks.to_csv(out / "cohort_metrics.csv", index=False)
    write_refresh_provenance(out, backend, workers)


def run_release(backend, workers, quick):
    """Run the public S11--S12 target without changing archived algorithms."""
    _require(CODE / "bud_experiment.py", "shared experiment launcher")
    _require(CODE / "backend.py", "shared numerical backend")
    _require(LEGACY / "survival.py", "locked survival implementation")
    started = perf_counter()
    if quick:
        out = ROOT / "smoke_results" / "supplement"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite existing smoke output: {out}")
        _run("code/theory_bootstrap.py", "--profile", "smoke", "--out", str(out / "bootstrap"))
        print(f"Supplement smoke completed in {perf_counter() - started:.1f}s", flush=True)
        return

    theory = RESULTS / "theory_validation"
    refresh_root = RESULTS / "budget_refresh"
    archived_reference = RESULTS / "archived_reference" / "confirmation_120"
    fixed = refresh_root / "fixed_grid"
    sensitivity = refresh_root / "simulation_budget_sensitivity"
    anchors = refresh_root / "anchor_ablation"
    for path in (theory, fixed, sensitivity, anchors):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing full output: {path}")
    _run("code/theory_geometry.py", "--out", str(theory / "geometry.json"))
    _run("code/theory_bootstrap.py", "--profile", "full", "--out", str(theory / "bootstrap"))
    _run("code/theory.py", "--study", str(theory), "--figures", str(FIGURES))
    _run("code/bud_experiment.py", "fixed-grid", "--backend", backend, "--workers", str(workers),
         "--cohorts", "50", "--draws", "20", "--bootstrap-replicates", "1000",
         "--out", str(fixed))
    confirmation = archived_reference / "raw.csv"
    if not confirmation.is_file():
        _run("code/bud_experiment.py", "simulation", "--backend", backend,
             "--workers", str(workers), "--cohorts", "50", "--draws", "20",
             "--bootstrap-replicates", "1000", "--m-group", "120", "--seed", "33891027",
             "--out", str(archived_reference))
    _require(confirmation, "archived 50-by-20 reference output for S12 budget sensitivity")
    run_sensitivity(refresh_root, confirmation, workers, backend)
    run_anchors(refresh_root, workers, backend)
    print(f"Supplement full run completed in {perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("anchors", "sensitivity"),
                        help="Run one of the S12 studies, including anchor resume.")
    parser.add_argument("--root", type=Path, help="Result root for a single S12 study.")
    parser.add_argument("--reference-raw", type=Path, help="Earlier 120-per-group draws for sensitivity.")
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.reference_raw is not None and args.command != "sensitivity":
        parser.error("--reference-raw requires the sensitivity command")
    if args.command is None:
        if args.root is not None:
            parser.error("--root is only available for a single S12 study")
        run_release(args.backend, args.workers, args.quick)
        return

    if args.quick:
        parser.error("--quick is only available for the normal supplement run")
    root = (args.root or RESULTS / "budget_refresh").resolve()
    if args.command == "anchors":
        run_anchors(root, args.workers, args.backend)
    else:
        reference_raw = (args.reference_raw or root / "confirmation" / "raw.csv").resolve()
        _require(reference_raw, "earlier 50-cohort, 20-draw reference output")
        run_sensitivity(root, reference_raw, args.workers, args.backend)


if __name__ == "__main__":
    main()
