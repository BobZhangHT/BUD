#!/usr/bin/env python3
"""Run locked BUD studies with the C or reference Python KM backend.

Every command requires a fresh ``--out`` directory.  The default backend is
the compiled C implementation; ``--backend python`` is the explicit reference
fallback.  This launcher changes no archived runner, protocol, or result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
LEGACY = CODE / "legacy_locked"

# Set only in spawned workers by _worker_initializer.
_WORKER_MODULE: Any = None

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import_runner(name: str):
    """Import one archived runner with its own local import roots first."""
    roots = [str(CODE)]
    if name in {"run_confirmation", "check_fixed_grid_final"}:
        roots.insert(0, str(LEGACY))
    for root in reversed(roots):
        if root not in sys.path:
            sys.path.insert(0, root)
    return importlib.import_module(name)


def _new_backend(kind: str):
    try:
        from km_backend import KMBackend
    except ImportError as exc:
        if kind == "c":
            raise RuntimeError(
                "The C KM backend is unavailable: code/km_backend.py could not be imported. "
                "Build/install its kernel first, or explicitly use --backend python."
            ) from exc
        raise RuntimeError(
            "KM backend interface is unavailable. Expected code/km_backend.py "
            "exporting KMBackend('c'|'python')."
        ) from exc
    try:
        return KMBackend(kind)
    except Exception as exc:
        if kind == "c":
            raise RuntimeError(
                f"The C KM backend is unavailable: {exc}\n"
                "Use a supported and permitted C toolchain or explicitly select --backend python."
            ) from exc
        raise


def _patch_backend(module: Any, backend: Any, *, bootstrap: bool) -> None:
    """Patch only the runner's imported runtime callables, never its source."""
    module.km_rmst = backend.km_rmst
    if bootstrap:
        module.bootstrap_rmst = backend.bootstrap_rmst
    if backend.name == "c":
        from design_backend import install
        install(module)


def _worker_initializer(module_name: str, backend_kind: str, bootstrap: bool) -> None:
    """Windows-safe worker setup: import and patch after process spawn."""
    global _WORKER_MODULE
    _WORKER_MODULE = _import_runner(module_name)
    _patch_backend(_WORKER_MODULE, _new_backend(backend_kind), bootstrap=bootstrap)


def _simulation_task(task):
    return _WORKER_MODULE.worker(task)


def _real_case_task(case):
    return _WORKER_MODULE.one_case(case)


def _prepare_out(out: Path, command: str, args: argparse.Namespace, backend: Any,
                 runner_files: list[Path]) -> Path:
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    metadata = backend.metadata()
    if backend.name == "c":
        from design_backend import native_metadata
        metadata.update(native_metadata())
    record = {
        "command": command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": metadata,
        "counts_and_seed": {key: value for key, value in vars(args).items()
                            if key not in {"out", "command", "backend", "workers"}},
        "workers": args.workers,
        "source_sha256": {
            path.relative_to(PROJECT).as_posix(): _sha256(path)
            for path in [Path(__file__), *runner_files, CODE / "km_backend.py",
                         CODE / "design_backend.py", CODE / "paired_sample_backend.py"]
            if path.exists()
        },
        "timing": {"started_utc": datetime.now(timezone.utc).isoformat()},
    }
    (out / "orchestration.json").write_text(json.dumps(record, indent=2))
    return out


def _finish_out(out: Path, started: float) -> None:
    path = out / "orchestration.json"
    record = json.loads(path.read_text())
    # Keep elapsed runtime outside numerical result tables and runner diagnostics.
    record["timing"].update(
        finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=perf_counter() - started
    )
    path.write_text(json.dumps(record, indent=2))


def run_simulation(args: argparse.Namespace, backend: Any, out: Path) -> None:
    runner = _import_runner("run_confirmation")
    spec = dict(cohorts=args.cohorts, draws=args.draws, B=args.bootstrap_replicates,
                n_group=args.n_group, m_group=args.m_group, workers=args.workers,
                seed=args.seed, methods=["SRS", "PC1-X", "BUD-fullspace-onepass"],
                family="all", out=str(out))
    tasks = [("new-confirmation", scene, cohort, spec)
             for scene in runner.VALIDATION for cohort in range(args.cohorts)]
    tasks += [("legacy", scene, cohort, spec)
              for scene in runner.SCENES for cohort in range(args.cohorts)]
    spec["interpretation"] = (
        "Rerun of the locked scenario families. Cohorts overlap the archived runs when the same seed is supplied."
    )
    spec["source_sha256"] = {name: _sha256(LEGACY / name) for name in
                              ("designs.py", "refinement.py", "scenarios.py", "survival.py",
                               "anchor_ablation.py", "run_confirmation.py")}
    (out / "protocol.json").write_text(json.dumps(spec, indent=2))
    started = perf_counter()
    all_records, all_diagnostics = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_initializer,
                             initargs=("run_confirmation", args.backend, True)) as pool:
        futures = {pool.submit(_simulation_task, task): task[:3] for task in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            family, scene, cohort = futures[future]
            records, diagnostics = future.result()
            all_records.extend(records)
            all_diagnostics.extend(diagnostics)
            import pandas as pd
            pd.DataFrame(records).to_csv(out / f"raw_{scene}_{cohort:03d}.csv", index=False)
            pd.DataFrame(diagnostics).to_csv(out / f"diag_{scene}_{cohort:03d}.csv", index=False)
            print(f"{number}/{len(tasks)} {family} {scene} cohort={cohort} "
                  f"elapsed={perf_counter() - started:.1f}s", flush=True)
    import pandas as pd
    raw = pd.DataFrame(all_records).sort_values(["family", "scene", "cohort", "draw", "method"])
    diagnostics = pd.DataFrame(all_diagnostics).sort_values(["family", "scene", "cohort", "group", "variant"])
    raw.to_csv(out / "raw.csv", index=False)
    diagnostics.to_csv(out / "diagnostics.csv", index=False)
    runner.summarize(raw, diagnostics, out)


def run_nwts_flc(args: argparse.Namespace, backend: Any, out: Path) -> None:
    runner = _import_runner("real_application")
    _patch_backend(runner, backend, bootstrap=False)
    spec = dict(replicates=args.replicates, bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed, tau=args.tau, workers=args.workers, out=str(out))
    spec.update(cases=[["nwtco", 200], ["nwtco", 400], ["flchain", 400], ["flchain", 800]],
                target="Single-cohort full-data exact weighted KM RMST on [0,1825] days",
                method="Locked BUD85 inverse marginal quantile GLP32 MD + exactly one full-X refinement",
                capacities="First N mod H cells have ceil(N/H); other cells have floor(N/H); H=m/2",
                bootstrap="Positive finite-population corrected Rao-Wu, s-1 multinomial draws; paired for s=2",
                interval="Basic 95 percent, linear empirical quantiles, B fixed in computation",
                pairing="Common iid priorities per replicate across all three methods",
                preprocessing={"nwtco": "age numeric + all-level dummies instit,stage,study; excludes histol,in.subcohort,IDs",
                               "flchain": "age,sample.yr,mgus,sex_M; log1p kappa,lambda,creatinine; median creatinine plus missing flag; excludes flc.grp,chapter,IDs"},
                source_hashes=runner.SOURCE_HASHES,
                source_sha256={str(path.relative_to(runner.ROOT)): _sha256(path) for path in
                               [CODE / "real_application.py", CODE / "real_core" / "designs.py",
                                CODE / "real_core" / "survival.py"]})
    (out / "protocol.json").write_text(json.dumps(spec, indent=2))
    runner.verification()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_initializer,
                             initargs=("real_application", args.backend, False)) as pool:
        futures = [pool.submit(_real_case_task, (name, m, spec)) for name, m in spec["cases"]]
        for future in as_completed(futures):
            print("COMPLETE", future.result(), flush=True)
    runner.summarize(spec)


def run_fixed_grid(args: argparse.Namespace, backend: Any, out: Path) -> None:
    runner = _import_runner("check_fixed_grid_final")
    _patch_backend(runner, backend, bootstrap=True)
    runner.COHORTS, runner.DRAWS, runner.B = args.cohorts, args.draws, args.bootstrap_replicates
    saved_argv = sys.argv
    try:
        sys.argv = [str(CODE / "check_fixed_grid_final.py"), "--out", str(out)]
        runner.main()
    finally:
        sys.argv = saved_argv


def _patch_real_runtime(module: Any, backend: Any) -> None:
    """Install equivalent numerical helpers in a real-cohort runner."""
    _patch_backend(module, backend, bootstrap=False)


def run_optional_real(args: argparse.Namespace, backend: Any, out: Path) -> None:
    if args.command == "rotterdam":
        runner = _import_runner("rotterdam_application")
        _patch_real_runtime(runner, backend)
        runner.REPS, runner.B = args.replicates, args.bootstrap_replicates
        runner.run(out)
        return
    study = _import_runner("small_budget_application")
    _patch_real_runtime(study, backend)
    if args.command == "gbsg-mgus2-small":
        study.DRAWS, study.RESAMPLES, study.BUDGETS = args.replicates, args.bootstrap_replicates, (40, 80)
        study.run(out)
        return
    if args.command == "gbsg-mgus2-medium":
        study.DRAWS, study.RESAMPLES, study.BUDGETS = args.replicates, args.bootstrap_replicates, (120, 240)
        study.run(out)
        return
    if args.command == "gbsg-planning":
        study.DRAWS, study.RESAMPLES, study.BUDGETS = args.replicates, args.bootstrap_replicates, (172,)
        study.COHORTS = {"gbsg": study.COHORTS["gbsg"]}
        study.run(out)
        return
    colon = _import_runner("colon_small_budget")
    if args.command == "colon-small":
        colon.study.DRAWS, colon.study.RESAMPLES, colon.study.BUDGETS = args.replicates, args.bootstrap_replicates, (40, 80)
    else:
        colon.study.DRAWS, colon.study.RESAMPLES, colon.study.BUDGETS = args.replicates, args.bootstrap_replicates, (120, 240)
    colon.run(out)


def _common(parser: argparse.ArgumentParser, *, default_replicates: int) -> None:
    parser.add_argument("--out", type=Path, required=True, help="New output directory (required).")
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=default_replicates)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    simulation = commands.add_parser("simulation", help="12-scenario primary confirmation simulation")
    simulation.add_argument("--out", type=Path, required=True)
    simulation.add_argument("--backend", choices=("c", "python"), default="c")
    simulation.add_argument("--workers", type=int, default=2)
    simulation.add_argument("--cohorts", type=int, default=50)
    simulation.add_argument("--draws", type=int, default=20)
    simulation.add_argument("--bootstrap-replicates", type=int, default=1000)
    simulation.add_argument("--seed", type=int, default=33891027)
    simulation.add_argument("--n-group", type=int, default=960)
    simulation.add_argument("--m-group", type=int, default=120)
    real = commands.add_parser("nwts-flc", help="Original NWTS and FLC evaluation")
    _common(real, default_replicates=1000)
    real.add_argument("--seed", type=int, default=2026112201)
    real.add_argument("--tau", type=float, default=1825.0)
    fixed = commands.add_parser("fixed-grid", help="Optional fixed-grid simulation")
    fixed.add_argument("--out", type=Path, required=True)
    fixed.add_argument("--backend", choices=("c", "python"), default="c")
    fixed.add_argument("--workers", type=int, default=1, help="Recorded only; archived runner is serial.")
    fixed.add_argument("--cohorts", type=int, default=50)
    fixed.add_argument("--draws", type=int, default=20)
    fixed.add_argument("--bootstrap-replicates", type=int, default=1000)
    rotterdam = commands.add_parser("rotterdam", help="Optional Rotterdam evaluation")
    _common(rotterdam, default_replicates=1000)
    for name, help_text, reps in [
        ("gbsg-mgus2-small", "GBSG/MGUS2 small-budget evaluation", 2000),
        ("gbsg-mgus2-medium", "GBSG/MGUS2 medium-budget evaluation", 1000),
        ("gbsg-planning", "GBSG external-calibration budget of 172", 1000),
        ("colon-small", "Colon small-budget evaluation", 2000),
        ("colon-medium", "Colon medium-budget evaluation", 1000),
    ]:
        item = commands.add_parser(name, help=help_text)
        _common(item, default_replicates=reps)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers < 1 or args.bootstrap_replicates < 2:
        parser.error("Require at least one worker and two bootstrap replicates")
    if args.command in {"simulation", "fixed-grid"}:
        if args.cohorts < 2 or args.draws < 1:
            parser.error("Require at least two cohorts and one draw per cohort")
    elif args.replicates < 2:
        parser.error("Require at least two phase-II replicates")
    if args.command.startswith("colon-") and args.bootstrap_replicates != 1000:
        parser.error("The archived colon runner requires exactly 1000 bootstrap replicates")
    if args.command == "nwts-flc" and args.tau != 1825:
        parser.error("The archived NWTS/FLC protocol fixes the horizon at 1825 days")
    backend = _new_backend(args.backend)  # Compile/load before any runner or pool starts.
    runner_files = {
        "simulation": [LEGACY / "run_confirmation.py"],
        "nwts-flc": [CODE / "real_application.py"],
        "fixed-grid": [CODE / "check_fixed_grid_final.py"],
        "rotterdam": [CODE / "rotterdam_application.py"],
        "gbsg-mgus2-small": [CODE / "small_budget_application.py"],
        "gbsg-mgus2-medium": [CODE / "small_budget_application.py", CODE / "medium_real_application.py"],
        "gbsg-planning": [CODE / "small_budget_application.py",
                          PROJECT / "results/budget_planning_20261122/gbsg_external_calibration.json"],
        "colon-small": [CODE / "colon_small_budget.py", CODE / "small_budget_application.py"],
        "colon-medium": [CODE / "colon_small_budget.py", CODE / "medium_real_application.py"],
    }[args.command]
    out = _prepare_out(args.out, args.command, args, backend, runner_files)
    started = perf_counter()
    try:
        if args.command == "simulation":
            run_simulation(args, backend, out)
        elif args.command == "nwts-flc":
            run_nwts_flc(args, backend, out)
        elif args.command == "fixed-grid":
            run_fixed_grid(args, backend, out)
        else:
            run_optional_real(args, backend, out)
    except Exception as exc:
        (out / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    _finish_out(out, started)


if __name__ == "__main__":
    main()
