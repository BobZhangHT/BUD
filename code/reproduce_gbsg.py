"""Reproduce the GBSG application and its external Rotterdam calibration.

This module is called by the release-root ``reproduce.py``.  It deliberately
uses subprocesses so each archived driver starts with ordinary script imports
from the standalone release root.  No source file or archived output is
overwritten: every target directory must be absent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
DATA = ROOT / "data"

DATASETS = {
    "gbsg": (
        "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
        "cc03c29690889dbe83089f5206e2422db8c3f71f/csv/survival/gbsg.csv",
        "9fa0fef0575d04d3273869be28405307f1b02993751498e8a0c844541cccd5d9",
    ),
    "rotterdam": (
        "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
        "cc03c29690889dbe83089f5206e2422db8c3f71f/csv/survival/rotterdam.csv",
        "62703670d3be5d49c5476fb7131b020bdca9f1c25bb4522fd6abe8f9dab61f1b",
    ),
}


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env


def _checked_data(name: str) -> Path:
    """Fetch one pinned public CSV once, retaining only checksum-matched bytes."""
    url, expected = DATASETS[name]
    path = DATA / f"{name}.csv"
    DATA.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = path.read_bytes()
    else:
        request = Request(url, headers={"User-Agent": "BUD reproducibility/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{name}: pinned source checksum changed ({actual})")
        path.write_bytes(payload)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{name}: local bytes have checksum {actual}, expected {expected}")
    return path


def _command(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, env=_environment(), check=True)


def _grid(budgets: tuple[int, ...], replicates: int, bootstrap_replicates: int,
          backend: str, out: Path) -> None:
    """Run archived GBSG mechanics only, without MGUS2 rows."""
    # This child command starts with code/ on sys.path, as a normal script
    # import.  It is intentionally not imported by the root dispatcher.
    import bud_experiment as engine

    study = engine._import_runner("small_budget_application")
    selected_backend = engine._new_backend(backend)
    engine._patch_real_runtime(study, selected_backend)
    args = argparse.Namespace(
        out=out, command="gbsg-release-grid", backend=backend, workers=1,
        budgets=budgets, replicates=replicates, bootstrap_replicates=bootstrap_replicates,
    )
    prepared = engine._prepare_out(
        out, args.command, args, selected_backend,
        [Path(__file__), Path(study.__file__), engine.CODE / "real_application.py",
         engine.CODE / "real_core" / "designs.py", engine.CODE / "real_core" / "survival.py",
         engine.PROJECT / "handoff" / "SMALL_BUDGET_PROTOCOL.md"],
    )
    study.BUDGETS = budgets
    study.DRAWS, study.RESAMPLES = replicates, bootstrap_replicates
    study.COHORTS = {"gbsg": study.COHORTS["gbsg"]}
    from time import perf_counter
    started = perf_counter()
    try:
        study.run(prepared)
    except Exception as exc:
        (prepared / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    engine._finish_out(prepared, started)


def _grid_cli(args: argparse.Namespace) -> None:
    _grid(tuple(args.budgets), args.replicates, args.bootstrap_replicates, args.backend, args.out)


def _subprocess_grid(budgets: tuple[int, ...], replicates: int, bootstrap_replicates: int,
                     backend: str, out: Path) -> tuple[str, ...]:
    return (
        str(CODE / "reproduce_gbsg.py"), "grid", "--backend", backend,
        "--replicates", str(replicates), "--bootstrap-replicates", str(bootstrap_replicates),
        "--out", str(out), "--budgets", *(str(value) for value in budgets),
    )


def run(backend: str, workers: int, quick: bool) -> None:
    """Run the GBSG release suite from a standalone release root.

    ``quick`` writes a two-draw smoke result only.  A full run writes the
    canonical seven-budget output directories and the Rotterdam evaluation.
    ``workers`` limits concurrent independent subprocesses (the archived
    drivers themselves are serial to preserve their seeded draw order).
    """
    if backend not in {"c", "python"}:
        raise ValueError("backend must be 'c' or 'python'")
    if workers < 1:
        raise ValueError("workers must be positive")
    _checked_data("gbsg")
    _checked_data("rotterdam")
    if quick:
        _command(*_subprocess_grid((40,), 2, 10, backend, ROOT / "smoke_results" / "gbsg"))
        return

    jobs = [
        _subprocess_grid((40, 80), 2000, 1000, backend,
                         ROOT / "results" / "budget_refresh_20261122" / "small_budgets"),
        _subprocess_grid((120, 240), 1000, 1000, backend,
                         ROOT / "results" / "budget_refresh_20261122" / "medium_gbsg"),
        _subprocess_grid((160, 200), 1000, 1000, backend,
                         ROOT / "results" / "bud_budget_20261122" / "gbsg_grid_160_200"),
    ]
    # Preserve the three published grid shapes. Rotterdam is an external,
    # checksum-pinned planning pilot here, not a retrospective evaluation.
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [pool.submit(_command, *job) for job in jobs]
        for future in futures:
            future.result()

    _command(str(CODE / "gbsg_bud_only_budget.py"), "plan")
    _command(str(CODE / "gbsg_bud_only_budget.py"), "evaluate", "--backend", backend)


def main() -> None:
    # ``grid`` is private child plumbing.  The other form is the interface
    # used by the release-root dispatcher: reproduce.py gbsg --backend ...
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("grid")
        parser.add_argument("--budgets", nargs="+", type=int, required=True)
        parser.add_argument("--replicates", type=int, required=True)
        parser.add_argument("--bootstrap-replicates", type=int, required=True)
        parser.add_argument("--backend", choices=("c", "python"), default="c")
        parser.add_argument("--out", type=Path, required=True)
        _grid_cli(parser.parse_args())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.data_only:
        for name in DATASETS:
            _checked_data(name)
    else:
        run(args.backend, args.workers, args.quick)


if __name__ == "__main__":
    main()
