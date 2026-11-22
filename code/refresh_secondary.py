"""Repeat the matched-anchor and budget-sensitivity studies in a new directory."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import importlib
import json
from pathlib import Path
import sys

import bud_experiment


def initialize(runner, backend_name):
    sys.path.insert(0, str(bud_experiment.LEGACY))
    module = importlib.import_module(runner)
    backend = bud_experiment._new_backend(backend_name)
    bud_experiment._patch_backend(module, backend, bootstrap=runner == "run_confirmation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("anchors", "sensitivity"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    args = parser.parse_args()
    output = args.root.resolve()
    if args.study == "anchors":
        import rerun_1000 as driver
        driver.OUTPUT = output
        driver.ProcessPoolExecutor = partial(ProcessPoolExecutor, initializer=initialize,
                                             initargs=("final_anchor_ablation", args.backend))
        driver.run_anchors(args.workers)
        out = output / "anchor_ablation"
    else:
        import simulation_budget_sensitivity as driver
        if not (output / "confirmation" / "raw.csv").is_file():
            parser.error("Sensitivity needs the refreshed confirmation raw.csv")
        driver.BASE = output
        def pool(max_workers):
            return ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                       initargs=("run_confirmation", args.backend))
        driver.ProcessPoolExecutor = pool
        driver.main()
        out = output / "simulation_budget_sensitivity"
    backend_metadata = bud_experiment._new_backend(args.backend).metadata()
    if args.backend == "c":
        from design_backend import native_metadata
        backend_metadata.update(native_metadata())
    files = [Path(__file__), Path(driver.__file__), Path(bud_experiment.__file__),
             bud_experiment.CODE / "design_backend.py", bud_experiment.CODE / "paired_sample_backend.py"]
    record = dict(backend=backend_metadata,
                  sha256={p.relative_to(bud_experiment.PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in files}, workers=args.workers)
    (out / "refresh_provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
