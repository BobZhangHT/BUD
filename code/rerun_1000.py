"""Launch archived experiment routines with the revised numerical counts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "rerun_1000_20261122"
LOCKED = ROOT / "code" / "legacy_locked"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(study):
    return {"driver_sha256": sha(Path(__file__)),
            "original_study_sha256": sha(Path(study.__file__))}


def run_real(study, name):
    out = OUTPUT / name
    out.mkdir(parents=True, exist_ok=False)
    (out / "rerun_driver.json").write_text(json.dumps(provenance(study), indent=2))
    study.B = 1000
    if name == "small_budgets":
        study.RESAMPLES = 1000
    study.run(out)


def anchor_chunk(tasks):
    sys.path.insert(0, str(LOCKED))
    from final_anchor_ablation import one_cohort

    return [row for task in tasks for row in one_cohort(task)]


def run_anchors(workers):
    sys.path.insert(0, str(LOCKED))
    import final_anchor_ablation as study
    import pandas as pd

    out = OUTPUT / "anchor_ablation"
    out.mkdir(parents=True, exist_ok=True)
    families = {"legacy": list(study.LEGACY_SCENES), "new": list(study.VALIDATION)}
    spec = {"seed": 975314, "n_group": 960, "m_group": 120, "cohorts": 1000,
            "families": families, "methods": list(study.METHODS), "bootstrap_applicable": False,
            "chunk_size": 10, **provenance(study)}
    spec["locked_sources"] = {p.relative_to(ROOT).as_posix(): sha(p) for p in LOCKED.rglob("*.py")}
    protocol = out / "protocol.json"
    if protocol.exists():
        if json.loads(protocol.read_text()) != spec:
            raise RuntimeError("Saved anchor protocol differs")
    else:
        protocol.write_text(json.dumps(spec, indent=2))
    jobs = {}
    for family, scenes in families.items():
        for scene in scenes:
            for initial in range(0, spec["cohorts"], spec["chunk_size"]):
                cohorts = range(initial, initial + spec["chunk_size"])
                path = out / f"chunk_{family}_{scene}_{initial:04d}.csv.gz"
                jobs[path] = [(family, scene, cohort, spec) for cohort in cohorts]
    pending = {p: tasks for p, tasks in jobs.items() if not p.exists()}
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
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
                print(f"Anchor chunks {completed}/{len(jobs)}, {perf_counter()-start:.1f}s", flush=True)
    data = pd.concat([pd.read_csv(path) for path in jobs], ignore_index=True)
    keys = ["family", "scene", "cohort", "group", "variant"]
    expected = len(jobs) * spec["chunk_size"] * 6
    if len(data) != expected or data.duplicated(keys).any() or not data.groupby(keys[:2]).size().eq(6000).all():
        raise RuntimeError("Invalid completed anchor archive")
    if not (data.full_after <= data.full_before + 1e-9).all():
        raise RuntimeError("Anchor scatter increased")
    data = data.sort_values(keys)
    data.to_csv(out / "diagnostics.csv.gz", index=False)
    study.summarize(data, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("rotterdam", "small_budgets", "anchors"))
    parser.add_argument("--workers", type=int, default=16, help="Anchor ablation process count")
    args = parser.parse_args()
    if args.study == "anchors":
        run_anchors(args.workers)
    elif args.study == "rotterdam":
        import rotterdam_application

        run_real(rotterdam_application, args.study)
    else:
        import small_budget_application

        run_real(small_budget_application, args.study)


if __name__ == "__main__":
    main()
