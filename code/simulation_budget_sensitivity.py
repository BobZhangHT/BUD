"""Compare three sampling fractions in four representative simulation settings."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "code" / "legacy_locked"
sys.path.insert(0, str(LOCKED))
from run_confirmation import worker

import pandas as pd
import numpy as np
from scipy.stats import t

BASE = ROOT/"results/rerun_1000"
SENTINELS = {"val_pc1": "new-confirmation", "val_multivariate": "new-confirmation",
             "val_null": "new-confirmation", "saddle3d": "legacy"}
METHODS = ["SRS", "PC1-X", "BUD-fullspace-onepass"]


def main():
    out = BASE/"simulation_budget_sensitivity"
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), *LOCKED.rglob("*.py")]
    protocol = dict(base_seed=33891027, n_group=960, budgets=[60, 120, 240], cohorts=50, draws=20,
                    B=1000, sentinels=SENTINELS, methods=METHODS, original_input="confirmation/raw.csv",
                    sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in paths})
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2))
    more = []
    base_spec = dict(seed=33891027, n_group=960, draws=20, B=1000, methods=METHODS)
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {}
        for budget in (60, 240):
            spec = dict(**base_spec, m_group=budget)
            for scene, family in SENTINELS.items():
                for cohort in range(50):
                    f = pool.submit(worker, (family, scene, cohort, spec))
                    futures[f] = (budget, scene, cohort)
        for done, f in enumerate(as_completed(futures), 1):
            budget, scene, cohort = futures[f]
            rows, diagnostics = f.result()
            frame = pd.DataFrame(rows).assign(m=budget)
            more.append(frame)
            frame.to_csv(out/f"raw_{budget}_{scene}_{cohort:03d}.csv.gz", index=False)
            if done % 50 == 0:
                print(f"Budget sensitivity {done}/{len(futures)}", flush=True)
    reference = pd.read_csv(BASE/"confirmation"/"raw.csv")
    reference = reference[reference.scene.isin(SENTINELS)].assign(m=120)
    full = pd.concat([reference, *more], ignore_index=True)
    full = full.sort_values(["scene", "m", "cohort", "draw", "method"])
    full.to_csv(out/"raw.csv.gz", index=False)
    raw = full[full.method.isin(METHODS)].copy()
    assert len(raw) == 4*3*3*1000 and not raw.duplicated(["scene", "m", "method", "cohort", "draw"]).any()
    assert raw.groupby(["scene", "m", "method"]).size().eq(1000).all()
    raw["sq_error"] = raw.error**2
    raw["empty_tail"] = raw.sample_risk_tau_min.eq(0)
    blocks = raw.groupby(["scene", "m", "method", "cohort"], as_index=False).agg(
        mse=("sq_error", "mean"), coverage=("coverage", "mean"), width=("width", "mean"),
        empty_tail=("empty_tail", "sum"),
    )
    summary = blocks.groupby(["scene", "m", "method"], as_index=False).agg(
        mse=("mse", "mean"), coverage=("coverage", "mean"), width=("width", "mean"),
        coverage_mcse=("coverage", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
        empty_tail_count=("empty_tail", "sum"),
    )
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    margin = t.ppf(.975, 49)*summary.coverage_mcse
    summary["coverage_mc_low"] = (summary.coverage-margin).clip(lower=0)
    summary["coverage_mc_high"] = (summary.coverage+margin).clip(upper=1)
    summary.to_csv(out/"summary.csv", index=False)
    blocks.to_csv(out/"cohort_metrics.csv", index=False)


if __name__ == "__main__":
    main()
