"""Run the prespecified GBSG and MGUS2 small-budget comparison."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
import traceback
from urllib.request import urlopen

from real_application import ROOT, build_partitions, seed
from survival import bootstrap_weights, draw_sample, km_rmst
from validate_real_application import direct_integral

import numpy as np
import pandas as pd
from scipy.stats import norm

REVISION = "cc03c29690889dbe83089f5206e2422db8c3f71f"
BASE_SEED, DRAWS, RESAMPLES, MC_DRAWS = 2026112203, 2000, 499, 4000
BUDGETS = (40, 80)
COHORTS = {
    "gbsg": dict(n=686, id="pid", time="rfstime", event="status", tau=1825,
                 numeric=["age", "size"], log=["nodes", "pgr", "er"],
                 indicators={"meno": 1}, categorical=["grade"]),
    "mgus2": dict(n=1384, id="id", time="futime", event="death", tau=60,
                  numeric=["age", "hgb"], log=["creat", "mspike"],
                  indicators={"sex": "M"}, categorical=[]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cohort(name, spec):
    url = f"https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/{REVISION}/csv/survival/{name}.csv"
    path = ROOT / "data" / f"{name}.csv"
    if not path.exists():
        with urlopen(url, timeout=30) as response:
            data = response.read()
        with path.open("xb") as stream:
            stream.write(data)
    original = pd.read_csv(path)
    subject = original[spec["id"]]
    if len(original) != spec["n"] or subject.isna().any() or not subject.is_unique:
        raise ValueError(f"{name}: unexpected count or invalid patient identifiers")
    original["source_row"] = np.arange(1, len(original) + 1)
    order = np.random.default_rng(seed(BASE_SEED, name, "row_order")).permutation(len(original))
    frame = original.iloc[order].reset_index(drop=True)
    x = pd.DataFrame(index=frame.index)
    for key in spec["numeric"] + spec["log"]:
        value = frame[key].astype(float)
        if value.isna().any():
            x[f"{key}_missing"] = value.isna().astype(float)
        value = value.fillna(value.median())
        x[key] = np.log1p(value) if key in spec["log"] else value
    for key, level in spec["indicators"].items():
        x[key] = (frame[key] == level).astype(float)
        if frame[key].isna().any():
            x[f"{key}_missing"] = frame[key].isna().astype(float)
    for key in spec["categorical"]:
        indicators = pd.get_dummies(frame[key].astype("string").fillna("missing"), prefix=key, dtype=float)
        x = x.join(indicators)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError(f"{name}: invalid covariate frame")
    return frame, x, {"source": url, "sha256": sha(path), "columns": x.columns.tolist()}


def evaluate(name, m, frame, time, event, tau, truth, partitions):
    rows, first = [], []
    start = perf_counter()
    for draw in range(DRAWS):
        rng = np.random.default_rng(seed(BASE_SEED, name, m, draw, "sample"))
        priorities = rng.random(len(frame))
        for method, (labels, allocation) in partitions.items():
            ids, weights, cells = draw_sample(labels, allocation, rng, priorities=priorities)
            if len(ids) != m or len(np.unique(ids)) != m or not np.isclose(weights.sum(), len(frame)):
                raise RuntimeError(f"Invalid sample: {(name, m, draw, method)}")
            t, e = time[ids], event[ids]
            estimate = km_rmst(t, e, weights, tau)
            boot_rng = np.random.default_rng(seed(BASE_SEED, name, m, draw, method, "bootstrap"))
            boot = km_rmst(t, e, bootstrap_weights(weights, cells, boot_rng, RESAMPLES), tau)
            if not np.isfinite(estimate) or not np.isfinite(boot).all():
                raise RuntimeError(f"Nonfinite estimate: {(name, m, draw, method)}")
            low, high = np.quantile(boot - estimate, [.025, .975], method="linear")
            lower, upper = estimate - high, estimate - low
            rows.append((name, m, draw, method, estimate, estimate - truth, lower, upper,
                         lower <= truth <= upper, upper - lower, int(np.count_nonzero(t >= tau))))
            if draw == 0:
                np.testing.assert_allclose(estimate, direct_integral(t, e, weights, tau), atol=1e-8, rtol=0)
                first.extend((name, m, method, int(i), int(h), w)
                             for i, h, w in zip(frame.source_row.iloc[ids], labels[ids], weights))
    return rows, first, perf_counter() - start


def summarize(raw):
    metrics = raw.assign(squared_error=raw.error**2, empty=raw.risk_at_tau.eq(0))
    summary = metrics.groupby(["cohort", "m", "method"], as_index=False).agg(
        draws=("error", "size"), mse=("squared_error", "mean"), bias=("error", "mean"),
        coverage=("covered", "mean"), mean_width=("width", "mean"),
        min_risk_at_tau=("risk_at_tau", "min"), empty_tail_count=("empty", "sum"),
    )
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    summary["coverage_mcse"] = np.sqrt(summary.coverage * (1 - summary.coverage) / summary.draws)
    assert summary.draws.eq(DRAWS).all()
    pairs = []
    for (name, m), subset in metrics.groupby(["cohort", "m"]):
        w = subset.pivot(index="draw", columns="method", values="squared_error")
        rng = np.random.default_rng(seed(BASE_SEED, name, m, "paired_mc"))
        ix = rng.integers(0, DRAWS, (MC_DRAWS, DRAWS))
        bud = w.BUD.to_numpy()
        boot_bud = bud[ix].mean(axis=1)
        for method in ("SRS", "PC1-X"):
            other = w[method].to_numpy()
            reduction = 100 * (1 - np.sqrt(bud.mean()/other.mean()))
            low, high = np.quantile(100 * (1 - np.sqrt(boot_bud/other[ix].mean(axis=1))), [.025, .975])
            difference = bud - other
            se = difference.std(ddof=1)/np.sqrt(DRAWS)
            pairs.append(dict(cohort=name, m=m, comparator=method, rmse_reduction_pct=reduction,
                              mc_low=low, mc_high=high, rmse_reduction=np.sqrt(other.mean())-np.sqrt(bud.mean()),
                              p_value=2*norm.sf(abs(difference.mean()/se))))
    paired = pd.DataFrame(pairs)
    ordered = paired.p_value.sort_values()
    paired["p_holm"] = (ordered * np.arange(len(ordered), 0, -1)).cummax().clip(upper=1)
    return summary, paired


def run(out):
    files = [Path(__file__), ROOT/"code/real_application.py",
             ROOT/"code/real_core/designs.py", ROOT/"code/real_core/survival.py",
             ROOT/"code/validate_real_application.py"]
    provenance = {
        "seed": BASE_SEED, "draws": DRAWS, "resamples": RESAMPLES, "mc_draws": MC_DRAWS,
        "budgets": BUDGETS, "definitions": COHORTS,
        "sources": {path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
        "cohorts": {},
    }
    cases, diagnostics = {}, {}
    for name, spec in COHORTS.items():
        frame, x, provenance["cohorts"][name] = load_cohort(name, spec)
        for m in BUDGETS:
            designs, geometry = build_partitions(x.to_numpy(), m, seed(BASE_SEED, name, m, "design"))
            cases[name, m] = (frame, designs)
            diagnostics[f"{name}_{m}"] = geometry
            pd.DataFrame({"source_row": frame.source_row, **{k: v[0] for k, v in designs.items()}}).to_csv(
                out/f"{name}_m{m}_partitions.csv", index=False)
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2))

    rows, first = [], []
    for (name, m), (frame, designs) in cases.items():
        spec = COHORTS[name]
        t, e = frame[spec["time"]].to_numpy(float), frame[spec["event"]].to_numpy(float)
        tau = spec["tau"]
        if not np.isfinite(t).all() or (t < 0).any() or not np.isin(e, [0, 1]).all():
            raise ValueError(f"{name}: invalid survival observations")
        weights = np.ones(len(t))
        truth = km_rmst(t, e, weights, tau)
        direct = direct_integral(t, e, weights, tau)
        np.testing.assert_allclose(truth, direct, atol=1e-8, rtol=0)
        run_rows, selections, seconds = evaluate(name, m, frame, t, e, tau, truth, designs)
        rows.extend(run_rows)
        first.extend(selections)
        diagnostics[f"{name}_{m}"].update(tau=tau, full_rmst=truth, direct_full_rmst=direct,
                                          full_risk_at_tau=int((t >= tau).sum()),
                                          full_event_count=int(e.sum()), evaluation_seconds=seconds)
        print(f"Complete: {name}, m={m}, {seconds:.1f} seconds", flush=True)
    raw = pd.DataFrame(rows, columns=["cohort", "m", "draw", "method", "estimate", "error", "lower", "upper",
                                      "covered", "width", "risk_at_tau"])
    assert len(raw) == len(cases)*DRAWS*3 and not raw.duplicated(["cohort", "m", "draw", "method"]).any()
    raw.to_csv(out/"raw.csv.gz", index=False)
    raw.loc[raw.draw.eq(0)].to_csv(out/"first_draw_estimates.csv", index=False)
    pd.DataFrame(first, columns=["cohort", "m", "method", "source_row", "stratum", "weight"]).to_csv(
        out/"first_draw_selected.csv", index=False)
    summary, paired = summarize(raw)
    summary.to_csv(out/"summary.csv", index=False)
    paired.to_csv(out/"paired.csv", index=False)
    (out/"diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/small_budgets_20261122"))
    out = ROOT / parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(out)
    except Exception:
        (out/"failure.txt").write_text(traceback.format_exc())
        raise
