"""Regenerate all four paper simulation budgets, summaries, and figures."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REFRESH = ROOT / "results/budget_refresh_20261122"
COMMON = ROOT / "results/budget_optimization_20261122"
BUD_ONLY = ROOT / "results/bud_budget_20261122"


def script(name, *args):
    command = [sys.executable, "-u", str(ROOT / "code" / f"{name}.py"), *map(str, args)]
    print(">", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summarize(raw, table_dir, figures, *, cohorts, grouped=False):
    script("analyze_simulations", "--confirmation", raw, "--out", table_dir,
           "--figures", figures, "--cohorts", cohorts, *(["--planned-table"] if grouped else []))


def run(backend, workers, quick):
    backend_options = ("--backend", backend, "--workers", workers)
    if quick:
        # Exercise the same scenario generators, designs, analysis and figure paths.
        # A quick run uses one reference budget and cannot be verified as paper results.
        out = ROOT / "smoke_results/simulation"
        raw = out / "confirmation"
        script("bud_experiment", "simulation", "--out", raw, "--cohorts", 2, "--draws", 20,
               "--bootstrap-replicates", 17, *backend_options)
        summarize(raw, out / "tables", out / "figures", cohorts=2)
        return

    # Pilot selection precedes evaluation on the distinct, fixed evaluation seed.
    # The 384 and 288 stages are retained because the supplement reports both.
    marginal_plan = REFRESH / "simulation_budget_plan.json"
    common_plan = COMMON / "simulation_plan.json"
    bud_plan = BUD_ONLY / "simulation_plan.json"
    REFRESH.mkdir(parents=True, exist_ok=True)
    COMMON.mkdir(parents=True, exist_ok=True)
    BUD_ONLY.mkdir(parents=True, exist_ok=True)
    script("simulation_plan_budget", "--out", marginal_plan)
    script("simulation_cell_budget", "--plan", common_plan, *backend_options)
    script("simulation_bud_only_budget", "--plan", bud_plan, *backend_options)
    budgets = [json.loads(path.read_bytes())["selected_m"] for path in (marginal_plan, common_plan, bud_plan)]
    if budgets != [384, 288, 242]:
        raise RuntimeError(f"Pilot budgets {budgets} differ from the paper. Check the numerical environment.")

    for m, raw in ((120, REFRESH / "confirmation"), (budgets[0], REFRESH / "confirmation_planned")):
        script("bud_experiment", "simulation", "--out", raw, "--m-group", m, *backend_options)
    for name, plan, out in (("simulation_cell_budget", common_plan, COMMON),
                            ("simulation_bud_only_budget", bud_plan, BUD_ONLY)):
        script(name, "--plan", plan, "--evaluate", out / "simulation_new", *backend_options)

    summarize(REFRESH / "confirmation", REFRESH / "paper_simulation", REFRESH / "figures", cohorts=50)
    summarize(REFRESH / "confirmation_planned", REFRESH / "paper_planned",
              REFRESH / "figures/planned", cohorts=50, grouped=True)
    for out, table in ((COMMON, "table_optimized.tex"), (BUD_ONLY, "table_bud_planned.tex")):
        paper = out / "simulation_paper"
        summarize(out / "simulation_new", paper, out / "figures/diagnostics", cohorts=50, grouped=True)
        (paper / "table_planned.tex").replace(paper / table)
    script("plot_simulation_budgets", "--planned-budget", budgets[2],
           "--planned-summary", BUD_ONLY / "simulation_paper",
           "--reference-table", REFRESH / "paper_simulation/table_reference_comparison.tex",
           "--out", BUD_ONLY / "figures/simulation_bud_budgets.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    options = parser.parse_args()
    run(options.backend, options.workers, options.quick)
