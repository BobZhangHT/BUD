"""Regenerate the independent-cohort main simulation and its planned BUD budget."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# Historical 50-by-20 results remain in their original directories for the
# archived common-budget comparisons and S12.  The current main simulation is
# deliberately separate: one phase-II draw from each independent cohort.
ARCHIVED_REFRESH = ROOT / "results/budget_refresh"
COMMON = ROOT / "results/budget_optimization"
INDEPENDENT = ROOT / "results/independent_cohorts"


def script(name, *args):
    command = [sys.executable, "-u", str(ROOT / "code" / f"{name}.py"), *map(str, args)]
    print(">", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summarize(raw, table_dir, figures, *, cohorts, draws, grouped=False):
    script("analyze_simulations", "--confirmation", raw, "--out", table_dir,
           "--figures", figures, "--cohorts", cohorts, "--draws", draws,
           *(["--planned-table"] if grouped else []))


def run(backend, workers, quick):
    backend_options = ("--backend", backend, "--workers", workers)
    if quick:
        # Exercise the same scenario generators, designs, analysis and figure paths.
        # A quick run uses one reference budget and cannot be verified as paper results.
        out = ROOT / "smoke_results/simulation"
        raw = out / "confirmation"
        script("bud_experiment", "simulation", "--out", raw, "--cohorts", 2, "--draws", 1,
               "--bootstrap-replicates", 17, *backend_options)
        summarize(raw, out / "tables", out / "figures", cohorts=2, draws=1)
        return

    # Pilot selection precedes the independent evaluation on fixed seed 33891027.
    # The archived 384/288 evaluations remain reported S9.7 tables and are
    # regenerated separately from the independent-cohort main analyses.
    marginal_plan = ARCHIVED_REFRESH / "simulation_budget_plan.json"
    common_plan = COMMON / "simulation_plan.json"
    bud_plan = INDEPENDENT / "simulation_plan.json"
    ARCHIVED_REFRESH.mkdir(parents=True, exist_ok=True)
    COMMON.mkdir(parents=True, exist_ok=True)
    INDEPENDENT.mkdir(parents=True, exist_ok=True)
    script("simulation_plan_budget", "--out", marginal_plan)
    script("simulation_cell_budget", "--plan", common_plan, *backend_options)
    script("simulation_bud_only_budget", "--plan", bud_plan, *backend_options)
    budgets = [json.loads(path.read_bytes())["selected_m"] for path in (marginal_plan, common_plan, bud_plan)]
    if budgets != [384, 288, 242]:
        raise RuntimeError(f"Pilot budgets {budgets} differ from the paper. Check the numerical environment.")

    script("bud_experiment", "simulation", "--out", ARCHIVED_REFRESH / "confirmation_planned",
           "--cohorts", 50, "--draws", 20, "--bootstrap-replicates", 1000,
           "--m-group", budgets[0], "--seed", 33891027, *backend_options)
    script("simulation_cell_budget", "--plan", common_plan, "--evaluate", COMMON / "simulation_new",
           *backend_options)
    summarize(ARCHIVED_REFRESH / "confirmation_planned", ARCHIVED_REFRESH / "paper_planned",
              ARCHIVED_REFRESH / "figures/planned", cohorts=50, draws=20, grouped=True)
    historical_paper = COMMON / "simulation_paper"
    summarize(COMMON / "simulation_new", historical_paper, COMMON / "figures/diagnostics",
              cohorts=50, draws=20, grouped=True)
    (historical_paper / "table_planned.tex").replace(historical_paper / "table_optimized.tex")

    script("bud_experiment", "simulation", "--out", INDEPENDENT / "confirmation_120",
           "--cohorts", 1000, "--draws", 1, "--bootstrap-replicates", 1000,
           "--m-group", 120, "--seed", 33891027, *backend_options)
    script("simulation_bud_only_budget", "--plan", bud_plan, "--evaluate",
           INDEPENDENT / "confirmation_242", "--cohorts", 1000, "--draws", 1,
           "--bootstrap-replicates", 1000, *backend_options)

    summarize(INDEPENDENT / "confirmation_120", INDEPENDENT / "paper_120",
              INDEPENDENT / "figures", cohorts=1000, draws=1)
    summarize(INDEPENDENT / "confirmation_242", INDEPENDENT / "paper_242",
              INDEPENDENT / "figures/planned", cohorts=1000, draws=1, grouped=True)
    script("plot_simulation_budgets", "--planned-budget", budgets[2],
           "--reference-summary", INDEPENDENT / "paper_120",
           "--planned-summary", INDEPENDENT / "paper_242",
           "--reference-table", INDEPENDENT / "paper_120/table_reference_comparison.tex",
           "--planned-table", INDEPENDENT / "paper_242/table_planned.tex",
           "--out", INDEPENDENT / "figures/simulation_bud_budgets.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    options = parser.parse_args()
    run(options.backend, options.workers, options.quick)
