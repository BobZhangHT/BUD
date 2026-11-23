"""Reproduce the historical and independent-cohort simulation analyses.

The full protocol retains the two historical budget outputs and evaluates the
primary independent cohorts at 120 and at the selected BUD-only budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_REFRESH = ROOT / "results/budget_refresh"
COMMON = ROOT / "results/budget_optimization"
INDEPENDENT = ROOT / "results/independent_cohorts"


def _run(script, *args):
    command = [sys.executable, "-u", str(ROOT / "code" / f"{script}.py"), *map(str, args)]
    print(">", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _analyze(raw, output, figures, *, cohorts, draws, grouped=False):
    arguments = ["analyze", "--confirmation", raw, "--out", output, "--figures", figures,
                 "--cohorts", cohorts, "--draws", draws]
    if grouped:
        arguments.append("--planned-table")
    _run("sim_results", *arguments)


def _plan(stage, output, backend, workers):
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        _run("sim_budget", stage, "--out", output, "--backend", backend, "--workers", workers)


def run(backend, workers, quick):
    options = ("--backend", backend, "--workers", workers)
    if quick:
        # The smoke target exercises the same locked runner and cohort-block analysis.
        output = ROOT / "smoke_results/simulation"
        raw = output / "confirmation"
        _run("bud_experiment", "simulation", "--out", raw, "--cohorts", 2, "--draws", 1,
             "--bootstrap-replicates", 17, *options)
        _analyze(raw, output / "tables", output / "figures", cohorts=2, draws=1)
        return

    marginal_plan = ARCHIVED_REFRESH / "simulation_budget_plan.json"
    common_plan = COMMON / "simulation_plan.json"
    bud_plan = INDEPENDENT / "simulation_plan.json"
    _plan("marginal-calibration", marginal_plan, backend, workers)
    _plan("common-cell-calibration", common_plan, backend, workers)
    _plan("bud-calibration", bud_plan, backend, workers)
    budgets = [json.loads(path.read_bytes())["selected_m"] for path in (marginal_plan, common_plan, bud_plan)]
    if budgets != [384, 288, 242]:
        raise RuntimeError(f"Pilot budgets {budgets} differ from the paper. Check the numerical environment.")

    _run("bud_experiment", "simulation", "--out", ARCHIVED_REFRESH / "confirmation_planned", "--cohorts", 50,
         "--draws", 20, "--bootstrap-replicates", 1000, "--m-group", budgets[0], "--seed", 33891027, *options)
    _run("sim_budget", "evaluate", "--plan", common_plan, "--out", COMMON / "simulation_new", *options)
    _analyze(ARCHIVED_REFRESH / "confirmation_planned", ARCHIVED_REFRESH / "paper_planned",
             ARCHIVED_REFRESH / "figures/planned", cohorts=50, draws=20, grouped=True)
    historical = COMMON / "simulation_paper"
    _analyze(COMMON / "simulation_new", historical, COMMON / "figures/diagnostics", cohorts=50, draws=20, grouped=True)
    (historical / "table_planned.tex").replace(historical / "table_optimized.tex")

    _run("bud_experiment", "simulation", "--out", INDEPENDENT / "confirmation_120", "--cohorts", 1000,
         "--draws", 1, "--bootstrap-replicates", 1000, "--m-group", 120, "--seed", 33891027, *options)
    _run("sim_budget", "evaluate", "--plan", bud_plan, "--out", INDEPENDENT / "confirmation_242",
         "--cohorts", 1000, "--draws", 1, "--bootstrap-replicates", 1000, *options)
    _analyze(INDEPENDENT / "confirmation_120", INDEPENDENT / "paper_120", INDEPENDENT / "figures", cohorts=1000, draws=1)
    _analyze(INDEPENDENT / "confirmation_242", INDEPENDENT / "paper_242", INDEPENDENT / "figures/planned", cohorts=1000, draws=1, grouped=True)
    _run("sim_results", "budget-plot", "--planned-budget", budgets[2], "--reference-summary", INDEPENDENT / "paper_120",
         "--planned-summary", INDEPENDENT / "paper_242", "--reference-table", INDEPENDENT / "paper_120/table_reference_comparison.tex",
         "--planned-table", INDEPENDENT / "paper_242/table_planned.tex", "--out", INDEPENDENT / "figures/simulation_bud_budgets.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args.backend, args.workers, args.quick)


if __name__ == "__main__":
    main()
