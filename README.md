# Balanced uniform-design subsampling for restricted mean survival time

Reproduction code for *Balanced uniform-design subsampling and bootstrap inference for restricted mean survival time*, by Hengtao Zhang, Yuanke Qu, and Xiaoya Xu. It regenerates the numerical results and figures for the main paper and its supplement.

BUD constructs strata using phase-I covariates, then samples two records independently without replacement within each stratum. The paper compares BUD with simple random sampling (SRS) and stratification on the first principal component of phase-I covariates (PC1-X). The target is the complete-cohort Kaplan–Meier restricted mean survival time (KM–RMST) or the difference between two complete-cohort KM–RMSTs.

## Run the experiments

Use **64-bit CPython 3.12** and install the pinned Python dependencies in a fresh environment.

```sh
python -m venv .venv
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. On macOS or Linux, use `source .venv/bin/activate`. Install the dependencies, check the environment, and launch the experiments.

```sh
python -m pip install -r requirements.txt
python reproduce.py doctor
python reproduce.py smoke
python reproduce.py run --workers 8
```

The default backend compiles the included C sources with GCC or Clang and caches the shared libraries in the system temporary directory. Set `BUD_CC` to a specific compiler executable if needed. The Python reference backend is available with `--backend python`. The S11 checks use their original NumPy/SciPy implementations with either option. The `doctor` command reports the environment and native kernel status. Run `python code/checks.py parity --design-benchmark` to compare the two backends.

The smoke profile uses reduced replication counts and writes to `smoke_results/`. The full main simulation evaluates 12 settings at 120 and 242 records per group. Each setting uses 1,000 independent complete cohorts, one phase-II draw per cohort, and 1,000 bootstrap replicates per draw. The independent planning stage selects 242, and the evaluation uses seed 33891027. Experimental stages run in separate processes to preserve their design imports and random-number streams. Use a fresh checkout to repeat a full run with a different backend.

Use `--target simulation`, `--target gbsg`, or `--target supplement` to run a single target. The supplement generates the earlier 50-cohort, 20-draw reference run when needed for S12. Each full target checks its protocol and numerical outputs against `reference_results.json`. To verify existing outputs, use `python reproduce.py verify`.

The anchor sensitivity study generates 12,000 cohorts and saves completed batches. To resume it with the same backend, run `python code/supplement.py anchors --root results/budget_refresh --workers 8`, followed by `python reproduce.py verify --target supplement`.

## Data and code

No application CSVs are shipped. The GBSG and Rotterdam datasets are downloaded from a pinned Rdatasets revision and checked against their SHA-256 digests. Download and check both separately with `python reproduce.py data`. The runner accepts cached files with the same digests in `data/`. GBSG supplies the application cohort. The node-positive subset of Rotterdam supplies the external pilot inputs for BUD's size rule. [GBSG source documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/gbsg.html) · [Rotterdam source documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/rotterdam.html)

`reproduce.py` runs the three targets and verifies their outputs against `reference_results.json`.

| Where to look | What it does |
| --- | --- |
| `code/simulation.py` | Runs the main simulations. `sim_budget.py` selects review budgets, and `sim_results.py` builds the summaries, tables, and figure. |
| `code/gbsg.py` | Downloads the pinned datasets and runs the GBSG study. `gbsg_plan.py` handles planning, and `gbsg_eval.py` evaluates the sampling designs. |
| `code/supplement.py` | Runs the theory and sensitivity studies and resumes the anchor study. |
| `code/backend.py` and `code/native/` | Provide the C kernels and NumPy reference paths for weighted KM, lattice design operations, and paired sampling. |
| `code/checks.py` | Checks the environment, verifies the paper results, and compares the numerical backends. |
| `code/legacy_locked/` and `code/real_core/` | Preserve the executed simulation and real-data routines used by the runners. |

## Outputs

| Paper material | Regenerated results |
| --- | --- |
| Main simulation | `results/independent_cohorts/` contains the 1,000-cohort, one-draw-per-cohort analyses at 120 and 242 per group, their summaries, and `figures/simulation_bud_budgets.pdf`. |
| Earlier common-budget comparisons | The simulation target regenerates and verifies the S9.7 tables at 384 and 288 per group using 50 cohorts and 20 draws per cohort. The respective summaries are in `results/budget_refresh/paper_planned/` and `results/budget_optimization/simulation_paper/`. |
| GBSG and Supplementary Section S10 | Read `summary.csv` and `paired.csv` under `results/budget_refresh/small_budgets/` (40, 80), `results/budget_refresh/medium_gbsg/` (120, 240), `results/bud_budget/gbsg/evaluation/` (136), and `results/bud_budget/gbsg_grid_160_200/` (160, 200). The 136-patient planning record and frozen partitions are in the parent `gbsg/` directory. |
| Supplementary Section S11 | `results/theory_validation/` contains the geometry, fixed-grid bootstrap, and sample-size checks. The three PDFs in `figures/` use the settings shown in the supplement. |
| Supplementary Section S12 | `results/budget_refresh/` contains the budget-sensitivity, fixed-grid Weibull, and anchor-sensitivity summaries. Its required historical 120-per-group raw input is isolated at `results/archived_reference/confirmation_120/`. |

The verifier checks selected numerical cells, interval endpoints, scenario sizes, and the source of reported plots. It uses relative tolerance `2e-9` and absolute tolerance `1e-11` for floating-point comparisons. Generated TeX tables are checked after normalizing line endings. Figure PDFs are checked as rendered files since their binary metadata can differ across builds. Raw draws, plots, local data, and compiled libraries are excluded from version control.

## Update with a manuscript revision

Update the analysis and protocol together. Keep prior seeds, settings, and result checks available in their published revision. Use new output directories when the protocol changes, then rebuild the affected tables and figures. Refresh `reference_results.json` from the verified outputs and keep the README commands and output map in sync. [Maintenance instructions](AGENTS.md)

Use GitHub's citation information or cite the accompanying manuscript when using these materials.
