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

The default backend compiles the included C sources with GCC or Clang and caches the shared libraries in the system temporary directory. Set `BUD_CC` to a specific compiler executable if needed. The Python reference backend is available with `--backend python`. The S11 checks use their original NumPy/SciPy implementations with either option. The `doctor` command reports the environment and native kernel status. Run `python code/validate_km_backends.py --design-benchmark` for additional C and Python parity checks.

The smoke profile uses reduced replication counts and writes to `smoke_results/`. The main full simulation evaluates all 12 settings at 120 per group and the independently planned 242 per group using 1,000 independent complete cohorts per setting, one phase-II draw per cohort, and 1,000 bootstrap resamples per draw; its fixed evaluation seed is 33891027. Experimental stages run in separate processes to preserve their original design imports, random-number streams, and source provenance. Use a fresh checkout for a repeat run with a different backend. For a selectable workload, add `--target simulation`, `--target gbsg`, or `--target supplement`. The supplement creates a separately labelled archived 50-by-20 120-per-group reference run only when needed for S12. Each full target checks its available protocol and numerical targets against `reference_results.json`. To repeat the checks without rerunning the experiments, use `python reproduce.py verify`.

The anchor sensitivity stage generates 12,000 cohorts and can take substantially longer than the other supplement checks. It saves completed chunks as it runs. To resume that stage alone with the same backend, use `python code/refresh_secondary.py anchors --root results/budget_refresh --workers 8`, then run `python reproduce.py verify --target supplement`.

## Data and code

No application CSVs are shipped. The GBSG and Rotterdam datasets are downloaded from a pinned Rdatasets revision and checked against their SHA-256 digests. Download and check both separately with `python reproduce.py data`. The runner accepts cached files with the same digests in `data/`. GBSG supplies the application cohort. The node-positive subset of Rotterdam supplies the external pilot inputs for BUD's size rule. [GBSG source documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/gbsg.html) · [Rotterdam source documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/rotterdam.html)

- `reproduce.py` is the main entry point. The three `code/reproduce_*.py` drivers specify the stage order and output locations.
- `code/native/` contains the C kernels for weighted KM, good-lattice-point design operations, and paired sampling. The adjacent Python backend modules adapt these kernels to the original analysis.
- `code/legacy_locked/` contains the locked simulation generators, designs, and survival routines. The real-data runners retain their original `code/real_core/` import paths for source provenance.
- `reference_results.json` contains compact numerical targets from the archived manuscript runs. It is used for comparison after computation.

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

Update the affected analysis and documented protocol together. Keep published seeds and setups available under their release tag, and use new result directories when a protocol changes. Rebuild every affected table and figure from fresh inputs. The independent-cohort protocol checks are already listed in `reference_results.json`; after the full runs finish, add the new numerical checks for `paper_120` and `paper_242` summary, paired-evidence, ratio, headline, and TeX-table outputs, plus the two-budget figure. Do not alter the archived 50-by-20 canonical checkpoints. Keep README commands and the paper-to-output map synchronized. See [maintenance instructions](AGENTS.md).

Use GitHub's citation information or cite the accompanying manuscript when using these materials.
