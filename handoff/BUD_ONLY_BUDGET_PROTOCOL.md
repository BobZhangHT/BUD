# BUD-specific phase-II planning

This revision chooses the budget using BUD alone. PC1-X and SRS are
evaluated at the chosen size and do not enter its selection. Keep the
original target half-widths of 0.15 for the two-group simulation contrast
and 90 days for GBSG. Retain nominal 95% normal-reference precision and a
1% projected empty-horizon tolerance. Each group in the simulation uses
the same even phase-II size. Capacities are near-equal with at least three
units per cell and two sampled per cell. The method, estimand, and interval
remain the locked BUD procedure.

The fixed-precision objective follows [Lavallee and Hidiroglou
(1988)](https://www150.statcan.gc.ca/n1/pub/12-001-x/1988001/article/14602-eng.pdf).
The external score-moment approach follows the anticipated-moment
principle of [Rivest (2002)](https://www150.statcan.gc.ca/n1/pub/12-001-x/2002002/article/6432-eng.pdf)
and [Baillargeon and Rivest (2011)](https://www.statcan.gc.ca/pub/12-001-x/2011001/article/11447-eng.pdf).
The BUD construction fixes the partition given the covariates and
candidate budget. The scan minimizes the sample size over these
partitions.

## Simulation search

Use the previously independent planning cohorts, 20 per scenario, with
seed 709153 and 960 per group. Enumerate each admissible even size from
4 through the previous feasible size of 288. Apply the existing
projected horizon check across the two groups and all scenarios first.
Reject any candidate with an excessive empty-horizon projection. For
efficiency, screen the remaining candidates using the nuisance-variable
scenario, which had the largest BUD planning half-width at 288. A failed
scenario rejects a size because the single reported budget must pass
all 12. At each surviving budget in ascending order, calculate the exact
pilot-cohort linearized variances for BUD in all other scenarios.
Average the sum of the group variances over the 20 separate planning
cohorts and multiply its square root by the normal critical value.
Stop at the first candidate passing every scenario. This procedure
checks all smaller candidates without assuming a monotone variance
curve. Record the selection and every rejection. Also record the
delta-method Monte Carlo standard error for the binding pilot projection.

## GBSG search

Use the same external Rotterdam scores, covariate harmonization,
out-of-fold residual second moment, and working risk count as in the
earlier calibration. For each feasible budget, the out-of-fold residual
moment times the sum of finite-population variance coefficients is a
lower bound on BUD's working variance. Use it to screen budgets before
partition construction and use the actual target covariate-only
partitions for the remaining candidates. Freeze the smallest passing
budget and all three partitions before opening the evaluation outcomes.

## Evaluation

Reuse the evaluation cohort and priority seeds from the existing
comparisons. For the new simulation budget, run 50 cohorts and 20 draws
per cohort in each of the 12 scenarios, and 1,000 bootstrap replicates
per draw. For GBSG, use 1,000 independent paired draws and 1,000 bootstrap
replicates. Keep all archived sizes. Report half-width, coverage, and
horizon support for every method, including comparisons above the
planning target. Planning inputs are independent of the simulation
evaluation cohorts. GBSG is a retrospective evaluation in the previously
studied released cohort, with the budget calculation based on its
phase-I covariates and Rotterdam outcomes.
