# Cell-based calibration of the phase-II budget

This evaluation reduces the common comparison budget while retaining the
existing precision and horizon tolerances. Outcome data from the evaluation
cohorts are excluded from planning. The locked algorithms and all existing
budget results are retained. For the simulations, near-equal cells extend the
equal-cell implementation to every feasible even budget. The first
\(N\bmod H\) cells receive \(\lceil N/H\rceil\) units, as in GBSG.
Each design samples two per cell, with exact capacity assignments and one
full-space BUD refinement.

## Planning rule

Use a 0.15 half-width for the two-group simulation contrast and a 90-day
half-width for GBSG, both with 95% normal-reference planning and a 1%
empty-horizon tolerance. Retain the original SRS score moment and projected
horizon count. Search the even budgets in increasing order, starting at the
first size at which SRS meets the criteria. This is a lower bound on the
common budget with those SRS inputs. At each candidate, use the original
horizon probability check and the exact stratified variance formula with a
working within-cell score variance for each of BUD and PC1-X. Select the
first candidate passing both checks for all three methods.

The simulation reuses the independent planning cohort protocol with 20
cohorts per scenario and 960 members per group. For each candidate and
method, compute each pilot cohort's exact linearized design variance on
its own partition and average the summed group variances across pilot
cohorts. Group horizon errors combine across independent groups. Report
the limiting planning case, the predecessor and every retained candidate.

For GBSG, estimate within-cell score variation on its fixed covariate frame
using an external Rotterdam working regression. Under independent
conditional score errors the expected within-cell variance is the variance
of the conditional means plus the average conditional residual variance.
Use common recorded predictors and assess prediction errors using Rotterdam
data held out from each fit. The projected moment is a working input
conditional on the target covariates. This calculation requires external
transportability and does not bound every realized bootstrap interval.

## Evaluation

For a new common simulation budget, reuse the existing evaluation seeds
and run every one of the 12 scenarios independently of the planning
cohorts. Use 50 cohorts, 20 phase-II draws per cohort, and 1,000 bootstrap
replicates per draw. For GBSG, evaluate the new common budget with 1,000
paired phase-II draws and 1,000 bootstrap replicates. Keep all old budgets,
scenarios, samples and outcomes in the archive. Report actual half-width,
design coverage, empty risk sets and Monte Carlo uncertainty. Keep GBSG
validation conditional on the released cohort and the fixed-grid theorem
separate from exact-time empirical interval performance.
