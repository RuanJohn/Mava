# optimal_parameters

CARBS-selected hyperparameters, one row per (`system_name`, `task`).

## matrix-games-carbs-sweep.csv — 59 rows

Two provenances, distinguishable by `run_id`:

| `run_id` | rows | sweep |
|---|---|---|
| `CEN-#####` | 56 | original TPU-v4 sweep, pulled from Neptune by `get_hyperparams/pull_carbs_params.py` |
| `carbs-climbing-*` | 3 | local GPU sweep, 2026-09-25, by `tune_climbing_missing.py` |

Same protocol either way: CARBS, 40 trials per algorithm, the 11-parameter space in
`mava/systems_tuning/tune_space.py`, objective = final evaluator return, cost =
`num_updates`. The CARBS version was unpinned in both installs, so the two sweeps are
not provably the same optimiser build.

The three local rows are `ff_ppo_central_factored_tabular`,
`ff_ppo_central_autoreg_tabular` and `ff_ppo_central_autoreg_chained_tabular`, which the
original sweep never produced. Per-trial records: `tuning_results/climbing-carbs-*-trials.csv`.

## Reading `episode_return` on Climbing

**The Climbing rows span two different environments.** The `CEN-*` rows predate the
matrax cooperative-climbing fix (commit `c2b92bb`); the `carbs-climbing-*` rows are
post-fix. This is visible in the column: pre-fix, policies without conditional
dependence settle on 7.0 (neural) or 6.0 (tabular); post-fix they settle on 5.0, the
shadowed equilibrium the analysis predicts.

`episode_return` is therefore the objective each sweep maximised, **not** a
cross-comparable result. For results use the rerun data
(`data/climbing-rerun-fixed-matrax/`), which is post-fix for all ten algorithms.
The tuned *hyperparameters* are unaffected — the fix changes one agent's view of the
payoff matrix, not the search space.

## Two columns to treat with care

- `wall_clock_time` — minutes. Local rows were recorded in seconds and converted.
- `time_series_total` — sum of the evaluation return series (e.g. `ff_ippo` on Climbing:
  799.78 over 122 evaluations = 6.56 mean against a 7.0 final). **Blank on the local
  rows:** that sweep logged only each trial's final return, not its curve, so there is
  no honest value to record.

## Consumers

`exp_launchers/launch_carbs_benchmark.py` and `launch_carbs_benchmark_manual_loop.py`
read this file with pandas and filter by `system_name`/`task`, so extra rows are inert
unless selected. `run_climbing_rerun.sh` does **not** read it — its `hparams()` values are
inlined, and are verified equal to this table.

> `get_hyperparams/pull_carbs_params.py` **overwrites** this file from Neptune. Neptune is
> discontinued, but if it is ever re-run the three local rows would be lost.
