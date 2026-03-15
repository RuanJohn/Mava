# GPU Memory Benchmark Script Design

## Goal

Measure peak GPU memory usage for every algorithm x task combination in the thesis. Output results to a CSV file.

## Algorithm x Task Matrix

### Group 1: Climbing Game (Matrax)

**10 algorithms** on `Climbing-stateless-v0`:
- Tabular: ff_ippo_tabular_split, ff_ppo_central_tabular, ff_ppo_central_factored_tabular, ff_ppo_central_autoreg_tabular, ff_ppo_central_autoreg_chained_tabular
- Neural: ff_ippo, ff_mappo, ff_ppo_central, ff_ppo_central_factored, ff_sable

### Group 2: Array Games (GeneralMatrax)

**Same 10 algorithms** on all `(N, |A|)` combinations:
- N (num_agents) in {2, 3, 4, 5, 6, 7}
- |A| (num_actions) in {2, 3, 4, 5, 6, 7, 8}
- 42 tasks per algorithm, 420 total runs
- Use `generate_shadowed_payoffs=True`, `key_integer=42`

### Group 3: Modern MARL Benchmarks

**5 neural FF algorithms**: ff_ippo, ff_mappo, ff_ppo_central, ff_ppo_central_factored, ff_sable

**45 tasks across 6 environments:**

| Environment | Config name | Scenario override | Tasks |
|---|---|---|---|
| RWARE | rware | scenario=X | tiny-2ag, tiny-2ag-hard, tiny-4ag, tiny-4ag-hard, small-4ag, small-4ag-hard, medium-4ag, medium-4ag-hard, large-4ag, large-4ag-hard, xlarge-4ag, xlarge-4ag-hard, medium-6ag, large-8ag, large-8ag-hard |
| SMAX | smax | env.scenario.task_name=X | 2s3z, 3s5z, 3s_vs_5z, 6h_vs_8z, 5m_vs_6m, 10m_vs_11m, 3s5z_vs_3s6z, 27m_vs_30m, smacv2_5_units, smacv2_10_units, smacv2_20_units |
| LBF | lbf | scenario=X | 8x8-2p-2f-coop, 2s-8x8-2p-2f-coop, 10x10-3p-3f, 2s-10x10-3p-3f, 15x15-3p-5f, 15x15-4p-3f, 15x15-4p-5f |
| MaBrax | mabrax | env.scenario.name=X env.scenario.task_name=X | hopper_3x1, halfcheetah_6x1, walker2d_2x3, ant_4x2, humanoid_9x8 |
| Connector | connector | scenario=X | con-5x5x3a, con-7x7x5a, con-10x10x10a, con-15x15x23a |
| MPE | mpe | scenario=X | simple_spread_3ag, simple_spread_5ag, simple_spread_10ag |

## Architecture

### Two files:
1. **`benchmark_memory.py`** — orchestrator that loops over all combos, spawns subprocesses, collects results into CSV
2. **`_memory_runner.py`** — subprocess worker that imports a system module, runs training via Hydra compose API, prints peak GPU memory

### Why subprocesses
JAX GPU memory cannot be fully freed within a single process. Each run needs a fresh process for accurate measurement.

### Config overrides (all runs)
```
system.num_updates=8
arch.num_evaluation=4
arch.num_envs=16
system.update_batch_size=1
system.ppo_epochs=1
system.num_minibatches=2
logger.use_wandb=False
logger.use_neptune=False
logger.use_json=False
logger.use_tb=False
logger.use_console=False
```

### Memory measurement
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` env var
- After `run_experiment()` completes, query `jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]`
- Print `PEAK_MEM_BYTES:<value>` to stdout
- Parent process parses this from subprocess stdout

### OOM handling
- Subprocess crashes with non-zero exit → record "OOM" in CSV
- Script continues to next combination

### CSV output
File: `memory_benchmark_results.csv`
Columns: `algorithm,task,env,peak_memory_mb`
- Appended after each run for crash resilience
- "OOM" recorded for failed runs

### Algorithm to system file mapping
```
ff_ippo                              → mava/systems/ppo/anakin/ff_ippo.py
ff_mappo                             → mava/systems/ppo/anakin/ff_mappo.py
ff_ppo_central                       → mava/systems/ppo/anakin/ff_ppo_central.py
ff_ppo_central_factored              → mava/systems/ppo/anakin/ff_ppo_central_factored.py
ff_sable                             → mava/systems/sable/anakin/ff_sable.py
ff_ippo_tabular_split                → mava/systems/ppo/anakin/ff_ippo_tabular_split.py
ff_ppo_central_tabular               → mava/systems/ppo/anakin/ff_ppo_central_tabular.py
ff_ppo_central_factored_tabular      → mava/systems/ppo/anakin/ff_ppo_central_factored_tabular.py
ff_ppo_central_autoreg_tabular       → mava/systems/ppo/anakin/ff_ppo_central_autoreg_tabular.py
ff_ppo_central_autoreg_chained_tabular → mava/systems/ppo/anakin/ff_ppo_central_autoreg_chained_tabular.py
```
