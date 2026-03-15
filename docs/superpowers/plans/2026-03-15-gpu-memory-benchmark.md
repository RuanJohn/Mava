# GPU Memory Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure peak GPU memory usage for every algorithm x task combination in the thesis, outputting results to a CSV.

**Architecture:** Two Python scripts — `_memory_runner.py` (subprocess worker that imports a system module, runs training via Hydra compose API, prints peak GPU memory) and `benchmark_memory.py` (orchestrator that loops all combos, spawns subprocesses, collects results into CSV). Each combo runs in its own subprocess for clean GPU state. OOM crashes are caught and recorded.

**Tech Stack:** Python, JAX (`jax.local_devices()[0].memory_stats()`), Hydra compose API, subprocess

---

## Chunk 1: Memory Runner (subprocess worker)

### Task 1: Create `_memory_runner.py`

**Files:**
- Create: `_memory_runner.py` (project root)

This script is invoked as a subprocess. It:
1. Accepts CLI args: `--module` (Python module path like `mava.systems.ppo.anakin.ff_ippo`), `--config-name` (like `ff_ippo.yaml`), plus remaining args as Hydra overrides
2. Uses Hydra compose API to build the config
3. Dynamically imports the module and calls `run_experiment(cfg)`
4. Queries `jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]`
5. Prints `PEAK_MEM_BYTES:<value>` to stdout

- [ ] **Step 1: Write `_memory_runner.py`**

```python
# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Subprocess worker for GPU memory benchmarking.

Usage:
    python _memory_runner.py --module mava.systems.ppo.anakin.ff_ippo \
        --config-name ff_ippo.yaml \
        env=rware scenario=tiny-2ag system.num_updates=8 ...
"""

import argparse
import importlib
import os
import sys

import jax
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="Python module path, e.g. mava.systems.ppo.anakin.ff_ippo")
    parser.add_argument("--config-name", required=True, help="Hydra config name, e.g. ff_ippo.yaml")
    args, overrides = parser.parse_known_args()

    # Resolve absolute config dir path.
    config_dir = os.path.join(os.getcwd(), "mava", "configs", "default")

    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    # Allow dynamic attributes (matches hydra_entry_point behaviour).
    OmegaConf.set_struct(cfg, False)

    # Import the system module and call run_experiment.
    module = importlib.import_module(args.module)
    module.run_experiment(cfg)

    # Query peak GPU memory after training.
    device = jax.local_devices()[0]
    mem_stats = device.memory_stats()
    if mem_stats is not None:
        peak_bytes = mem_stats["peak_bytes_in_use"]
    else:
        # Fallback: memory_stats not available (CPU-only).
        peak_bytes = -1

    print(f"PEAK_MEM_BYTES:{peak_bytes}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the runner manually**

Run from project root:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python _memory_runner.py \
    --module mava.systems.ppo.anakin.ff_ippo \
    --config-name ff_ippo.yaml \
    env=rware scenario=tiny-2ag \
    system.num_updates=8 arch.num_evaluation=4 arch.num_envs=16 \
    system.update_batch_size=1 system.ppo_epochs=1 system.num_minibatches=2 \
    logger.use_wandb=False logger.use_neptune=False logger.use_json=False \
    logger.use_tb=False logger.use_console=False
```

Expected: training completes and last line of stdout contains `PEAK_MEM_BYTES:<some_number>`.

- [ ] **Step 3: Commit**

```bash
git add _memory_runner.py
git commit -m "feat: add memory runner subprocess worker for GPU benchmarking"
```

---

## Chunk 2: Benchmark Orchestrator

### Task 2: Create `benchmark_memory.py`

**Files:**
- Create: `benchmark_memory.py` (project root)

This script defines all algorithm x task combinations and orchestrates the runs.

**Key design decisions:**
- The algorithm-to-module/config mapping is a dict of dicts
- Tasks are grouped into 3 categories: climbing, array_games, modern_benchmarks
- Each combo is run via subprocess with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- Results appended to CSV after each run for crash resilience
- `PEAK_MEM_BYTES:` parsed from stdout; non-zero exit = "OOM"
- The script accepts optional `--group` flag to run only one group (climbing/array/modern) and `--resume` to skip already-completed rows in the CSV

**Algorithm → (module, config_name, default_config) mapping:**

For array games / climbing, these algorithms use their own default config (which defaults to `generalised-matrax` or needs env override for climbing):

| Algorithm | Module | Config name | Notes |
|---|---|---|---|
| ff_ippo | mava.systems.ppo.anakin.ff_ippo | ff_ippo.yaml | Default env is rware, override to matrax/generalised-matrax |
| ff_mappo | mava.systems.ppo.anakin.ff_mappo | ff_mappo.yaml | Same |
| ff_ppo_central | mava.systems.ppo.anakin.ff_ppo_central | ff_ppo_central.yaml | Default env is generalised-matrax |
| ff_ppo_central_factored | mava.systems.ppo.anakin.ff_ppo_central_factored | ff_ppo_central_factored.yaml | Same |
| ff_sable | mava.systems.sable.anakin.ff_sable | ff_sable.yaml | Default env is rware |
| ff_ippo_tabular_split | mava.systems.ppo.anakin.ff_ippo_tabular_split | ff_ippo_tabular_split.yaml | Default env is generalised-matrax |
| ff_ppo_central_tabular | mava.systems.ppo.anakin.ff_ppo_central_tabular | ff_ppo_central_tabular.yaml | Same |
| ff_ppo_central_factored_tabular | mava.systems.ppo.anakin.ff_ppo_central_factored_tabular | ff_ppo_central_factored_tabular.yaml | Same |
| ff_ppo_central_autoreg_tabular | mava.systems.ppo.anakin.ff_ppo_central_autoreg_tabular | ff_ppo_central_autoreg_tabular.yaml | Same |
| ff_ppo_central_autoreg_chained_tabular | mava.systems.ppo.anakin.ff_ppo_central_autoreg_chained_tabular | ff_ppo_central_autoreg_chained_tabular.yaml | Same |

**Modern benchmarks use only the first 5 (neural FF algorithms).**

**Environment → overrides mapping:**

For SMAX, scenarios are set via `env.scenario.task_name=X` (no separate scenario file).
For MaBrax, scenarios are set via `env.scenario.name=X env.scenario.task_name=X`.
For RWARE/LBF/Connector/MPE, scenarios are set via `scenario=X` (Hydra defaults).

- [ ] **Step 1: Write `benchmark_memory.py`**

```python
# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU memory benchmarking orchestrator.

Loops through all algorithm x task combinations, spawns each as a subprocess,
and records peak GPU memory to a CSV file.

Usage:
    python benchmark_memory.py [--group climbing|array|modern|all] [--resume] [--output results.csv]
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


# --- Algorithm definitions ---
# (module_path, config_name)
ALGORITHMS = {
    "ff_ippo": ("mava.systems.ppo.anakin.ff_ippo", "ff_ippo.yaml"),
    "ff_mappo": ("mava.systems.ppo.anakin.ff_mappo", "ff_mappo.yaml"),
    "ff_ppo_central": ("mava.systems.ppo.anakin.ff_ppo_central", "ff_ppo_central.yaml"),
    "ff_ppo_central_factored": (
        "mava.systems.ppo.anakin.ff_ppo_central_factored",
        "ff_ppo_central_factored.yaml",
    ),
    "ff_sable": ("mava.systems.sable.anakin.ff_sable", "ff_sable.yaml"),
    "ff_ippo_tabular_split": (
        "mava.systems.ppo.anakin.ff_ippo_tabular_split",
        "ff_ippo_tabular_split.yaml",
    ),
    "ff_ppo_central_tabular": (
        "mava.systems.ppo.anakin.ff_ppo_central_tabular",
        "ff_ppo_central_tabular.yaml",
    ),
    "ff_ppo_central_factored_tabular": (
        "mava.systems.ppo.anakin.ff_ppo_central_factored_tabular",
        "ff_ppo_central_factored_tabular.yaml",
    ),
    "ff_ppo_central_autoreg_tabular": (
        "mava.systems.ppo.anakin.ff_ppo_central_autoreg_tabular",
        "ff_ppo_central_autoreg_tabular.yaml",
    ),
    "ff_ppo_central_autoreg_chained_tabular": (
        "mava.systems.ppo.anakin.ff_ppo_central_autoreg_chained_tabular",
        "ff_ppo_central_autoreg_chained_tabular.yaml",
    ),
}

# Only neural FF algorithms for modern benchmarks.
NEURAL_ALGORITHMS = [
    "ff_ippo",
    "ff_mappo",
    "ff_ppo_central",
    "ff_ppo_central_factored",
    "ff_sable",
]

# All 10 algorithms for array games and climbing.
ALL_ALGORITHMS = list(ALGORITHMS.keys())

# --- Shared config overrides for all runs ---
SHARED_OVERRIDES = [
    "system.num_updates=8",
    "arch.num_evaluation=4",
    "arch.num_envs=16",
    "system.update_batch_size=1",
    "system.ppo_epochs=1",
    "system.num_minibatches=2",
    "logger.use_wandb=False",
    "logger.use_neptune=False",
    "logger.use_json=False",
    "logger.use_tb=False",
    "logger.use_console=False",
    "logger.checkpointing.save_model=False",
]

# --- Task definitions ---

# Climbing game (Matrax environment).
CLIMBING_TASKS = [
    {
        "task": "Climbing-stateless-v0",
        "env": "Matrax",
        "overrides": [
            "env=matrax",
            "env.scenario.task_name=Climbing-stateless-v0",
        ],
    }
]

# Array games (GeneralMatrax environment).
# (N, |A|) in {2..7} x {2..8}
def _make_array_game_tasks() -> List[Dict]:
    tasks = []
    for n_agents in range(2, 8):
        for n_actions in range(2, 9):
            task_name = f"matrax-{n_agents}-ag-{n_actions}-act"
            tasks.append(
                {
                    "task": task_name,
                    "env": "GeneralMatrax",
                    "overrides": [
                        "env=generalised-matrax",
                        f"env.scenario.task_name={task_name}",
                        f"env.scenario.task_config.num_agents={n_agents}",
                        f"env.scenario.task_config.num_actions={n_actions}",
                        "env.scenario.task_config.key_integer=42",
                        "env.kwargs.generate_shadowed_payoffs=True",
                    ],
                }
            )
    return tasks


ARRAY_GAME_TASKS = _make_array_game_tasks()

# Modern MARL benchmark tasks.
MODERN_TASKS = []

# RWARE (15 tasks) — use scenario=X override.
_RWARE_SCENARIOS = [
    "tiny-2ag", "tiny-2ag-hard", "tiny-4ag", "tiny-4ag-hard",
    "small-4ag", "small-4ag-hard", "medium-4ag", "medium-4ag-hard",
    "large-4ag", "large-4ag-hard", "xlarge-4ag", "xlarge-4ag-hard",
    "medium-6ag", "large-8ag", "large-8ag-hard",
]
for s in _RWARE_SCENARIOS:
    MODERN_TASKS.append(
        {"task": s, "env": "RobotWarehouse", "overrides": ["env=rware", f"env/scenario={s}"]}
    )

# SMAX (11 tasks) — use env.scenario.task_name=X override.
_SMAX_SCENARIOS = [
    "2s3z", "3s5z", "3s_vs_5z", "6h_vs_8z", "5m_vs_6m",
    "10m_vs_11m", "3s5z_vs_3s6z", "27m_vs_30m",
    "smacv2_5_units", "smacv2_10_units", "smacv2_20_units",
]
for s in _SMAX_SCENARIOS:
    MODERN_TASKS.append(
        {"task": s, "env": "Smax", "overrides": ["env=smax", f"env.scenario.task_name={s}"]}
    )

# LBF (7 tasks) — use scenario=X override.
_LBF_SCENARIOS = [
    "8x8-2p-2f-coop", "2s-8x8-2p-2f-coop", "10x10-3p-3f",
    "2s-10x10-3p-3f", "15x15-3p-5f", "15x15-4p-3f", "15x15-4p-5f",
]
for s in _LBF_SCENARIOS:
    MODERN_TASKS.append(
        {"task": s, "env": "LevelBasedForaging", "overrides": ["env=lbf", f"env/scenario={s}"]}
    )

# MaBrax (5 tasks) — use env.scenario.name=X env.scenario.task_name=X.
_MABRAX_SCENARIOS = [
    "hopper_3x1", "halfcheetah_6x1", "walker2d_2x3", "ant_4x2", "humanoid_9x8",
]
for s in _MABRAX_SCENARIOS:
    MODERN_TASKS.append(
        {
            "task": s,
            "env": "MaBrax",
            "overrides": ["env=mabrax", f"env.scenario.name={s}", f"env.scenario.task_name={s}"],
        }
    )

# Connector (4 tasks) — use scenario=X override.
_CONNECTOR_SCENARIOS = ["con-5x5x3a", "con-7x7x5a", "con-10x10x10a", "con-15x15x23a"]
for s in _CONNECTOR_SCENARIOS:
    MODERN_TASKS.append(
        {
            "task": s,
            "env": "MaConnector",
            "overrides": ["env=connector", f"env/scenario={s}"],
        }
    )

# MPE (3 tasks) — use scenario=X override.
_MPE_SCENARIOS = ["simple_spread_3ag", "simple_spread_5ag", "simple_spread_10ag"]
for s in _MPE_SCENARIOS:
    MODERN_TASKS.append(
        {
            "task": s,
            "env": "MPE",
            "overrides": ["env=mpe", f"env/scenario={s}"],
        }
    )


def load_completed(csv_path: str) -> set:
    """Load already-completed (algorithm, task) pairs from existing CSV."""
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["algorithm"], row["task"]))
    return completed


def run_single(
    algorithm: str,
    module_path: str,
    config_name: str,
    task_overrides: List[str],
) -> Optional[float]:
    """Run a single algorithm x task combo in a subprocess.

    Returns peak memory in MB, or None if OOM/crash.
    """
    cmd = [
        sys.executable,
        "_memory_runner.py",
        "--module", module_path,
        "--config-name", config_name,
    ] + SHARED_OVERRIDES + task_overrides

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per run.
            env=env,
        )

        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            # Print last few lines of stderr for debugging.
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-5:]:
                print(f"    {line}")
            return None

        # Parse peak memory from stdout.
        for line in result.stdout.split("\n"):
            if line.startswith("PEAK_MEM_BYTES:"):
                peak_bytes = int(line.split(":")[1])
                return peak_bytes / (1024 * 1024)  # Convert to MB.

        print("  WARNING: PEAK_MEM_BYTES not found in output")
        return None

    except subprocess.TimeoutExpired:
        print("  TIMEOUT (>600s)")
        return None


def write_row(csv_path: str, row: Dict, write_header: bool) -> None:
    """Append a single row to the CSV."""
    fieldnames = ["algorithm", "task", "env", "peak_memory_mb"]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_group(
    algorithms: List[str],
    tasks: List[Dict],
    csv_path: str,
    completed: set,
    write_header: bool,
) -> bool:
    """Run all algorithm x task combos for a group. Returns updated write_header."""
    for task_info in tasks:
        for algo_name in algorithms:
            key = (algo_name, task_info["task"])
            if key in completed:
                print(f"SKIP {algo_name} x {task_info['task']} (already done)")
                continue

            module_path, config_name = ALGORITHMS[algo_name]
            print(f"RUN  {algo_name} x {task_info['task']} ({task_info['env']})")

            peak_mb = run_single(algo_name, module_path, config_name, task_info["overrides"])

            row = {
                "algorithm": algo_name,
                "task": task_info["task"],
                "env": task_info["env"],
                "peak_memory_mb": f"{peak_mb:.1f}" if peak_mb is not None else "OOM",
            }
            write_row(csv_path, row, write_header)
            write_header = False

            if peak_mb is not None:
                print(f"  OK: {peak_mb:.1f} MB")

    return write_header


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU memory benchmarking orchestrator.")
    parser.add_argument(
        "--group",
        choices=["climbing", "array", "modern", "all"],
        default="all",
        help="Which task group to benchmark.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (algorithm, task) pairs already in the CSV.",
    )
    parser.add_argument(
        "--output",
        default="memory_benchmark_results.csv",
        help="Output CSV file path.",
    )
    args = parser.parse_args()

    completed = load_completed(args.output) if args.resume else set()
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0

    if args.group in ("climbing", "all"):
        print("\n=== CLIMBING ===")
        write_header = run_group(ALL_ALGORITHMS, CLIMBING_TASKS, args.output, completed, write_header)

    if args.group in ("array", "all"):
        print("\n=== ARRAY GAMES ===")
        write_header = run_group(ALL_ALGORITHMS, ARRAY_GAME_TASKS, args.output, completed, write_header)

    if args.group in ("modern", "all"):
        print("\n=== MODERN BENCHMARKS ===")
        write_header = run_group(NEURAL_ALGORITHMS, MODERN_TASKS, args.output, completed, write_header)

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add benchmark_memory.py
git commit -m "feat: add GPU memory benchmark orchestrator"
```

---

## Chunk 3: Verification

### Task 3: End-to-end smoke test

- [ ] **Step 1: Test a single climbing run**

```bash
python benchmark_memory.py --group climbing --output test_results.csv
```

Verify: `test_results.csv` has header + 10 rows (one per algorithm), with numeric values or "OOM".

- [ ] **Step 2: Test a single array game run**

```bash
python benchmark_memory.py --group array --output test_results_array.csv
```

Let a few combos run, then Ctrl+C. Verify CSV has rows with data.

- [ ] **Step 3: Test resume functionality**

Re-run the same command. Verify it skips already-completed rows:
```bash
python benchmark_memory.py --group array --output test_results_array.csv --resume
```

- [ ] **Step 4: Clean up test files and commit**

```bash
rm -f test_results.csv test_results_array.csv
git add _memory_runner.py benchmark_memory.py
git commit -m "feat: complete GPU memory benchmarking scripts"
```

---

## Running the full benchmark

Not part of the implementation — the user will run this manually:

```bash
# All groups:
python benchmark_memory.py --group all

# Or one at a time:
python benchmark_memory.py --group climbing
python benchmark_memory.py --group array --resume
python benchmark_memory.py --group modern --resume
```

Total runs: 10 (climbing) + 420 (array games) + 225 (modern) = **655 runs**.
At ~1-2 min each (mostly JAX compilation), expect ~10-20 hours for the full sweep.
