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
import subprocess
import sys
from typing import Dict, List, Optional

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
    "tiny-2ag",
    "tiny-2ag-hard",
    "tiny-4ag",
    "tiny-4ag-hard",
    "small-4ag",
    "small-4ag-hard",
    "medium-4ag",
    "medium-4ag-hard",
    "large-4ag",
    "large-4ag-hard",
    "xlarge-4ag",
    "xlarge-4ag-hard",
    "medium-6ag",
    "large-8ag",
    "large-8ag-hard",
]
for s in _RWARE_SCENARIOS:
    MODERN_TASKS.append(
        {"task": s, "env": "RobotWarehouse", "overrides": ["env=rware", f"env/scenario={s}"]}
    )

# SMAX (11 tasks) — use env.scenario.task_name=X override.
_SMAX_SCENARIOS = [
    "2s3z",
    "3s5z",
    "3s_vs_5z",
    "6h_vs_8z",
    "5m_vs_6m",
    "10m_vs_11m",
    "3s5z_vs_3s6z",
    "27m_vs_30m",
    "smacv2_5_units",
    "smacv2_10_units",
    "smacv2_20_units",
]
for s in _SMAX_SCENARIOS:
    MODERN_TASKS.append(
        {"task": s, "env": "Smax", "overrides": ["env=smax", f"env.scenario.task_name={s}"]}
    )

# LBF (7 tasks) — use scenario=X override.
_LBF_SCENARIOS = [
    "8x8-2p-2f-coop",
    "2s-8x8-2p-2f-coop",
    "10x10-3p-3f",
    "2s-10x10-3p-3f",
    "15x15-3p-5f",
    "15x15-4p-3f",
    "15x15-4p-5f",
]
for s in _LBF_SCENARIOS:
    MODERN_TASKS.append(
        {"task": s, "env": "LevelBasedForaging", "overrides": ["env=lbf", f"env/scenario={s}"]}
    )

# MaBrax (5 tasks) — use env.scenario.name=X env.scenario.task_name=X.
_MABRAX_SCENARIOS = [
    "hopper_3x1",
    "halfcheetah_6x1",
    "walker2d_2x3",
    "ant_4x2",
    "humanoid_9x8",
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
    cmd = (
        [
            sys.executable,
            "_memory_runner.py",
            "--module",
            module_path,
            "--config-name",
            config_name,
        ]
        + SHARED_OVERRIDES
        + task_overrides
    )

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
        write_header = run_group(
            ALL_ALGORITHMS, CLIMBING_TASKS, args.output, completed, write_header
        )

    if args.group in ("array", "all"):
        print("\n=== ARRAY GAMES ===")
        write_header = run_group(
            ALL_ALGORITHMS, ARRAY_GAME_TASKS, args.output, completed, write_header
        )

    if args.group in ("modern", "all"):
        print("\n=== MODERN BENCHMARKS ===")
        write_header = run_group(
            NEURAL_ALGORITHMS, MODERN_TASKS, args.output, completed, write_header
        )

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
