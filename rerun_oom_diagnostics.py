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

"""Rerun memory benchmark rows marked OOM and classify True OOM vs other failures."""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import benchmark_memory as bm


def _python_for_benchmarks() -> str:
    """Prefer project venv (JAX) over bare sys.executable."""
    if os.environ.get("MAVA_BENCHMARK_PYTHON"):
        return os.environ["MAVA_BENCHMARK_PYTHON"]
    repo = Path(__file__).resolve().parent
    venv_py = repo / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable

OUTPUT_CSV = "memory_benchmark_oom_rerun_results.csv"


def _build_task_overrides_lookup() -> Dict[Tuple[str, str], List[str]]:
    lookup: Dict[Tuple[str, str], List[str]] = {}
    for group in (bm.CLIMBING_TASKS, bm.ARRAY_GAME_TASKS, bm.MODERN_TASKS):
        for t in group:
            key = (t["task"], t["env"])
            if key in lookup and lookup[key] != t["overrides"]:
                raise ValueError(f"Duplicate task/env with different overrides: {key}")
            lookup[key] = t["overrides"]
    return lookup


def _is_true_oom(combined: str) -> bool:
    """Heuristic: JAX/XLA/CUDA memory exhaustion vs other errors.

    Also treats XLA kernel grid limits (needs more blocks than hardware) as True OOM:
    same practical outcome — the workload does not fit the device.
    """
    s = combined.lower()
    if "resource_exhausted" in s:
        return True
    # Kernel launch exceeds GPU block/grid limits (often from huge broadcasts).
    if "needs more blocks" in s and "allowed by hardware" in s:
        return True
    if "out of memory" in s and ("xla" in s or "cuda" in s or "gpu" in s or "jax" in s):
        return True
    if "cuda error: out of memory" in s or ("cudnn" in s and "alloc" in s):
        return True
    if re.search(r"could not allocate \d+", s) and ("gpu" in s or "device" in s or "cuda" in s):
        return True
    # PyTorch-style (unlikely here but harmless)
    if "cuda out of memory" in s:
        return True
    return False


def _short_reason(exit_code: int, stdout: str, stderr: str, success: bool) -> str:
    if success:
        return "Success on rerun (previous failure was not True OOM or was transient)"
    text = (stderr or "") + "\n" + (stdout or "")
    if _is_true_oom(text):
        return "True OOM"
    # Prefer last meaningful error line from stderr
    err_lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in reversed(err_lines[-12:]):
        if "Error" in ln or "Exception" in ln or "error:" in ln.lower():
            # Single line, CSV-safe-ish (replace newlines)
            cleaned = ln.replace("\n", " ")[:500]
            return cleaned
    tail = (stderr or stdout or "").strip().replace("\n", " ")[:500]
    return tail if tail else f"Non-zero exit {exit_code} (no stderr)"


def run_single_capture(
    algorithm: str,
    module_path: str,
    config_name: str,
    task_overrides: List[str],
) -> Tuple[Optional[float], int, str, str]:
    cmd = [
        _python_for_benchmarks(),
        "_memory_runner.py",
        "--module",
        module_path,
        "--config-name",
        config_name,
        *bm.SHARED_OVERRIDES,
        *task_overrides,
    ]
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Avoid protobuf 4+ / jax stack "Descriptors cannot be created directly" on import.
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    peak_mb: Optional[float] = None
    if result.returncode == 0:
        for line in result.stdout.split("\n"):
            if line.startswith("PEAK_MEM_BYTES:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    peak_bytes = int(parts[1])
                    peak_mb = peak_bytes / (1024 * 1024)
                break
    return peak_mb, result.returncode, result.stdout, result.stderr


def load_oom_rows(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("peak_memory_mb") == "OOM":
                rows.append(row)
    return rows


def main() -> None:
    lookup = _build_task_overrides_lookup()
    oom_rows = load_oom_rows("memory_benchmark_results.csv")
    fieldnames = [
        "algorithm",
        "task",
        "env",
        "oom_reason",
        "peak_memory_mb_rerun",
        "exit_code",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(oom_rows):
            algo = row["algorithm"]
            task = row["task"]
            env_name = row["env"]
            key = (task, env_name)
            if key not in lookup:
                writer.writerow(
                    {
                        "algorithm": algo,
                        "task": task,
                        "env": env_name,
                        "oom_reason": f"Unknown task/env (not in benchmark_memory): {key}",
                        "peak_memory_mb_rerun": "",
                        "exit_code": "",
                    }
                )
                continue
            module_path, config_name = bm.ALGORITHMS[algo]
            overrides = lookup[key]
            print(f"[{i + 1}/{len(oom_rows)}] {algo} x {task} ...", flush=True)
            peak_mb, code, out, err = run_single_capture(algo, module_path, config_name, overrides)
            ok = peak_mb is not None and code == 0
            reason = _short_reason(code, out, err, ok)
            writer.writerow(
                {
                    "algorithm": algo,
                    "task": task,
                    "env": env_name,
                    "oom_reason": reason,
                    "peak_memory_mb_rerun": f"{peak_mb:.1f}" if peak_mb is not None else "",
                    "exit_code": str(code),
                }
            )
            f.flush()
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
