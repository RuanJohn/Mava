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

import logging
import subprocess
import textwrap
import time
from typing import Any, Callable

import pandas as pd

_env_scenario_registry = {
    "mabrax": [
        "humanoid_9|8",
        "ant_4x2",
        "halfcheetah_6x1",
        "hopper_3x1",
        "walker2d_2x3",
    ],
    "lbf": [
        "2s-8x8-2p-2f-coop",
        "8x8-2p-2f-coop",
        "2s-10x10-3p-3f",
        "10x10-3p-3f",
        "15x15-3p-5f",
        "15x15-4p-3f",
        "15x15-4p-5f",
    ],
    "smax": [
        "2s3z",
        "3s5z",
        "5m_vs_6m",
        "10m_vs_11m",
        "27m_vs_30m",
        "3s5z_vs_3s6z",
        "3s_vs_5z",
        "6h_vs_8z",
        "smacv2_5_units",
        "smacv2_10_units",
        "smacv2_20_units",
    ],
    "rware": [
        "tiny-2ag",
        "tiny-4ag",
        "small-4ag",
        "tiny-2ag-hard",
        "tiny-4ag-hard",
        "small-4ag-hard",
        "medium-6ag",
        "large-8ag",
        "large-8ag-hard",
        "medium-4ag",
        "medium-4ag-hard",
        "large-4ag",
        "large-4ag-hard",
        "xlarge-4ag",
        "xlarge-4ag-hard",
    ],
    "vector-connector": [
        "con-5x5x3a",
        "con-7x7x5a",
        "con-10x10x10a",
        "con-15x15x23a",
    ],
    "mpe": [
        "simple_spread_3ag",
        "simple_spread_5ag",
        "simple_spread_10ag",
    ],
}

systems_to_run = [
    # "ff_ppo_central",
    "rec_ppo_central",
    # "ff_ppo_central_factored",
    "rec_ppo_central_factrored",
]

system_scenarios_to_skip = {
    "rec_ppo_central_factrored": [
        "simple_spread_3ag",
        "simple_spread_5ag",
        "simple_spread_10ag",
        "ant_4x2",
        "halfcheetah_6x1",
        "hopper_3x1",
        "humanoid_9|8",
        "walker2d_2x3",
    ],
    "rec_ppo_central": [
        "con-15x15x23a",
        "con-10x10x10a",
        "3s5z",
        "3s5z_vs_3s6z",
        "smacv2_10_units",
        "smacv2_20_units",
        "6h_vs_8z",
        "10m_vs_11m",
        "27m_vs_30m",
    ],
}

_system_run_file_map = {
    "ff_ppo_central": "mava/systems/ppo/anakin/ff_ppo_central.py",
    "rec_ppo_central": "mava/systems/ppo/anakin/rec_ppo_central.py",
    "ff_ppo_central_factored": "mava/systems/ppo/anakin/ff_ppo_central_factored.py",
    "rec_ppo_central_factrored": "mava/systems/ppo/anakin/rec_ppo_central_factored.py",
}


def compute_should_run(system_name: str, scenario: str) -> bool:
    scenarios_to_skip = system_scenarios_to_skip.get(system_name)

    if system_name in systems_to_run and scenario not in scenarios_to_skip:
        return True
    else:
        return False


def get_script_contents(
    system_name: str,
    env: str,
    scenario: str,
) -> str:
    scenario_job_name = scenario

    if scenario == "humanoid_9|8":
        scenario = '"humanoid_9|8"'
        scenario_job_name = "humanoid"

    system_run_file = _system_run_file_map[system_name]

    job_name = f"sweep-{scenario_job_name}"

    if system_name.startswith("rec"):
        job_name = f"rec-{job_name}"

    job_name = f'"{job_name}"'

    base_script = textwrap.dedent(f"""\
    #!/bin/sh
    #SBATCH --account=l40sfree
    #SBATCH --partition=l40s
    #SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
    #SBATCH --time=48:00:00
    #SBATCH --job-name={job_name}
    #SBATCH --mail-user=dkcrua001@myuct.ac.za
    #SBATCH --mail-type=ALL

    module load python/miniconda3-py3.12

    source /home/dkcrua001/Mava/.venv/bin/activate

    cd Mava

    python {system_run_file} -m env={env} \\
    """)

    # Append the environment-specific scenario line directly
    if env == "smax":
        env_script = f"env.scenario.task_name={scenario}\n"
    elif env in ["rware", "lbf", "vector-connector", "mpe"]:
        env_script = f"env/scenario={scenario}\n"
    elif env == "mabrax":
        env_script = f"env.scenario.name={scenario} env.scenario.task_name={scenario}\n"

    script = base_script + env_script
    return script


def safe_cast(value: Any, type_func: Callable) -> Any:
    return type_func(value) if pd.notna(value) else value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tune_envs = [
        "mabrax",
        "lbf",
        "smax",
        "rware",
        "vector-connector",
        "mpe",
    ]

    for env in tune_envs:
        scenarios = _env_scenario_registry[env]

        for scenario in scenarios:
            for system in systems_to_run:
                should_run = compute_should_run(system, scenario)

                if should_run:
                    logging.info(f"Running experiment - {system} - {scenario}")

                    script_contents = get_script_contents(
                        system_name=system,
                        env=env,
                        scenario=scenario,
                    )
                    with open("run.sh", "w") as f:
                        f.write(script_contents)
                    try:
                        logging.info(f'Attempting to submit: "{system}" - "{scenario}"')
                        subprocess.run(["sbatch", "run.sh"], check=True)
                        time.sleep(3)
                    except Exception as e:
                        logging.error(f"Error submitting job: {e}")
