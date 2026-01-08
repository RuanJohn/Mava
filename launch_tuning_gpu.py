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

systems_to_run = [
    "ff_ppo_central",
    # "rec_ppo_central",
]

env_to_scenario_map = {
    "mabrax": [
        "humanoid_9|8",
        "ant_4x2",
        "halfcheetah_6x1",
        "hopper_3x1",
        "walker2d_2x3",
    ],
    "mpe": [
        "simple_spread_3ag",
        "simple_spread_5ag",
        "simple_spread_10ag",
    ],
}

envs_to_run = ["mabrax", "mpe"]
scenarios_to_run = [
    "humanoid_9|8",
    "ant_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "walker2d_2x3",
    "simple_spread_3ag",
    "simple_spread_5ag",
    "simple_spread_10ag",
]

_system_run_file_map = {
    "ff_ppo_central": "mava/systems/ppo/anakin/ff_ppo_central.py",
    "rec_ppo_central": "mava/systems/ppo/anakin/rec_ppo_central.py",
}


def compute_should_run(env: str, scenario: str) -> bool:
    if env in envs_to_run and scenario in scenarios_to_run:
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

    job_name = f"ff-cont-sweep-{scenario_job_name}"

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

    source /home/dkcrua001/Mava/.venv/bin/activate

    cd Mava

    python {system_run_file} -m \\
    env={env} \\
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for env in envs_to_run:
        scenarios = env_to_scenario_map[env]
        for scenario in scenarios:
            should_run = compute_should_run(env, scenario)

            if should_run:
                logging.info(f"Running experiment - {env} - {scenario}")

                script_contents = get_script_contents(
                    system_name="ff_ppo_central",
                    env=env,
                    scenario=scenario,
                )


                with open("run.sh", "w") as f:
                    f.write(script_contents)
                try:
                    logging.info(f'Attempting to submit: "ff_ppo_central" - "{scenario}"')
                    subprocess.run(["sbatch", "run.sh"], check=True)
                    time.sleep(2)
                except Exception as e:
                    logging.error(f"Error submitting job: {e}")
