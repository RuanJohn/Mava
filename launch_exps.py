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

seed_strings = [
    "0,1,2",
    "3,4,5",
    "6,7,8,9",
]


def get_script_contents(
    system_name: str,
    env_name: int,
    task_name: str,
    seed_string: str,
) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if system_name == "ff_ppo_central":
        system_run_file = "mava/systems/ppo/ff_ppo_central.py"
    elif system_name == "ff_ippo":
        system_run_file = "mava/systems/ppo/ff_ippo.py"
    elif system_name == "ff_mappo":
        system_run_file = "mava/systems/ppo/ff_mappo.py"
    elif system_name == "ff_ippo_tabular":
        system_run_file = "mava/systems/ppo/ff_ippo_tabular.py"
    elif system_name == "ff_ppo_central_tabular":
        system_run_file = "mava/systems/ppo/ff_ppo_central_tabular.py"
    elif system_name == "ff_ippo_tabular_split":
        system_run_file = "mava/systems/ppo/ff_ippo_tabular_split.py"

    script = f"""\
        #!/bin/bash
        # Script auto-generated by experiment_launch_exps.py on {timestamp}
        python {system_run_file} -m system.seed={seed_string} \\
        env={env_name} env.scenario.task_name={task_name}
        """

    return textwrap.dedent(script)


task_names = [
    "smacv2_5_units",
    "2s3z",
    "5m_vs_6m",
]
system_names = [
    "ff_ppo_central",
    "ff_ippo",
    "ff_mappo",
]


def should_run(system_name: str, task_name: str) -> bool:
    return True


def make_task_name(num_agents: int) -> str:
    return f"matrax-{num_agents}-ag-4-act"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for seed_string in seed_strings:
        for task_name in task_names:
            for system_name in system_names:
                if should_run(system_name, task_name):
                    logging.info(f"Running experiment {system_name} - {task_name}")

                    script_contents = get_script_contents(
                        system_name=system_name,
                        env_name="smax",
                        task_name=task_name,
                        seed_string=seed_string
                    )
                    with open("run.sh", "w") as f:
                        f.write(script_contents)
                    try:
                        subprocess.run(["./run.sh"], check=True)
                        logging.info("Experiment launched successfully")

                    except subprocess.CalledProcessError as e:
                        logging.error(f"Error launching the experiment: {e}")
