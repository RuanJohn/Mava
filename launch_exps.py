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


def get_script_contents(
    system_name: str,
    system_seed: int,
    env_seed: int,
    task_name: str,
    num_agent: int,
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

    system_run_seed = int(env_seed * 10 + system_seed)

    script = f"""\
        #!/bin/bash
        # Script auto-generated by experiment_launch_exps.py on {timestamp}
        python {system_run_file} -m system.seed={system_run_seed} \\
        env.scenario.task_name={task_name} env.scenario.task_config.num_agents={num_agent} \\
        env.scenario.task_config.key_integer={env_seed} \\
        """

    return textwrap.dedent(script)


num_agents = [2, 3, 4, 5, 6, 7]
env_seeds = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
system_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
system_names = [
    "ff_ippo_tabular",
    "ff_ippo_tabular_split",
    "ff_ppo_central_tabular",
    "ff_ippo",
    "ff_mappo",
    "ff_ppo_central",
]


def should_run(system_name: str, task_name: str) -> bool:
    return True


def make_task_name(num_agents: int) -> str:
    return f"matrax-{num_agents}-ag-4-act"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for system_name in system_names:
        for num_agent in num_agents:
            for env_seed in env_seeds:
                for system_seed in system_seeds:
                    task_name = make_task_name(num_agent)

                    if should_run(system_name, task_name):
                        logging.info(f"Running experiment {system_name} - {task_name}")

                        script_contents = get_script_contents(
                            system_name=system_name,
                            env_seed=env_seed,
                            system_seed=system_seed,
                            task_name=task_name,
                            num_agent=num_agent,
                        )
                        with open("run.sh", "w") as f:
                            f.write(script_contents)
                        try:
                            subprocess.run(["./run.sh"], check=True)
                            logging.info("Experiment launched successfully")

                        except subprocess.CalledProcessError as e:
                            logging.error(f"Error launching the experiment: {e}")
