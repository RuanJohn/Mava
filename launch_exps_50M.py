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
from typing import Tuple

system_seeds = ["5", "6", "7", "8", "9"]
# system_seeds = ["0", "1", "2", "3", "4"]

agent_action_pairs = [
    [5, 5],
    # [5, 6],
    # [6, 4],
    # [6, 5],
    # [6, 6],
]
env_seeds = [42]
system_names = [
    # "ff_ippo_tabular",
    "ff_ippo_tabular_split",
    "ff_ppo_central_tabular",
    # "ff_ippo",
    # "ff_mappo",
    # "ff_ppo_central",
]
num_steps = [
    # "1M",
    # "5M",
    # "10M",
    # "20M",
    "50M"
]


def get_eval_updates(step_count: str) -> Tuple[int, int]:
    if step_count == "1M":
        return (165, 33)
    elif step_count == "5M":
        return (815, 163)
    elif step_count == "10M":
        return (1630, 326)
    elif step_count == "20M":
        return (3255, 651)
    elif step_count == "50M":
        return (8140, 814)


def get_script_contents(
    system_name: str,
    system_seed: str,
    env_seed: int,
    task_name: str,
    num_agent: int,
    num_actions: int,
    num_steps: str,
    is_shadowed: bool = False,
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

    neptune_tag = f'["grid-plot-exp-{num_steps}"]'
    num_updates, num_evals = get_eval_updates(num_steps)

    shadowed_str = "True" if is_shadowed else "False"

    script = f"""\
        #!/bin/bash
        # Script auto-generated by experiment_launch_exps.py on {timestamp}
        python {system_run_file} -m system.seed={system_seed} \\
        arch.num_evaluation={num_evals} system.num_updates={num_updates} \\
        env.scenario.task_name={task_name} env.scenario.task_config.num_agents={num_agent} \\
        env.scenario.task_config.num_actions={num_actions} \\
        env.scenario.task_config.key_integer={env_seed} \\
        logger.kwargs.neptune_tag='{neptune_tag}' \\
        env.kwargs.generate_shadowed_payoffs={shadowed_str} \\
        """

    return textwrap.dedent(script)


is_shadowed_list = [True]


def should_run(system_name: str, task_name: str) -> bool:
    return True


def make_task_name(num_agents: int, num_actions: int) -> str:
    return f"matrax-{num_agents}-ag-{num_actions}-act"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for is_shadowed in is_shadowed_list:
        for num_step in num_steps:
            for system_name in system_names:
                for pair in agent_action_pairs:
                    num_agent = pair[0]
                    num_action = pair[1]
                    for env_seed in env_seeds:
                        for system_seed in system_seeds:
                            task_name = make_task_name(num_agent, num_action)

                            if should_run(system_name, task_name):
                                logging.info(f"Running experiment {system_name} - {task_name}")

                                script_contents = get_script_contents(
                                    system_name=system_name,
                                    env_seed=env_seed,
                                    system_seed=system_seed,
                                    task_name=task_name,
                                    num_agent=num_agent,
                                    is_shadowed=is_shadowed,
                                    num_actions=num_action,
                                    num_steps=num_step,
                                )
                                with open("run.sh", "w") as f:
                                    f.write(script_contents)
                                try:
                                    subprocess.run(["./run.sh"], check=True)
                                    logging.info("Experiment launched successfully")
                                    time.sleep(5)

                                except subprocess.CalledProcessError as e:
                                    logging.error(f"Error launching the experiment: {e}")
