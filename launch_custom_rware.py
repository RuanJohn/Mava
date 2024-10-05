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

# task_names = [
#     "small-4ag",
#     "tiny-4ag",
#     "tiny-2ag",
# ]
# system_names = [
#     "ff_ippo",
#     "ff_mappo",
# ]

exp_runner_details = {
    # "small-4ag": {
    #     "ff_mappo": "8,9",
    # },
    "tiny-4ag": {
        "ff_ippo": "8,9",
    },
    "tiny-2ag": {
        "ff_ippo": "6,7,8,9",
    },
}


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
        env={env_name} env/scenario={task_name}
        """

    return textwrap.dedent(script)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for task_name in exp_runner_details:
        for system_name in exp_runner_details[task_name]:
            seed_string = exp_runner_details[task_name][system_name]
            logging.info(f"Running experiment {system_name} - {task_name}")

            script_contents = get_script_contents(
                system_name=system_name,
                env_name="rware",
                task_name=task_name,
                seed_string=seed_string,
            )
            with open("run.sh", "w") as f:
                f.write(script_contents)
            try:
                subprocess.run(["./run.sh"], check=True)
                logging.info("Experiment launched successfully")

            except subprocess.CalledProcessError as e:
                logging.error(f"Error launching the experiment: {e}")
