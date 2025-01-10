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
    # "ff_ppo_central",
    # "rec_ppo_central",
    # "ff_ppo_central_factored",
    "rec_ppo_central_factored",
]

scenarios_to_run = [
    "smacv2_5_units",
    "2s3z",
    "con-7x7x5a",
    "15x15-4p-5f",
    "3s_vs_5z",
    "15x15-4p-3f",
    "10x10-3p-3f",
    "8x8-2p-2f-coop",
    "15x15-3p-5f",
    "2s-8x8-2p-2f-coop",
    "medium-6ag",
    "xlarge-4ag",
    "2s-10x10-3p-3f",
    "xlarge-4ag-hard",
    "con-5x5x3a",
    "large-4ag-hard",
    "small-4ag-hard",
    "large-4ag",
    "small-4ag",
    "tiny-4ag-hard",
    "medium-4ag-hard",
    "medium-4ag",
    "tiny-2ag",
    "tiny-4ag",
    "tiny-2ag-hard",
]

scenarios_to_skip = [
    "con-15x15x23a",
    "con-10x10x10a",
    "con-7x7x5a",
    "con-5x5x3a",
    "xlarge-4ag",
    "xlarge-4ag-hard",
    "large-4ag-hard",
    "xlarge-4ag",
    "medium-4ag-hard",
    "large-8ag",
    "large-8ag-hard",
    "medium-4ag",
    "medium-6ag",
]

_system_run_file_map = {
    "ff_ppo_central": "mava/systems/ppo/anakin/ff_ppo_central.py",
    "rec_ppo_central": "mava/systems/ppo/anakin/rec_ppo_central.py",
    "ff_ppo_central_factored": "mava/systems/ppo/anakin/ff_ppo_central_factored.py",
    "rec_ppo_central_factored": "mava/systems/ppo/anakin/rec_ppo_central_factored.py",
}


def compute_should_run(system_name: str, scenario: str) -> bool:
    if system_name in systems_to_run and scenario not in scenarios_to_skip:
        return True
    else:
        return False


def get_script_contents(
    system_name: str,
    env: str,
    scenario: str,
    actor_lr: float,
    clip_eps: float,
    ent_coef: float,
    max_grad_norm: float,
    num_minibatches: int,
    ppo_epochs: int,
    critic_lr: float,
    recurrent_chunk_size: int,
) -> str:
    scenario_job_name = scenario

    if scenario == "humanoid_9|8":
        scenario = '"humanoid_9|8"'
        scenario_job_name = "humanoid"

    system_run_file = _system_run_file_map[system_name]

    job_name = f"benchmark-{scenario_job_name}"

    if system_name == "rec_ppo_central":
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

    python {system_run_file} -m system.seed=0,1,2,3,4,5,6,7,8,9 \\
    env={env} \\
    system.actor_lr={actor_lr}  \\
    system.critic_lr={critic_lr}  \\
    system.clip_eps={clip_eps} \\
    system.ent_coef={ent_coef} \\
    system.max_grad_norm={max_grad_norm} \\
    system.num_minibatches={num_minibatches} \\
    system.ppo_epochs={ppo_epochs} \\
    """)

    # Directly append without extra indentation
    if system_name.startswith("rec"):
        base_script += f"system.recurrent_chunk_size={recurrent_chunk_size} \\\n"

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
    df = pd.read_csv("best_hyperparams/central_ppo_factored.csv")

    for _, row in df.iterrows():
        system_name = row["system_name"]
        scenario = row["task"]
        env = row["env_name"]
        actor_lr = safe_cast(row["actor_lr"], float)
        clip_eps = safe_cast(row["clip_eps"], float)
        ent_coef = safe_cast(row["ent_coef"], float)
        max_grad_norm = safe_cast(row["max_grad_norm"], float)
        num_minibatches = safe_cast(row["num_minibatches"], int)
        ppo_epochs = safe_cast(row["ppo_epochs"], int)
        critic_lr = safe_cast(row["critic_lr"], float)
        recurrent_chunk_size = safe_cast(row["recurrent_chunk_size"], int)

        should_run = compute_should_run(system_name, scenario)

        if should_run:
            logging.info(f"Running experiment - {system_name} - {scenario}")

            script_contents = get_script_contents(
                system_name=system_name,
                env=env,
                scenario=scenario,
                actor_lr=actor_lr,
                clip_eps=clip_eps,
                ent_coef=ent_coef,
                max_grad_norm=max_grad_norm,
                num_minibatches=num_minibatches,
                ppo_epochs=ppo_epochs,
                critic_lr=critic_lr,
                recurrent_chunk_size=recurrent_chunk_size,
            )
            with open("run.sh", "w") as f:
                f.write(script_contents)
            try:
                logging.info(f'Attempting to submit: "{system_name}" - "{scenario}"')
                subprocess.run(["sbatch", "run.sh"], check=True)
                time.sleep(3)
            except Exception as e:
                logging.error(f"Error submitting job: {e}")
