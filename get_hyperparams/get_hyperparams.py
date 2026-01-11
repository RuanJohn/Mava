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

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import neptune
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from tqdm import tqdm

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
        "tiny-2ag-hard",
        "tiny-4ag",
        "tiny-4ag-hard",
        "small-4ag",
        "small-4ag-hard",
        "medium-6ag",
        "large-8ag",
        "large-8ag-hard",
        "xlarge-4ag",
        "xlarge-4ag-hard",
        "medium-4ag",
        "medium-4ag-hard",
        "large-4ag",
        "large-4ag-hard",
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

# Define algo search spaces
algo_search_spaces = {
    # "ff_ppo_central": [
    #     "config/system/actor_lr",
    #     "config/system/critic_lr",
    #     "config/system/ppo_epochs",
    #     "config/system/num_minibatches",
    #     "config/system/clip_eps",
    #     "config/system/max_grad_norm",
    #     "config/system/ent_coef",
    # ],
    # "rec_ppo_central": [
    #     "config/system/actor_lr",
    #     "config/system/critic_lr",
    #     "config/system/ppo_epochs",
    #     "config/system/num_minibatches",
    #     "config/system/clip_eps",
    #     "config/system/max_grad_norm",
    #     "config/system/ent_coef",
    #     "config/system/recurrent_chunk_size",
    # ],
    "ff_ppo_central_cont_full": [
        "config/system/actor_lr",
        "config/system/critic_lr",
        "config/system/ppo_epochs",
        "config/system/num_minibatches",
        "config/system/clip_eps",
        "config/system/max_grad_norm",
        "config/system/ent_coef",
    ],
}

env_renaming = {
    "RobotWarehouse": "rware",
    "MaBrax": "mabrax",
    "VectorMaConnector": "vector-connector",
    "Smax": "smax",
    "LevelBasedForaging": "lbf",
    "MPE": "mpe",
}

NEPTUNE_TAGS = [
    "ff-ppo-central-full-cont-tune",
]

algo_column_name = "config/logger/system_name"
task_column_name = "config/env/scenario/task_name"


def process_csv(df):
    # Rename columns to keep only the text after the last '/'
    df.columns = [col.split("/")[-1] for col in df.columns]

    # Rename 'task_name' to 'task'
    df.rename(columns={"task_name": "task"}, inplace=True)

    # Rename 'env_name' using the env_renaming dictionary
    df["env_name"] = df["env_name"].replace(env_renaming)

    return df


def get_target_metric(row):
    return "evaluator/episode_return/mean"


def get_absolute_metric(target_metric):
    base_metric_name = "/".join(target_metric.split("/")[1:])
    return f"absolute/{base_metric_name}"


def process_algorithm_task(algo_name, task, algo_df):
    # Filter the dataframe for the current task
    task_df = algo_df[algo_df[task_column_name] == task].copy()

    # Determine the target metric
    target_metric = get_target_metric(task_df.iloc[0])
    absolute_metric = get_absolute_metric(target_metric)

    # Sort by absolute metric (if it exists) and then by target metric
    sort_columns = []
    if absolute_metric in task_df.columns:
        sort_columns.append(absolute_metric)
    if target_metric in task_df.columns:
        sort_columns.append(target_metric)
    
    if sort_columns:
        task_df = task_df.sort_values(sort_columns, ascending=[False] * len(sort_columns))

    # Select top 10 runs
    top_10_runs = task_df.head(10).copy()

    # Initialize lists to store additional metrics
    final_means = []
    aucs = []
    first_means = []
    time_series_total = []

    # Process each of the top 10 runs
    for _, row in top_10_runs.iterrows():
        run_id = row["sys/id"]
        test_run = neptune.init_run(
            project="ruan-marl-masters/centralised-marl-msc",
            with_id=run_id,
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
            mode="read-only",
        )

        # Fetch time series data
        run_time_series = test_run[target_metric].fetch_values()["value"]

        # Compute mean of last 50 values
        final_mean = np.mean(run_time_series[-50:])

        # Compute the mean of the first 30 values
        first_mean = np.mean(run_time_series[:30])

        # Compute area under the curve using scipy.integrate.trapezoid
        curve_auc = trapezoid(y=run_time_series, x=range(len(run_time_series)))

        # Compute the total of the timeseries
        time_series_total.append(np.sum(run_time_series))

        final_means.append(final_mean)
        aucs.append(curve_auc)
        first_means.append(first_mean)

        # Close the run
        test_run.stop()

    # Add new columns to the dataframe
    top_10_runs.loc[:, "final_mean"] = final_means
    top_10_runs.loc[:, "auc"] = aucs
    top_10_runs.loc[:, "first_mean"] = first_means
    top_10_runs.loc[:, "time_series_total"] = time_series_total

    # Sort by all metrics
    sort_cols = ["time_series_total"]
    if absolute_metric in top_10_runs.columns:
        sort_cols.append(absolute_metric)
    if target_metric in top_10_runs.columns:
        sort_cols.append(target_metric)
    top_10_runs = top_10_runs.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    # Select the best run
    best_row = top_10_runs.iloc[0]

    return best_row


def process_combination(algo_name, task, algo_df):
    best_row = process_algorithm_task(algo_name, task, algo_df)

    # Create a dictionary to store the results for this combination
    result = {
        "system_name": algo_name,
        "task_name": task,
        "env_name": best_row["config/env/env_name"],
        "num_envs": 64,
        "num_updates": 1220,
        "num_evaluation": 122,
        "run_id": best_row["sys/id"],
    }

    # Add the target metric value
    target_metric = get_target_metric(best_row)
    if target_metric == "evaluator/win_rate":
        result["win_rate"] = best_row[target_metric]
        result["episode_return"] = np.nan
    else:
        result["episode_return"] = best_row[target_metric]
        result["win_rate"] = np.nan

    # Add the hyperparameters
    for param in full_search_space:
        if param in algo_search_spaces[algo_name]:
            result[param] = best_row.get(param, np.nan)
        else:
            result[param] = np.nan

    # Add the new metrics
    result["final_mean"] = best_row["final_mean"]
    result["auc"] = best_row["auc"]
    result["time_series_total"] = best_row["time_series_total"]

    return result


# Initialize Neptune project
project = neptune.init_project(
    project="ruan-marl-masters/centralised-marl-msc",
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    mode="read-only",
)

# Get all the data
joint_df = pd.DataFrame()

for tag in NEPTUNE_TAGS:
    runs_table = project.fetch_runs_table(tag=tag).to_pandas()
    joint_df = pd.concat([joint_df, runs_table])

joint_df = joint_df.reset_index(drop=True)

# Create a full search space for a joint csv
full_search_space = set()
for search_space in algo_search_spaces.values():
    full_search_space.update(search_space)
full_search_space = list(full_search_space)

# Create a list to store all the combinations
combinations = []
for algo_name in algo_search_spaces.keys():
    algo_df = joint_df[joint_df[algo_column_name] == algo_name]
    tasks = algo_df[task_column_name].unique()
    for task in tasks:
        combinations.append((algo_name, task, algo_df))

# Precompute the total number of task-algorithm combinations
total_combinations = len(combinations)

# Create a progress bar
progress_bar = tqdm(total=total_combinations, desc="Processing task-algorithm combinations")

# Create an empty list to store the results
results = []

# Use ThreadPoolExecutor to process combinations in parallel
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    # Submit all tasks to the executor
    future_to_combo = {
        executor.submit(process_combination, *combo): combo for combo in combinations
    }

    # Process results as they complete
    for future in as_completed(future_to_combo):
        combo = future_to_combo[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as exc:
            print(f"Combination {combo} generated an exception: {exc}")
        finally:
            progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# Create a DataFrame from the results
output_df = pd.DataFrame(results)

# Reorder columns
columns = (
    [
        "system_name",
        "task_name",
        "env_name",
        "run_id",
        "episode_return",
        "win_rate",
        "time_series_total",
    ]
    + list(full_search_space)
    + ["num_envs", "num_updates", "num_evaluation"]
)

output_df = output_df[columns]

output_df = process_csv(output_df)

# Save the DataFrame to a CSV file
file_name = "best_hyperparams/ff_ppo_central_full_cont.csv"
output_df.to_csv(file_name, index=False)

print(f"CSV file {file_name} has been created successfully.")
