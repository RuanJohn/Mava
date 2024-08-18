import concurrent.futures
import os

import neptune
import pandas as pd
from neptune.exceptions import MissingFieldException


def fetch_run_data(run_id, project_name, api_token):
    try:
        run = neptune.init_run(
            project=project_name, api_token=api_token, with_id=run_id, mode="read-only"
        )

        algorithm = run["config/logger/system_name"].fetch()
        num_actions = run["config/env/scenario/task_config/num_actions"].fetch()
        num_agents = run["config/env/scenario/task_config/num_agents"].fetch()

        score_series = run["absolute/episode_return/mean"].fetch_values()
        score = score_series["value"].iloc[-1] if len(score_series) > 0 else None

        print(
            f"Run {run_id}: {algorithm}, {num_agents} agents, {num_actions} actions, score {score}"
        )

        return {
            "algorithm": algorithm,
            "num_actions": num_actions,
            "num_agents": num_agents,
            "score": score,
        }
    except MissingFieldException as e:
        print(f"Skipping run {run_id} due to missing field: {e!s}")
        return None


def fetch_neptune_data(project_name, api_token, tag):
    project = neptune.init_project(project=project_name, api_token=api_token, mode="read-only")
    runs = project.fetch_runs_table(tag=tag).to_pandas()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_run = {
            executor.submit(fetch_run_data, run_id, project_name, api_token): run_id
            for run_id in runs["sys/id"].values
        }
        data = []
        for future in concurrent.futures.as_completed(future_to_run):
            run_data = future.result()
            if run_data is not None:
                data.append(run_data)

    df = pd.DataFrame(data)

    # Aggregate data by taking the mean of scores for each combination
    aggregated_df = (
        df.groupby(["algorithm", "num_agents", "num_actions"])["score"].mean().reset_index()
    )

    return aggregated_df


def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def main():
    project_name = "ruan-marl-masters/centralised-marl-msc"
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    tag = "grid-plot-exp-1M"

    df = fetch_neptune_data(project_name, api_token, tag)

    # Save the data to a CSV file
    csv_filename = "1M_data_aggregated.csv"
    save_to_csv(df, csv_filename)


if __name__ == "__main__":
    main()
