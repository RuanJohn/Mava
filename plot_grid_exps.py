import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(data_file_name, rename_algorithms=False):
    df = pd.read_csv(data_file_name)
    df["num_agents"] = df["num_agents"].astype(int)
    df["num_actions"] = df["num_actions"].astype(int)

    # rename ff_ppo_central_tabular to ppo_central
    # rename ff_ippo_tabular to ippo
    # rename ff_ippo_tabular_split to ippo
    if rename_algorithms:
        df["algorithm"] = df["algorithm"].replace(
            {
                "ff_ppo_central_tabular": "ppo_central",
                "ff_ippo_tabular": "ippo",
                "ff_ippo_tabular_split": "ippo",
            }
        )

    return df


def create_plot(df, plot_file_name):
    if df.empty:
        print("No data to plot. The CSV file is empty.")
        return

    # Create a pivot table with the best algorithm for each combination
    pivot = df.loc[df.groupby(["num_agents", "num_actions"])["score"].idxmax()]
    pivot = pivot.pivot(index="num_actions", columns="num_agents", values="algorithm")

    # Sort the index (num_actions) in descending order to invert the y-axis
    pivot = pivot.sort_index(ascending=False)

    # Get unique algorithms and assign a color to each
    algorithms = df["algorithm"].unique()
    color_palette = sns.color_palette("husl", n_colors=len(algorithms))
    color_dict = dict(zip(algorithms, color_palette))

    # Create a numerical representation of the pivot table
    pivot_numeric = pivot.applymap(lambda x: algorithms.tolist().index(x) if pd.notnull(x) else -1)

    # Create a mask for combinations with no data
    mask = pivot_numeric == -1

    # Create the plot
    # plt.figure(figsize=(16, 12))
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(pivot_numeric, annot=True, fmt="d", cmap=color_palette, mask=mask, cbar=False)

    # Add 'X' for combinations with no data
    for i, j in zip(*np.where(mask)):
        ax.text(j + 0.5, i + 0.5, "X", ha="center", va="center", color="red", fontweight="bold")

    plt.title("Best Algorithm by Number of Agents and Actions", fontsize=16)
    plt.xlabel("Number of Agents", fontsize=12)
    plt.ylabel("Number of Actions", fontsize=12)

    # Rotate x-axis labels for better readability if needed
    plt.xticks(rotation=0)

    # Create a custom legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
        for color in color_dict.values()
    ]
    plt.legend(
        legend_elements,
        color_dict.keys(),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Algorithms",
    )

    # Adjust layout to prevent cutting off labels and legend
    plt.tight_layout()

    # Save the plot as a high-resolution PNG file
    plt.savefig(f"{plot_file_name}.png", dpi=300, bbox_inches="tight")
    print("Plot saved as 'best_algorithm_heatmap.png'")

    # Show the plot (optional, comment out if you don't want to display it)
    # plt.show()


def main():
    data_file_name = "20M_data_aggregated.csv"
    df = load_data(data_file_name, rename_algorithms=True)
    plot_file_name = "20m_best_algorithm_heatmap"
    create_plot(df, plot_file_name)


if __name__ == "__main__":
    main()
