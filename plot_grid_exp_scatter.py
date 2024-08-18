import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(filename):
    df = pd.read_csv(filename)
    df["num_agents"] = df["num_agents"].astype(int)
    df["num_actions"] = df["num_actions"].astype(int)
    return df


def create_scatter_plot(df):
    if df.empty:
        print("No data to plot. The CSV file is empty.")
        return

    # Create a pivot table with the best algorithm for each combination
    pivot = df.loc[df.groupby(["num_agents", "num_actions"])["score"].idxmax()]

    # Get unique algorithms and assign a color to each
    algorithms = pivot["algorithm"].unique()
    color_palette = sns.color_palette("husl", n_colors=len(algorithms))
    color_dict = dict(zip(algorithms, color_palette))

    # Create the plot
    # plt.figure(figsize=(16, 12))
    plt.figure(figsize=(8, 6))

    # Create scatter plot
    for algorithm in algorithms:
        data = pivot[pivot["algorithm"] == algorithm]
        plt.scatter(
            data["num_agents"],
            data["num_actions"],
            c=[color_dict[algorithm]],
            label=algorithm,
            s=100,
            alpha=0.7,
        )

    plt.title("Best Algorithm by Number of Agents and Actions", fontsize=16)
    plt.xlabel("Number of Agents", fontsize=12)
    plt.ylabel("Number of Actions", fontsize=12)

    # Set integer ticks for both axes
    plt.xticks(range(min(pivot["num_agents"]), max(pivot["num_agents"]) + 1))
    plt.yticks(range(min(pivot["num_actions"]), max(pivot["num_actions"]) + 1))

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add legend
    plt.legend(title="Algorithms", loc="center left", bbox_to_anchor=(1, 0.5))

    # Adjust layout to prevent cutting off labels and legend
    plt.tight_layout()

    # Save the plot as a high-resolution PNG file
    plt.savefig("best_algorithm_scatter.png", dpi=300, bbox_inches="tight")
    print("Plot saved as 'best_algorithm_scatter.png'")

    # Show the plot (optional, comment out if you don't want to display it)
    # plt.show()


def main():
    filename = "5M_data_aggregated.csv"
    df = load_data(filename)
    create_scatter_plot(df)


if __name__ == "__main__":
    main()
