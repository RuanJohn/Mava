import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load your datasets
data_1m = pd.read_csv("1M_data_aggregated.csv")
data_5m = pd.read_csv("5M_data_aggregated.csv")
data_10m = pd.read_csv("10M_data_aggregated.csv")

# Rename the algorithms
rename_mapping = {
    "ff_ppo_central_tabular": "ff_ppo_central",
    "ff_ippo_tabular": "ippo",
    "ff_ippo_tabular_split": "ippo",
}

data_1m["algorithm"] = data_1m["algorithm"].replace(rename_mapping)
data_5m["algorithm"] = data_5m["algorithm"].replace(rename_mapping)
data_10m["algorithm"] = data_10m["algorithm"].replace(rename_mapping)

# Add timesteps columns
data_1m["timesteps"] = 1_000_000
data_5m["timesteps"] = 5_000_000
data_10m["timesteps"] = 10_000_000

# Combine all datasets into one
combined_data = pd.concat([data_1m, data_5m, data_10m], ignore_index=True)

# Encode the algorithm as a binary variable (0 for 'ippo', 1 for 'ff_ppo_central')
combined_data["algorithm"] = combined_data["algorithm"].apply(
    lambda x: 1 if x == "ff_ppo_central" else 0
)

# Define the features (X) and target (y)
X = combined_data[["num_agents", "num_actions", "timesteps"]]
y = combined_data["algorithm"]

# Create polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Fit a polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Create a meshgrid for the 3D plot
num_agents_range = np.linspace(X["num_agents"].min(), X["num_agents"].max(), 20)
num_actions_range = np.linspace(X["num_actions"].min(), X["num_actions"].max(), 20)
timesteps_range = np.linspace(X["timesteps"].min(), X["timesteps"].max(), 20)

num_agents_grid, num_actions_grid, timesteps_grid = np.meshgrid(
    num_agents_range, num_actions_range, timesteps_range
)
X_grid = np.c_[num_agents_grid.ravel(), num_actions_grid.ravel(), timesteps_grid.ravel()]

# Transform the grid to polynomial features
X_grid_poly = poly.transform(X_grid)

# Predict using the polynomial regression model
predictions = poly_model.predict(X_grid_poly)
predictions = predictions.reshape(num_agents_grid.shape)

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot with color based on the predicted value
scatter = ax.scatter(
    num_agents_grid,
    num_actions_grid,
    timesteps_grid / 1_000_000,
    c=predictions,
    cmap="coolwarm",
    alpha=0.95,
)

ax.set_xlabel("Number of Agents")
ax.set_ylabel("Number of Actions")
ax.set_zlabel("Timesteps (in millions)")
ax.set_title("Algorithm Prediction Surface")

# Add a color bar
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Algorithm (0: Centralized, 1: Independent)")

# plt.savefig("3d_curve_and_plot.png")
plt.show()
