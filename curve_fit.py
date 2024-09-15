import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Combine all data
data_1m = pd.read_csv("1M_data_aggregated.csv")
data_5m = pd.read_csv("5M_data_aggregated.csv")
data_10m = pd.read_csv("10M_data_aggregated.csv")

data_1m["timesteps"] = 1
data_5m["timesteps"] = 5
data_10m["timesteps"] = 10

data = pd.concat([data_1m, data_5m, data_10m])

# Rename algorithms
data["algorithm"] = data["algorithm"].replace(
    {
        "ff_ppo_central_tabular": "ff_ppo_central",
        "ff_ippo_tabular": "ippo",
        "ff_ippo_tabular_split": "ippo",
    }
)

# Prepare features and target
X = data[["timesteps", "num_agents", "num_actions"]]
y = (data["algorithm"] == "ff_ppo_central").astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Print model performance
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)
print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Get coefficients
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# Print equation
feature_names = ["timesteps", "num_agents", "num_actions"]
equation = f"z = {intercept:.4f}"
for name, coef in zip(feature_names, coefficients):
    equation += f" + {coef:.4f} * {name}"

print("\nFinal equation:")
print(equation)
print("\nProbability of choosing ff_ppo_central:")
print("P(ff_ppo_central) = 1 / (1 + e^(-z))")
