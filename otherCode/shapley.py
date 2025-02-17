import itertools
import numpy as np
import math


# Define a simple prediction function (e.g., a linear model)
def model(X):
    return 3 * X[0] + 5 * X[1]  # Example function f(x) = 3x1 + 5x2


# Compute Shapley values
def shapley_values(model, X, feature_idx):
    """
    Computes the Shapley value for a given feature index.

    Args:
        model: The function that makes predictions.
        X: The input feature vector (numpy array).
        feature_idx: Index of the feature for which to compute Shapley value.

    Returns:
        Shapley value for the feature.
    """
    n = len(X)  # Number of features
    feature_indices = list(range(n))  # List of feature indices

    S_values = []  # Store marginal contributions
    weights = []  # Store weights for averaging

    # Iterate over all possible subsets of features excluding the current feature
    for S in itertools.chain.from_iterable(
        itertools.combinations(feature_indices, r) for r in range(n + 1)
    ):
        S = set(S)

        if feature_idx in S:
            continue  # Skip subsets that already contain the feature

        # Compute marginal contribution
        S_with = S | {feature_idx}  # Add the feature to the subset

        # Convert subsets to feature vectors
        X_S = np.array([X[i] if i in S else 0 for i in range(n)])  # Without feature
        X_S_with = np.array(
            [X[i] if i in S_with else 0 for i in range(n)]
        )  # With feature

        # Compute model outputs
        f_S = model(X_S)  # Without feature
        f_S_with = model(X_S_with)  # With feature

        # Compute marginal contribution
        marginal_contribution = f_S_with - f_S
        # Compute weighting factor: |S|!(n-|S|-1)! / n!
        weight = (
            math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
        )

        # Store values

        # Store values
        S_values.append(marginal_contribution)
        weights.append(weight)

    # Compute Shapley value as the weighted sum of marginal contributions
    shapley_value = np.sum(np.array(S_values) * np.array(weights))
    return shapley_value


# Example input features
X = np.array([2, 4])  # Feature values: x1 = 2, x2 = 4

# Compute Shapley values for each feature
shapley_x1 = shapley_values(model, X, feature_idx=0)
shapley_x2 = shapley_values(model, X, feature_idx=1)

# Print results
print(f"Shapley Value for x1: {shapley_x1}")
print(f"Shapley Value for x2: {shapley_x2}")
