import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def initialize_parameters(X, n_components):
    """
    Initializes parameters (mu, sigma, pi) for GMM.

    Args:
        X (numpy.ndarray): Input data.
        n_components (int): Number of GMM components.

    Returns:
        tuple: Initialized mu, sigma, and pi.
    """
    n_samples, n_features = X.shape

    # Initialize means randomly from data points
    indices = np.random.choice(n_samples, n_components, replace=False)
    mu = X[indices]

    # Initialize covariances as identity matrices
    sigma = [np.eye(n_features)] * n_components

    # Initialize mixing coefficients uniformly
    pi = np.ones(n_components) / n_components

    return mu, sigma, pi


def e_step(X, mu, sigma, pi):
    """
    Expectation step: Calculates responsibilities (gamma).

    Args:
        X (numpy.ndarray): Input data.
        mu (numpy.ndarray): Means of GMM components.
        sigma (list): Covariances of GMM components.
        pi (numpy.ndarray): Mixing coefficients of GMM components.

    Returns:
        numpy.ndarray: Responsibilities (gamma).
    """
    n_samples, n_components = X.shape[0], len(pi)
    gamma = np.zeros((n_samples, n_components))

    for k in range(n_components):
        gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])

    gamma_sum = np.sum(gamma, axis=1, keepdims=True)
    gamma /= gamma_sum

    return gamma


def m_step(X, gamma):
    """
    Maximization step: Updates parameters (mu, sigma, pi).

    Args:
        X (numpy.ndarray): Input data.
        gamma (numpy.ndarray): Responsibilities.

    Returns:
        tuple: Updated mu, sigma, and pi.
    """
    n_samples, n_features = X.shape
    n_components = gamma.shape[1]

    # Calculate Nk (sum of responsibilities for each component)
    Nk = np.sum(gamma, axis=0)

    # Update mixing coefficients (pi)
    pi = Nk / n_samples

    # Update means (mu)
    mu = np.zeros((n_components, n_features))
    for k in range(n_components):
        mu[k] = np.sum(gamma[:, k, np.newaxis] * X, axis=0) / Nk[k]

    # Update covariances (sigma)
    sigma = []
    for k in range(n_components):
        diff = X - mu[k]
        sigma_k = (gamma[:, k, np.newaxis] * diff).T @ diff / Nk[k]
        sigma.append(sigma_k)

    return mu, sigma, pi


def fit_gmm(X, n_components, n_iterations=100):
    """
    Fits a Gaussian Mixture Model to the data using EM algorithm.

    Args:
        X (numpy.ndarray): Input data.
        n_components (int): Number of GMM components.
        n_iterations (int): Number of EM iterations.

    Returns:
        tuple: Fitted mu, sigma, pi, and responsibilities (gamma).
    """
    mu, sigma, pi = initialize_parameters(X, n_components)

    for _ in range(n_iterations):
        gamma = e_step(X, mu, sigma, pi)
        mu, sigma, pi = m_step(X, gamma)

    return mu, sigma, pi, gamma


def generate_random_data(n_samples=300):
    """
    Generates a random dataset for GMM fitting (for demonstration).

    Args:
        n_samples (int): Number of data samples.

    Returns:
        numpy.ndarray: Randomly generated data.
    """
    np.random.seed(0)  # for reproducibility
    # Create three clusters
    cluster1 = np.random.multivariate_normal(
        [2, 2], [[0.5, 0], [0, 0.5]], n_samples // 3
    )
    cluster2 = np.random.multivariate_normal(
        [8, 3], [[0.8, 0.2], [0.2, 0.8]], n_samples // 3
    )
    cluster3 = np.random.multivariate_normal(
        [3, 6], [[0.3, -0.1], [-0.1, 0.7]], n_samples - 2 * (n_samples // 3)
    )
    X = np.concatenate((cluster1, cluster2, cluster3))
    return X


def plot_gmm(X, mu, sigma, gamma):
    """
    Plots the data points and the fitted GMM components.

    Args:
        X (numpy.ndarray): Input data.
        mu (numpy.ndarray): Means of GMM components.
        sigma (list): Covariances of GMM components.
        gamma (numpy.ndarray): Responsibilities.
    """
    n_components = mu.shape[0]
    colors = ["r", "g", "b"]  # , 'c', 'm', 'y', 'k'] # Extend if more components

    plt.figure(figsize=(8, 6))
    for k in range(n_components):
        # Plot data points colored by responsibilities
        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=colors[k],
            alpha=0.5,
            label=f"Cluster {k+1}" if k == 0 else None,
            marker="o",
        )  # Only label the first one

        # Plot GMM components as ellipses (simplified for 2D)
        eigenvalues, eigenvectors = np.linalg.eigh(sigma[k])
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues) * 2  # Scale for better visualization
        ellipse = plt.matplotlib.patches.Ellipse(
            mu[k],
            width,
            height,
            angle=angle,
            facecolor="none",
            edgecolor=colors[k],
            linewidth=2,
        )
        plt.gca().add_patch(ellipse)

    plt.scatter(
        mu[:, 0], mu[:, 1], c="black", s=100, marker="x", label="Means"
    )  # Plot means
    plt.title("GMM Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate sample data
    X = generate_random_data()

    # Fit GMM with 3 components
    n_components = 3
    mu, sigma, pi, gamma = fit_gmm(X, n_components)

    print("Means (mu):\n", mu)
    print("\nCovariances (sigma):\n", sigma)
    print("\nMixing Coefficients (pi):\n", pi)
    print("\nResponsibilities (gamma - first 5 rows):\n", gamma[:5])

    # Plot the results
    plot_gmm(X, mu, sigma, gamma)
    print(
        f"Shape of X: {X.shape}, mu: {mu.shape}, sigma: {len(sigma)}, pi: {pi.shape}, gamma: {gamma.shape}"
    )
