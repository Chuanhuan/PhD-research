# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.spatial.distance import cdist

# Generate synthetic dataset (non-linearly separable)
np.random.seed(42)
X1 = np.random.randn(50, 2) * 0.3 + np.array([0, 0])  # Class 1 (centered at 0,0)
X2 = np.random.randn(50, 2) * 0.3 + np.array([1, 1])  # Class 2 (centered at 1,1)
X = np.vstack((X1, X2))
y = np.hstack((np.ones(50), -np.ones(50)))  # Labels: +1 and -1

# ----- 1️⃣ Gaussian SVM -----
svm = SVC(kernel="rbf", gamma=100.0, C=1.0)
svm.fit(X, y)
sv_support_vectors = svm.support_vectors_  # Extract support vectors
print(sv_support_vectors.shape)


# ----- 2️⃣ RBF Neural Network from Scratch -----
class RBFNetwork:
    def __init__(self, centers, gamma=1.0):
        self.centers = centers  # Predefined centers (like SVs)
        self.gamma = gamma
        self.weights = np.random.randn(len(centers))  # Random initial weights
        self.bias = np.random.randn()  # Bias term

    def rbf(self, X):
        """Compute Gaussian RBF for each center."""
        return np.exp(-self.gamma * cdist(X, self.centers, metric="sqeuclidean"))

    def forward(self, X):
        """Compute network output."""
        Phi = self.rbf(X)  # Compute RBF activations
        return Phi @ self.weights + self.bias  # Linear aggregation

    def train(self, X, y, lr=0.1, epochs=100):
        """Train the RBF network using gradient descent."""
        for _ in range(epochs):
            Phi = self.rbf(X)
            y_pred = Phi @ self.weights + self.bias
            error = y - y_pred  # Compute error

            # Gradient descent update
            self.weights += lr * Phi.T @ error / len(X)
            self.bias += lr * np.mean(error)


# Train RBF Network (using SVM's support vectors as centers)
rbf_net = RBFNetwork(centers=sv_support_vectors, gamma=1.0)
rbf_net.train(X, y)


# ----- Plot Decision Boundaries -----
def plot_decision_boundary(model, X, y, title, model_type="svm"):
    """Helper function to plot decision boundary."""
    xx, yy = np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100))
    X_test = np.c_[xx.ravel(), yy.ravel()]

    if model_type == "svm":
        Z = model.decision_function(X_test).reshape(xx.shape)
    else:
        Z = model.forward(X_test).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=50, alpha=0.5, cmap="coolwarm")
    plt.scatter(X1[:, 0], X1[:, 1], color="blue", label="Class 1")
    plt.scatter(X2[:, 0], X2[:, 1], color="red", label="Class 2")

    if model_type == "svm":
        plt.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100,
            edgecolors="black",
            facecolors="none",
            label="Support Vectors",
        )
    else:
        plt.scatter(
            model.centers[:, 0],
            model.centers[:, 1],
            s=100,
            edgecolors="black",
            facecolors="none",
            label="RBF Centers",
        )

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()


# Plot results for Gaussian SVM and RBF Network
plot_decision_boundary(svm, X, y, "Gaussian SVM (RBF Kernel)", model_type="svm")
plot_decision_boundary(
    rbf_net, X, y, "RBF Neural Network (Using SVM SVs as Centers)", model_type="rbf"
)

# %%
# Define toy example points and labels
X = np.array([[1, 1], [2, 2], [3, 3], [4, 1], [5, 2], [6, 3]])  # Class +1  # Class -1

y = np.array([1, 1, 1, -1, -1, -1])  # Labels

# Train a soft-margin SVM with a linear kernel
clf = SVC(kernel="linear", C=1.0)
clf.fit(X, y)

# Get the separating hyperplane
w = clf.coef_[0]  # Normal vector to the hyperplane
b = clf.intercept_[0]

# Plot decision boundary and margin
xx = np.linspace(0, 7, 100)
yy = (-w[0] * xx - b) / w[1]

# Support vectors
sv = X[clf.support_]

# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(
    X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=100, label="Data points"
)
plt.scatter(
    sv[:, 0],
    sv[:, 1],
    s=200,
    facecolors="none",
    edgecolors="k",
    label="Support Vectors",
)

plt.plot(xx, yy, "k-", label="Decision Boundary")
plt.legend()
plt.xlim(0, 7)
plt.ylim(0, 4)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Soft-Margin SVM with Support Vectors")

plt.show()
# print g(x) = w^T x + b

for point in X:
    print(f"g({point}) = {np.dot(w, point) + b}")
print(f"g({sv})")


# Compute decision function values for all points
decision_values = np.dot(X, w) + b

# Compute y_i * f(x_i) to check which points satisfy the margin conditions
margin_check = y * decision_values

# Support vectors are where 0 < alpha < C, which corresponds to y_i * f(x_i) ≈ 1
support_vector_manual = X[
    (margin_check <= 1 + 1e-5)
]  # Adding small tolerance for numerical stability

# Print the manually computed support vectors
print(support_vector_manual)
