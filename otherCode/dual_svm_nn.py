import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cvxopt import matrix, solvers

# -------------------------
# 1. Create a small synthetic dataset
# -------------------------
torch.manual_seed(42)
np.random.seed(42)
N = 100  # number of data points
input_dim = 2  # input features dimension

# Generate random inputs
X = torch.randn(N, input_dim)

# Generate binary labels in {-1, 1}
y = torch.randint(0, 2, (N,)).float() * 2 - 1  # converts 0/1 to -1/1


# -------------------------
# 2. Define a simple neural network as a feature extractor
# -------------------------
class Net(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Net, self).__init__()
        # A single linear layer mapping input to feature space.
        self.fc = nn.Linear(input_dim, feature_dim)

    def forward(self, x):
        return self.fc(x)


feature_dim = 5  # dimension of the feature space φ(x)
model = Net(input_dim, feature_dim)

# -------------------------
# 3. Parameterize the dual variables (one per training example)
# -------------------------
# We define raw_alpha as an unconstrained parameter.
raw_alpha = nn.Parameter(torch.zeros(N))
C = 1.0  # Regularization parameter for box constraints


def get_alpha(raw_alpha):
    """Map raw_alpha to alpha in [0, C] using the sigmoid function."""
    return C * torch.sigmoid(raw_alpha)


# -------------------------
# 4. Define the dual SVM loss function
# -------------------------
def dual_svm_loss(phi, y, raw_alpha, lambda_eq=10.0):
    """
    Computes the loss based on the dual SVM objective:

      L_dual = sum_i α_i - ½ ∑_{ij} α_i α_j y_i y_j K(x_i,x_j)

    where K(x_i, x_j)= φ(x_i)^T φ(x_j).

    To use it for minimization we define:

      loss = - L_dual + penalty

    with a squared penalty enforcing the equality constraint
      ∑_i α_i y_i = 0.
    """
    # Compute the kernel matrix: K = φ φᵀ, shape (N, N)
    K = phi @ phi.t()

    # Get the dual variables (ensured to be in [1,C])
    alpha = get_alpha(raw_alpha)  # shape: (N,)

    term1 = torch.sum(alpha)
    ay = alpha * y
    term2 = 0.5 * (ay @ (K @ ay))

    dual_obj = term1 - term2
    # Soft penalty to enforce: sum_i α_i y_i = 0.
    constraint_violation = torch.sum(alpha * y)
    penalty = lambda_eq * (constraint_violation**2)

    loss = -dual_obj + penalty
    return loss


# -------------------------
# 5. Train the neural network with the dual SVM loss
# -------------------------
optimizer = optim.Adam(list(model.parameters()) + [raw_alpha], lr=0.01)
num_epochs = 5000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    phi = model(X)  # get feature representation φ(x)
    loss = dual_svm_loss(phi, y, raw_alpha, lambda_eq=10.0)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Get the neural network’s learned dual variables (alpha)
alpha_nn = get_alpha(raw_alpha).detach().numpy()
print("\nNeural network dual variables (alpha) after training:")
print(alpha_nn)

# -------------------------
# 6. Solve the dual SVM QP using CVXOPT for verification
# -------------------------
# Use the fixed feature representation from the trained network.
phi_np = model(X).detach().numpy()  # shape: (N, feature_dim)
K = phi_np @ phi_np.T  # Kernel matrix computed from features

# Convert labels to a NumPy array
y_np = y.detach().numpy().reshape(-1)

# Build the QP matrices for the dual problem:
#   minimize   (1/2) αᵀ P α - qᵀ α
#   subject to  G α ≤ h and A α = b,
#
# where:
#   P = (y_i y_j K_{ij}),  q = -1 (vector)
#   Inequality constraints:  0 ≤ α_i ≤ C
#   Equality constraint:       ∑_i α_i y_i = 0

P_np = np.outer(y_np, y_np) * K  # shape (N, N)
q_np = -np.ones(N)

# For inequalities, we need to enforce:  α_i ≤ C  and  -α_i ≤ 0.
G_np = np.vstack((np.eye(N), -np.eye(N)))  # α_i <= C  # -α_i <= 0
h_np = np.hstack((C * np.ones(N), np.zeros(N)))  # upper bound C  # lower bound 0

A_np = y_np.reshape(1, -1)  # shape (1, N)
b_np = np.zeros(1)

# Convert to cvxopt matrices (type 'double')
P_cvx = matrix(P_np.astype(np.double))
q_cvx = matrix(q_np.astype(np.double))
G_cvx = matrix(G_np.astype(np.double))
h_cvx = matrix(h_np.astype(np.double))
A_cvx = matrix(A_np.astype(np.double))
b_cvx = matrix(b_np.astype(np.double))

# Optionally, disable solver progress output
solvers.options["show_progress"] = False

# Solve the QP problem
solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
alpha_cvx = np.array(solution["x"]).flatten()
print("\nCVXOPT dual variables (alpha) solution:")
print(alpha_cvx)


# -------------------------
# 7. Compare Dual Objectives
# -------------------------
def dual_objective(alpha, phi, y):
    """
    Compute the dual objective value:
      L_dual = ∑_i α_i - ½ (α * y)ᵀ K (α * y)
    where K = φ φᵀ.
    """
    K = phi @ phi.T
    term1 = np.sum(alpha)
    term2 = 0.5 * ((alpha * y) @ (K @ (alpha * y)))
    return term1 - term2


dual_obj_nn = dual_objective(alpha_nn, phi_np, y_np)
dual_obj_cvx = dual_objective(alpha_cvx, phi_np, y_np)
print("\nDual objective from NN solution:  {:.4f}".format(dual_obj_nn))
print("Dual objective from CVXOPT solution: {:.4f}".format(dual_obj_cvx))
