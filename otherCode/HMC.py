r"""°°°
# Hamiltonian Monte Carlo (HMC) Algorithm

Hamiltonian Monte Carlo (HMC) is a Markov Chain Monte Carlo (MCMC) method that uses Hamiltonian dynamics to propose moves in a way that explores the target distribution more efficiently than random-walk-based methods like Metropolis-Hastings.

The core idea is to introduce auxiliary momentum variables and simulate a system of particles evolving according to the laws of Hamiltonian dynamics. The positions correspond to the target distribution's parameters, and the momenta are auxiliary variables used to facilitate exploration.

## Key Components of the HMC Algorithm

1. **Hamiltonian Dynamics**: The system's evolution is governed by the Hamiltonian, which is the sum of potential energy (related to the target distribution) and kinetic energy (related to the auxiliary momentum variables).
2. **Momentum Variables**: These are introduced to help explore the state space efficiently.
3. **Leapfrog Integration**: This is a numerical method to simulate the continuous dynamics of the system using discrete steps.

### Hamiltonian Function

The Hamiltonian function $H(x, p)$ is the sum of the **potential energy** $U(x)$ and the **kinetic energy** $K(p)$:

$$
H(x, p) = U(x) + K(p)
$$

- $U(x)$ is the potential energy, which corresponds to the negative log of the target distribution. If the target distribution is $p(x)$, then:

$$
U(x) = -\log(p(x))
$$

- $K(p)$ is the kinetic energy, which is typically chosen to be a Gaussian distribution over the momentum variables $p$. For simplicity, we often assume the mass matrix $M = I$, the identity matrix. Hence, the kinetic energy is:

$$
K(p) = \frac{1}{2} p^T M^{-1} p
$$

For a unit mass matrix, $M^{-1} = I$, and we have:

$$
K(p) = \frac{1}{2} p^T p
$$

Thus, the Hamiltonian becomes:

$$
H(x, p) = -\log(p(x)) + \frac{1}{2} p^T p
$$

### Equations of Motion

Hamilton’s equations describe how the position $x$ and momentum $p$ evolve over time. These equations are derived from the Hamiltonian:

1. **For position $x$**, the rate of change is the gradient of the Hamiltonian with respect to the momentum $p$:

$$
\frac{dx}{dt} = \frac{\partial H(x, p)}{\partial p} = M^{-1} p
$$

2. **For momentum $p$**, the rate of change is the negative gradient of the Hamiltonian with respect to position $x$:

$$
\frac{dp}{dt} = -\frac{\partial H(x, p)}{\partial x} = -\nabla U(x)
$$

These equations describe the continuous time dynamics of the system.

### Leapfrog Integration

To simulate the continuous Hamiltonian dynamics discretely, we use the **leapfrog integration method**. The leapfrog algorithm alternates between updating the momentum and position over small time steps.

1. **Half-step update of momentum**:

$$
p_{\frac{1}{2}} = p - \frac{\epsilon}{2} \nabla U(x)
$$

2. **Full-step update of position**:

$$
x' = x + \epsilon M^{-1} p_{\frac{1}{2}}
$$

3. **Half-step update of momentum**:

$$
p' = p_{\frac{1}{2}} - \frac{\epsilon}{2} \nabla U(x')
$$

Here, $\epsilon$ is the step size, and $M^{-1}$ is the inverse mass matrix (usually the identity matrix for simplicity).

### HMC Algorithm

The full Hamiltonian Monte Carlo algorithm proceeds as follows:

1. **Initialization**: Start with an initial position $x_0$.
2. **Momentum Sampling**: Sample an initial momentum $p_0 \sim \mathcal{N}(0, M)$, typically a Gaussian distribution with mean 0 and covariance $M$.
3. **Leapfrog Integration**: Perform $L$ leapfrog steps to simulate the Hamiltonian dynamics and propose a new state $(x', p')$.
4. **Metropolis Acceptance Step**: Accept the proposed state with probability:

$$
\alpha = \min \left( 1, \exp \left( -H(x', p') + H(x_0, p_0) \right) \right)
$$

If the move is accepted, set $x_0 = x'$. If rejected, keep $x_0$ unchanged.

### Full Algorithm Steps

```python
# HMC Algorithm
1. Initialize x_0
2. For iteration in 1 to N:
   1. Sample momentum p_0 ~ N(0, M)
   2. Simulate dynamics using leapfrog for L steps
   3. Propose new state (x', p')
   4. Calculate acceptance probability:
      α = min(1, exp(-H(x', p') + H(x_0, p_0)))
   5. Accept or reject the new state based on α
```

## Derivations

### Hamiltonian Function

The Hamiltonian is given by:

$$
H(x, p) = U(x) + K(p)
$$

Where:

- $U(x) = -\log(p(x))$ (the potential energy based on the target distribution).
- $K(p) = \frac{1}{2} p^T M^{-1} p$ (the kinetic energy, assuming unit mass).

### Equations of Motion

Hamilton’s equations describe the time evolution of position and momentum:

1. **For position $x$**:

$$
\frac{dx}{dt} = \frac{\partial H}{\partial p} = M^{-1} p
$$

2. **For momentum $p$**:

$$
\frac{dp}{dt} = -\frac{\partial H}{\partial x} = -\nabla U(x)
$$

### Leapfrog Integration

The leapfrog method involves three main steps:

1. **Half-step update for momentum**:

$$
p_{\frac{1}{2}} = p - \frac{\epsilon}{2} \nabla U(x)
$$

2. **Full-step update for position**:

$$
x' = x + \epsilon M^{-1} p_{\frac{1}{2}}
$$

3. **Half-step update for momentum**:

$$
p' = p_{\frac{1}{2}} - \frac{\epsilon}{2} \nabla U(x')
$$

### Metropolis Acceptance Probability

The acceptance criterion ensures the detailed balance of the Markov Chain. The acceptance probability is given by:

$$
\alpha = \min \left( 1, \exp \left( -H(x', p') + H(x_0, p_0) \right) \right)
$$

This step ensures that the trajectory is reversible and that the algorithm samples from the correct distribution.

---

## Conclusion

Hamiltonian Monte Carlo is a powerful technique for efficiently sampling from complex, high-dimensional distributions. By leveraging the dynamics of Hamiltonian mechanics, HMC avoids the random-walk behavior typical of simpler MCMC methods and allows for more efficient exploration of the target distribution.
°°°"""

# |%%--%%| <hxhXs7yxKu|woyvkj0Uli>
import numpy as np
import matplotlib.pyplot as plt


# Target distribution: Unnormalized log-probability (negative potential energy)
def log_target_distribution(x):
    return -0.5 * x**2  # Example: Standard Gaussian distribution


# Gradient of the log-target distribution (negative gradient of potential energy)
def grad_log_target_distribution(x):
    return -x  # Derivative of -0.5 * x^2 is -x


# Hamiltonian Monte Carlo (HMC) function
def hmc(
    log_target, grad_log_target, initial_position, step_size, num_steps, num_samples
):
    position = initial_position
    samples = []

    for _ in range(num_samples):
        # Sample momentum from a standard Gaussian distribution
        momentum = np.random.normal(0, 1)

        # Initial energy (Hamiltonian)
        current_U = -log_target(position)
        current_K = 0.5 * momentum**2
        current_H = current_U + current_K

        # Leapfrog integration
        new_position, new_momentum = position, momentum
        for _ in range(num_steps):
            # Half-step update for momentum
            new_momentum -= 0.5 * step_size * grad_log_target(new_position)
            # Full-step update for position
            new_position += step_size * new_momentum
            # Half-step update for momentum
            new_momentum -= 0.5 * step_size * grad_log_target(new_position)

        # Proposed energy (Hamiltonian)
        proposed_U = -log_target(new_position)
        proposed_K = 0.5 * new_momentum**2
        proposed_H = proposed_U + proposed_K

        # Metropolis-Hastings acceptance step
        if np.random.rand() < np.exp(current_H - proposed_H):
            position = new_position  # Accept proposal
        samples.append(position)

    return np.array(samples)


# Parameters
initial_position = 0.0  # Starting point
step_size = 0.1  # Leapfrog step size
num_steps = 10  # Number of Leapfrog steps per iteration
num_samples = 1000  # Number of samples to generate

# Run HMC
samples = hmc(
    log_target_distribution,
    grad_log_target_distribution,
    initial_position,
    step_size,
    num_steps,
    num_samples,
)

# Plot results
plt.hist(samples, bins=30, density=True, alpha=0.5, label="HMC Samples")
x = np.linspace(-5, 5, 1000)
plt.plot(
    x,
    np.exp(log_target_distribution(x)) / np.sqrt(2 * np.pi),
    label="True Distribution",
)
plt.title("HMC Sampling from a Gaussian Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()
