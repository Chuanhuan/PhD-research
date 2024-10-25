#%%
#SECTION: non-differentiable sampling
import torch

mu = torch.tensor(0.0, requires_grad=True)
sigma = torch.tensor(1.0, requires_grad=True)

# Direct sampling using torch.normal, which is non-differentiable
z = torch.normal(mu, sigma)

# Loss function
loss = (z - 5)**2

# Print gradients before backward
print("Gradients before backward:")
print("mu.grad:", mu.grad)
print("sigma.grad:", sigma.grad)

# Attempt to compute gradients
loss.backward()

# Print gradients after backward
print("Gradients after backward:")
print("mu.grad:", mu.grad) # returns 0
print("sigma.grad:", sigma.grad) # returns 0


#%%
#SECTION: reparameterization trick

mu = torch.tensor(0.0, requires_grad=True)
log_sigma = torch.tensor(0.0, requires_grad=True)  # we use log(sigma) for numerical stability
sigma = torch.exp(log_sigma)

# Sample epsilon from N(0, 1)
epsilon = torch.randn_like(sigma)

# Reparameterization trick
z = mu + sigma * epsilon

# Loss function
loss = (z - 5)**2

# Print gradients before backward
print("Gradients before backward:")
print("mu.grad:", mu.grad)
print("sigma.grad:", sigma.grad)

# Attempt to compute gradients
loss.backward()

# Print gradients after backward
print("Gradients after backward:")
print("mu.grad:", mu.grad)
print("sigma.grad:", sigma.grad)

#%%
#SECTION: reparameterization trick with sigmoid parameters

mu = torch.tensor([0.0], requires_grad=True)
log_sigma = torch.tensor([0.0], requires_grad=True)
sigma = torch.exp(log_sigma)

# Sample epsilon from N(0, 1)
epsilon = torch.randn_like(sigma)
z = mu + sigma * epsilon

# Instead of hard binarization, use a sigmoid function for c
w = torch.tensor([0.8], requires_grad=True)  # Learnable parameter for c
c = torch.sigmoid(w)  # c will be between 0 and 1, differentiable
# NOTE: c>0 will break the gradients
# c = c>0.5  # Hard binarization , return none

# Now, sample cz
cz = c * z

# Loss function
loss = (cz - 5)**2

# Compute gradients
loss.backward()

# Check gradients
print("Gradient of mu:", mu.grad)
print("Gradient of log_sigma:", log_sigma.grad)
print("Gradient of w (for c):", w.grad)

#%%

import torch

# Logits (raw scores from a neural network, for example)
logits = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)

# Sample Gumbel noise
gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))

# Temperature parameter (lower = closer to one-hot)
tau = 0.5

# Apply Gumbel-Softmax trick
soft_sample = torch.softmax((logits + gumbel_noise) / tau, dim=-1)

# Loss function (for example)
loss = torch.sum((soft_sample - torch.tensor([0, 1, 0]))**2)

# Backpropagate
loss.backward()

# Check gradients
print(logits.grad)

#%%
#SECTION: reparameterization trick with Gumbel-Softmax

# Logits (raw scores from a neural network, for example)
logits = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)

# Sample Gumbel noise
gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))

# Temperature parameter (lower = closer to one-hot)
tau = 0.5

# Apply Gumbel-Softmax trick
soft_sample = torch.softmax((logits + gumbel_noise) / tau, dim=-1)

# Loss function (for example)
loss = torch.sum((soft_sample - torch.tensor([0, 1, 0]))**2)

# Backpropagate
loss.backward()

# Check gradients
print(f'logits.grad: {logits.grad}, logits: {logits}')

