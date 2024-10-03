# %%
# NOTE: Define a CNN model for MNIST dataset and load the model weights

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.distributions as dist
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output


model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

trainset = MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = MNIST(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1000, shuffle=True)


# %%
# save the mode weights in .pth format (99.25% accuracy
# torch.save(model.state_dict(), 'CNN_MNSIT.pth')

# NOTE: load the model weights
model.load_state_dict(torch.load("CNN_MNSIT.pth"))

# %%
img_id = 3
input = testset[img_id]
img = input[0].squeeze(0).clone()
img = transform(img)
plt.imshow(img, cmap="gray")
plt.savefig("original_image.png")
print(
    f"True y = {input[1]}, the probability: {F.softmax(model(input[0].unsqueeze(0)), dim=1).max():.5f}"
)
print(f"pixel from {img.max()} to {img.min()}")

# %%
# NOTE: The model in XAI saliency map evaluation
# Wrap the image in a Variable, set requires_grad=True to compute gradients
image = torch.autograd.Variable(
    img.clone().unsqueeze(0).unsqueeze(0), requires_grad=True
)

# Forward pass
outputs = model(image)

# Get the index of the max log-probability
_, predicted = outputs.max(1)

# Zero the gradients of the model parameters
model.zero_grad()

# Backward pass
outputs[0, predicted].backward()
# The saliency map is in the grad of the image
saliency_map = image.grad.data.abs().max(dim=1)[0]


# Convert the saliency map from Torch Tensor to numpy array and display it
saliency_map = saliency_map.numpy()
plt.imshow(saliency_map[0], cmap=plt.cm.hot)
# New code to add a colorbar
plt.colorbar()
plt.savefig("saliency_map.png")
plt.clf()
print(f"max pixel value: {saliency_map.max()}, min pixel value: {saliency_map.min()}")


# %%
# NOTE: The model in VI evaluation
# TODO: ADD ELBO in the model


def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * (y - mu) ** 2


class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 784),
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 784),
        )

    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var


# %%
def elbo(y_pred, y, mu, log_var, model=model, predicted=predicted):
    # HACK: use the CNN model predition as the input
    model.eval()
    input = y_pred.view(1, 1, 28, 28)
    # Forward pass
    outputs = model(input)

    log_p_y = torch.log(F.softmax(outputs, dim=1)[:, predicted])
    sigma = log_var.exp() ** 0.5
    # likelihood of observing y given Variational mu and sigma
    likelihood = dist.Normal(mu, sigma).log_prob(y)

    # prior probability of y_pred
    log_prior = dist.Normal(0, 1).log_prob(y_pred)

    # variational probability of y_pred
    log_p_q = dist.Normal(mu, sigma).log_prob(y_pred)

    # by taking the mean we approximate the expectation
    return (log_p_y + likelihood + log_prior - log_p_q).mean()


def det_loss(y_pred, y, mu, log_var, model=model, predicted=predicted):
    return -elbo(y_pred, y, mu, log_var, model, predicted)


# %%
# NOTE: Train the VI model
# HACK: if not specify "predicted" digits, than will attack to certain become another value
epochs = 5000

m = VI()
optim = torch.optim.Adam(m.parameters(), lr=0.005)


for epoch in range(epochs):
    X = img.view(784).clone()
    Y = img.view(784).clone()
    optim.zero_grad()
    y_pred, mu, log_var = m(X)
    # print(f"y_pred shape: {y_pred.shape}")
    # print(f"mu shape: {mu.shape}, log_var shape: {log_var.shape}")
    # Get the index of the max log-probability
    _, predicted = outputs.max(1)
    loss = det_loss(y_pred, Y, mu, log_var, model, predicted)
    print(f"loss: {loss}")
    loss.backward()
    optim.step()

# %%

with torch.no_grad():
    X = img.view(784).clone()
    Y = img.view(784).clone()
    y_pred, mu, log_var = m(X)
    print(torch.abs(y_pred - Y).mean())

new_image = m(X)[0].view(1, 1, 28, 28)
F.softmax(model(new_image), dim=1).max()
print(
    f"True y = {input[1]}, the probability: {F.softmax(model(new_image), dim=1).max():.5f}"
)
print(f"model prediction: {model(new_image).max()}")
plt.imshow(new_image.squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
plt.colorbar()
plt.savefig("new_image.png")
plt.clf()

# %%
# NOTE: The model in VI evaluation
print(f"variance: {log_var.exp().max()}")
highest_var = log_var.exp().max()
k = 0.5
high_var_index = np.where(log_var.view(28, 28).exp() > highest_var * k)
plt.imshow(X.view(28, 28).detach().numpy())
# plt.scatter(high_var_index[1], high_var_index[0], s=10, c="red")
# Assume log_var is a tensor and you compute its exponential
exp_values = log_var.view(28, 28).exp()

# Flatten the tensor to 1D for scatter plot
exp_values_flatten = exp_values[high_var_index[0], high_var_index[1]]


# Scatter plot with colors corresponding to exp_values
plt.scatter(
    high_var_index[1], high_var_index[0], s=10, c=exp_values_flatten, cmap="viridis"
)

# Add a colorbar to show the mapping from colors to values
plt.colorbar(label="exp(log_var)")
plt.savefig("high_var_index.png")
plt.clf()
