import os
import sys

# Add the directory containing helper.py to the Python path
# sys.path.append(os.path.abspath("/home/jack/Documents/PhD-research/XAI"))
os.chdir(os.path.abspath("./XAI"))
# Explicitly import the required functions from helper
from helper import *

# Other imports
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torch.distributions as dist
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

from vae_model import *

# |%%--%%| <eDqhzE1998|nTJR7AkbsO>

# SECTION: Import other necessary modules


# Load the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

trainset = MNIST(
    root="~/Documents//data", train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = MNIST(
    root="~/Documents//data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=1000, shuffle=True)


# %%
# SECTION: load MNIST dataa only 8


# Create the dataset for digit 8
testset_8 = MNIST_8(testset)
testloader_8 = DataLoader(testset_8, batch_size=32, shuffle=True)


# Create the dataset for digit 9
testset_9 = MNIST_9(testset)
testloader_9 = DataLoader(testset_9, batch_size=32, shuffle=True)
# |%%--%%| <nTJR7AkbsO|0>


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
"""## Load CNN Weights"""

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# save the mode weights in .pth format (99.25% accuracy
# torch.save(model.state_dict(), 'CNN_MNSIT.pth')

# NOTE: load the model weights


model.load_state_dict(torch.load("./CNN_MNSIT.pth", weights_only=True))
# Set the model to evaluation mode
model.eval()

# Initialize variables to track the number of correct predictions and the total number of samples
correct = 0
total = 0

# Disable gradient calculation for evaluation
with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

# Calculate and print the accuracy
accuracy = 100 * correct / total
print(f"Accuracy of the model on the test images: {accuracy:.2f}%")

# %%
"""## Inital image setup"""

img_id = 1
input = testset_8[img_id]
true_y = input[1]

img = input[0].squeeze(0).clone().to(device)
predicted = model(img.unsqueeze(0).unsqueeze(0)).argmax().item()
prob = F.softmax(model(img.unsqueeze(0).unsqueeze(0)), dim=1)[0][predicted].item()

img_cpu = img.cpu().numpy()
plt.imshow(img_cpu, cmap="gray")
plt.savefig(f"ID {img_id}-Digit {input[1]} original_image.png")
print(f"ID: {img_id}, True y = {input[1]}, probability: {prob:.5f}")
plt.show()
plt.clf()
# |%%--%%| <0|clFqBIZsYc>


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention = self.softmax(torch.bmm(query.unsqueeze(2), key.unsqueeze(1)))
        out = torch.bmm(attention, value.unsqueeze(2)).squeeze(2)
        return out


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.attention_mu = SelfAttention(128)
        self.attention_logvar = SelfAttention(128)
        self.fc_mu = nn.Linear(128, 20)
        self.fc_logvar = nn.Linear(128, 20)

        # Decoder
        self.fc2 = nn.Linear(20, 128)
        self.fc3 = nn.Linear(128, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h2 = h2.view(-1, 64 * 7 * 7)
        h3 = F.relu(self.fc1(h2))
        h3_mu = self.attention_mu(h3)
        h3_logvar = self.attention_logvar(h3)
        return self.fc_mu(h3_mu), self.fc_logvar(h3_logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc2(z))
        h5 = F.relu(self.fc3(h4))
        h5 = h5.view(-1, 64, 7, 7)
        h6 = F.relu(self.deconv1(h5))
        return torch.sigmoid(self.deconv2(h6))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE = ConvVAE().to(device)

# Create a dummy input tensor with the same shape as your MNIST data
dummy_input = torch.randn(1, 1, 28, 28).to(device)
output, mu, logvar = VAE(dummy_input)
print(
    f"Output shape: {output.shape}, mu shape: {mu.shape}, logvar shape: {logvar.shape}"
)

# |%%--%%| <clFqBIZsYc|k3kBTixKj4>

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Define the loss function
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD


def loss_function(recon_x, x, mu, logvar):

    # print(f"recon_x shape: {recon_x.shape}, x shape: {x.shape}")
    # BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    BCE = F.mse_loss(recon_x, x, reduction="sum")
    # print(f"BCE: {BCE.item()}")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(f"KLD: {KLD.item()}")
    return BCE + KLD


# Initialize the model, optimizer, and device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
VAE = ConvVAE().to(device)
optimizer = torch.optim.Adam(VAE.parameters(), lr=1e-3)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    VAE.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = VAE(data)
        loss = loss_function(recon_x, data, mu, logvar)
        loss.backward()
        # train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
    )


# |%%--%%| <k3kBTixKj4|bEUM4lTUyi>


def loss_function(x, x_recon, mu, log_var):
    # NOTE: original loss
    reproduction_loss = F.mse_loss(x_recon, x)

    # NOTE: original KLD
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # KLD = -0.5 * torch.sum((1 + log_var - mu.pow(2) - log_var.exp()) * phi)

    return reproduction_loss + KLD


# Freeze all layers in the cnn model
for param in model.parameters():
    param.requires_grad = False
