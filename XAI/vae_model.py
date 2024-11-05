# SECTION: Define a CNN model for MNIST dataset and load the model weights

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
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
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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


# SECTION: Model definition


class CustomTanh(nn.Module):
    def __init__(self, min_val, max_val):
        super(CustomTanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return (torch.tanh(x) + 1) * (self.max_val - self.min_val) / 2 + self.min_val


class Generator(nn.Module):
    def __init__(self, channels_z, channels_img, min_val=-1, max_val=1):
        super().__init__()
        self.channels_z = channels_z
        self.channels_img = channels_img
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.channels_z, self.channels_img, kernel_size=2, stride=2, padding=0
            ),
            CustomTanh(min_val, max_val),
        )

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon


class Learner(nn.Module):
    def __init__(self, channels_img, k=10, min_val=-1, max_val=1):
        super().__init__()
        self.channels_img = channels_img
        self.k = k
        # encoder
        self.mean_layer = nn.Sequential(
            nn.Conv2d(channels_img, self.k, kernel_size=2, stride=2),
            nn.InstanceNorm2d(self.k, affine=True),
            CustomTanh(min_val, max_val),
            # nn.LeakyReLU(0.2),
        )
        self.logvar_layer = nn.Sequential(
            nn.Conv2d(channels_img, self.k, kernel_size=2, stride=2),
            nn.InstanceNorm2d(self.k, affine=True),
            nn.Tanh(),
        )
        self.c_layer = nn.Sequential(
            nn.Conv2d(channels_img, self.k, kernel_size=2, stride=2),
            nn.InstanceNorm2d(self.k, affine=True),
            nn.Softmax(dim=1),
        )

    def reparameterization(self, mu, log_var, phi):
        epsilon = torch.randn_like(phi)
        sigma = torch.exp(0.5 * log_var) + 1e-5
        z = mu + sigma * epsilon

        # HACK: log z phi

        min_z = z.min()
        z = z - min_z + 1
        log_z_phi = torch.log(z) * phi
        z = log_z_phi.exp()
        z = z - z.min()

        return z

    def forward(self, x):
        mu = self.mean_layer(x)
        log_var = self.logvar_layer(x)
        phi = self.c_layer(x)
        z = self.reparameterization(mu, log_var, phi)
        return z, mu, log_var, phi


if __name__ == "__main__":
    # Load the MNIST dataset
    min_val = img.min()
    max_val = img.max()
    # Adjust the number of channels to match between encoder and decoder
    channels_img = 1
    latent_dim = 10

    G = Generator(latent_dim, channels_img).to(device)
    L = Learner(channels_img, latent_dim).to(device)

    x = torch.randn(1, 1, 28, 28).to(device)
    z, mu, log_var, phi = L(x)
    x_recon = G(z)
    print(f"mu:{mu.shape}, log_var:{log_var.shape}, x_recon:{x_recon.shape}")
