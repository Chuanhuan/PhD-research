import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import imageio
import os

# Hyperparameters
batch_size = 100
latent_dim = 20
epochs = 5
num_classes = 10
img_dim = 28
filters = 16
intermediate_dim = 256

# Dataset preparation (MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size=3, stride=2, padding=1)
        self.fc_mean = nn.Linear(filters * 2 * 7 * 7, latent_dim)
        self.fc_log_var = nn.Linear(filters * 2 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(x.size(0), -1)  # Flatten
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, filters * 2 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(
            filters * 2,
            filters * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.ConvTranspose2d(
            filters * 2, filters, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv3 = nn.ConvTranspose2d(
            filters, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(z.size(0), filters * 2, 7, 7)  # Reshape
        z = F.leaky_relu(self.conv1(z), 0.2)
        z = F.leaky_relu(self.conv2(z), 0.2)
        z = torch.sigmoid(self.conv3(z))
        return z


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, num_classes)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        y = F.softmax(self.fc2(z), dim=-1)
        return y


class Gaussian(nn.Module):
    """Simple layer that defines the mean for q(z|y) with one mean per class.
    Outputs z - mean to help compute loss later."""

    def __init__(self, num_classes, latent_dim):
        super(Gaussian, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        # Create a learnable tensor for the means
        self.mean = nn.Parameter(torch.zeros(num_classes, latent_dim))

    def forward(self, z):
        # z.shape = (batch_size, latent_dim)
        # Expand z and the mean to enable broadcasting
        z = z.unsqueeze(1)  # shape: (batch_size, 1, latent_dim)
        mean_expanded = self.mean.unsqueeze(0)  # shape: (1, num_classes, latent_dim)
        return z - mean_expanded  # shape: (batch_size, num_classes, latent_dim)


# Instantiate the Gaussian layer
gaussian = Gaussian(num_classes, latent_dim)


# Modify the VAE model forward pass to include this layer:
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()
        self.gaussian = Gaussian(num_classes, latent_dim)

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(z_log_var / 2) * epsilon

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decoder(z)
        y = self.classifier(z)
        z_prior_mean = self.gaussian(z)  # Apply the Gaussian layer
        return x_recon, z_mean, z_log_var, y, z_prior_mean


# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#         self.classifier = Classifier()
#
#     def reparameterize(self, z_mean, z_log_var):
#         epsilon = torch.randn_like(z_mean)
#         return z_mean + torch.exp(z_log_var / 2) * epsilon
#
#     def forward(self, x):
#         z_mean, z_log_var = self.encoder(x)
#         z = self.reparameterize(z_mean, z_log_var)
#         x_recon = self.decoder(z)
#         y = self.classifier(z)
#         return x_recon, z_mean, z_log_var, y


def vae_loss(x, x_recon, z_mean, z_log_var, z_prior_mean, y, lamb=2.5):
    xent_loss = 0.5 * torch.mean((x - x_recon) ** 2)
    kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    prior_loss = torch.mean(z_prior_mean.pow(2))  # Loss involving z - mean
    cat_loss = torch.mean(y * torch.log(y + 1e-8))
    return lamb * xent_loss + kl_loss + cat_loss + prior_loss


# # Loss functions
# def vae_loss(x, x_recon, z_mean, z_log_var, y, lamb=2.5):
#     xent_loss = 0.5 * torch.mean((x - x_recon) ** 2)
#     kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
#     cat_loss = torch.mean(y * torch.log(y + 1e-8))
#     return lamb * xent_loss + kl_loss + cat_loss


# Instantiate model, optimizer
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data
        optimizer.zero_grad()
        # x_recon, z_mean, z_log_var, y = model(data)
        # loss = vae_loss(data, x_recon, z_mean, z_log_var, y)
        x_recon, z_mean, z_log_var, y, z_prior_mean = model(data)
        loss = vae_loss(data, x_recon, z_mean, z_log_var, z_prior_mean, y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader)}")


# FIXME: This code is not working
# Sample generation and cluster sampling (similar to cluster_sample and random_sample functions)
# def cluster_sample(path, category=0):
#     n = 8
#     figure = np.zeros((img_dim * n, img_dim * n))
#     model.eval()
#     with torch.no_grad():
# # Assume train_data is your training data
#         train_data = torch.cat([data for data, _ in train_loader])
# # Pass the training data through the model's classifier to get the predicted labels
#         y_train_pred = model.classifier(model.encoder(train_data)[0]).argmax(dim=1)
#         idxs = np.where(y_train_pred == category)[0]
#         for i in range(n):
#             for j in range(n):
#                 digit = train_dataset[idxs[np.random.choice(len(idxs))]][0]
#                 digit = digit.squeeze().cpu().numpy()
#                 figure[
#                     i * img_dim : (i + 1) * img_dim, j * img_dim : (j + 1) * img_dim
#                 ] = digit
#         imageio.imwrite(path, figure * 255)
#
#
# def random_sample(path, category=0, std=1):
#     n = 8
#     figure = np.zeros((img_dim * n, img_dim * n))
#     model.eval()
#     with torch.no_grad():
#         for i in range(n):
#             for j in range(n):
#                 z_sample = torch.randn(1, latent_dim) * std + torch.tensor(
#                     means[category]
#                 )
#                 x_recon = model.decoder(z_sample).cpu().numpy()
#                 digit = x_recon[0].reshape((img_dim, img_dim))
#                 figure[
#                     i * img_dim : (i + 1) * img_dim, j * img_dim : (j + 1) * img_dim
#                 ] = digit
#         imageio.imwrite(path, figure * 255)
#
#
# if not os.path.exists("samples"):
#     os.mkdir("samples")
#
# for i in range(10):
#     cluster_sample(f"samples/cluster_category_{i}.png", i)
#     random_sample(f"samples/sample_category_{i}.png", i)

