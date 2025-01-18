r"""°°°
# Import packages
°°°"""

# |%%--%%| <qFQdkbdE5T|eDqhzE1998>
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

# |%%--%%| <eDqhzE1998|6TyTxpf0QE>
r"""°°°
# Load MNIST data
°°°"""
# |%%--%%| <6TyTxpf0QE|nTJR7AkbsO>

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
# |%%--%%| <nTJR7AkbsO|E85Xc4uKNC>
r"""°°°
# define the CNN model
°°°"""
# |%%--%%| <E85Xc4uKNC|0>


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
# plt.savefig(f"ID {img_id}-Digit {input[1]} original_image.png")
print(f"ID: {img_id}, True y = {input[1]}, probability: {prob:.5f}")
plt.show()
plt.clf()
# |%%--%%| <0|4fkDXCdano>
r"""°°°
# CNN VAE clustering
°°°"""
# |%%--%%| <4fkDXCdano|1jmKeeiiqD>


class Encoder_cluster(nn.Module):
    def __init__(self):
        super(Encoder_cluster, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc_mu = nn.Linear(128, 20)
        self.fc_logvar = nn.Linear(128, 20)

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        h1 = h1.view(-1, 64 * 14 * 14)
        h2 = F.relu(self.fc1(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder_cluster(nn.Module):
    def __init__(self):
        super(Decoder_cluster, self).__init__()
        self.fc2 = nn.Linear(20, 128)
        self.fc3 = nn.Linear(128, 64 * 14 * 14)
        self.deconv1 = nn.ConvTranspose2d(
            64, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def decode(self, z):
        h3 = F.relu(self.fc2(z))
        h4 = F.relu(self.fc3(h3))
        h4 = h4.view(-1, 64, 14, 14)
        return torch.sigmoid(self.deconv1(h4))


class ConvVAE_cluster(nn.Module):
    def __init__(self):
        super(ConvVAE_cluster, self).__init__()
        self.encoder = Encoder_cluster()
        self.decoder = Decoder_cluster()

    def forward(self, x):
        mu, logvar = self.encoder.encode(x)
        z = self.encoder.reparameterize(mu, logvar)
        recon_x = self.decoder.decode(z)
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar):

    BCE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Check for MPS support
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# |%%--%%| <1jmKeeiiqD|9WQIdjZnEO>


# patch_model v1
# class Patch_Model(nn.Module):
#     def __init__(self, input_dim):
#         super(Patch_Model, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, input_dim)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x

# patch_model v2
# class Patch_Model(nn.Module):
#     def __init__(self, input_channels, input_height, input_width):
#         super(Patch_Model, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=1, stride=1)
#         self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
#         self.conv3 = nn.Conv2d(128, input_channels, kernel_size=1, stride=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.conv3(x)
#         x = self.sigmoid(x)
#         return x


# patch_model v3
class Patch_Model(nn.Module):
    def __init__(self, input_channels, input_height, input_width):
        super(Patch_Model, self).__init__()
        # Use kernel_size=3 and padding=1 to maintain input and output dimensions
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, input_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


# |%%--%%| <9WQIdjZnEO|SI0zhI7YHo>

# x = img.clone().flatten()
# patch_model = Patch_Model(x.size(0)).to(device)

x = img.clone()
x = x.unsqueeze(0).unsqueeze(0).to(device)
input_hight = img.shape[0]
input_width = img.shape[1]

patch_model = Patch_Model(1, input_hight, input_width).to(device)
optimizer_patch = torch.optim.Adam(patch_model.parameters(), lr=1e-4)

model.eval()
for i in range(2000):
    w = patch_model(x)
    wx = w * x
    model.zero_grad()
    y0 = model(img.unsqueeze(0).unsqueeze(0))
    y1 = model(wx.reshape(1, 1, 28, 28))
    loss = F.mse_loss(y0, y1) + w.norm(1) + torch.sum(w)
    # loss = -torch.sum(w*torch.log(w+1e-6))
    loss.backward()
    optimizer_patch.step()
    # print(f"Loss: {loss.item()}")

    # Visualize wx image and w points
    if i % 200 == 0:  # Show every 10 iterations
        wx_image = wx.reshape(28, 28).cpu().detach().numpy()
        w_image = w.reshape(28, 28).cpu().detach().numpy()

        plt.imshow(wx_image, cmap="gray")
        plt.title(f"Iteration {i}")

        # Plot w points with different colors
        y_coords, x_coords = np.where(
            w_image > 0
        )  # Get coordinates of positive weights
        colors = w_image[y_coords, x_coords]  # Use weights as colors

        plt.scatter(x_coords, y_coords, c=colors, cmap="viridis", edgecolor="red")
        plt.colorbar()  # Add a color bar to show the weight values
        plt.show()
        plt.clf()

#|%%--%%| <SI0zhI7YHo|sZQNDDuS8n>

input_height = 28
input_width = 28

patch_model = Patch_Model(1, input_height, input_width).to(device)
optimizer_patch = torch.optim.Adam(patch_model.parameters(), lr=1e-4)

model.eval()
for epoch in range(10):  # Number of epochs
    for batch_idx, (data, target) in enumerate(trainloader):
        data = data.to(device)
        batch_size = data.size(0)
        w = patch_model(data)
        wx = w * data
        model.zero_grad()
        y0 = model(data)
        y1 = model(wx)
        loss = F.mse_loss(y0, y1) + w.norm(1) + torch.sum(w)
        loss.backward()
        optimizer_patch.step()

        if batch_idx % 100 == 0:  # Show every 100 batches
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

            # Visualize wx image and w points
            wx_image = wx[0].reshape(28, 28).cpu().detach().numpy()
            w_image = w[0].reshape(28, 28).cpu().detach().numpy()

            plt.imshow(wx_image, cmap="gray")
            plt.title(f"Epoch {epoch}, Batch {batch_idx}")

            # Plot w points with different colors
            y_coords, x_coords = np.where(w_image > 0)  # Get coordinates of positive weights
            colors = w_image[y_coords, x_coords]  # Use weights as colors

            plt.scatter(x_coords, y_coords, c=colors, cmap="viridis", edgecolor="red")
            plt.colorbar()  # Add a color bar to show the weight values
            plt.savefig(f"w_image_iteration_{batch_idx}.png")
            plt.show()
            plt.clf()


# |%%--%%| <sZQNDDuS8n|UojAbttNC4>


# |%%--%%| <UojAbttNC4|QSLOS9OOub>

VAE_cluster = ConvVAE_cluster().to(device)
patch_model = Patch_Model(1, input_hight, input_width).to(device)

optimizer_vae = torch.optim.Adam(VAE_cluster.parameters(), lr=1e-3)
optimizer_patch = torch.optim.Adam(patch_model.parameters(), lr=1e-4)


model.eval()
# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    VAE_cluster.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        batch_size = data.shape[0]
        w = torch.ones(batch_size, 1, img.shape[0], img.shape[1]).to(device)
        data = data.to(device)
        X = data * w

        # step 1 train encoder and decoder
        optimizer_vae.zero_grad()
        recon_x, mu, logvar = VAE_cluster(X)
        loss_vae = loss_function(recon_x, X, mu, logvar)
        loss_vae.backward()
        train_loss += loss_vae.item()
        optimizer_vae.step()

        # step 2 train D and fix VAE_cluster

        for i in range(10):
            VAE_cluster.eval()
            optimizer_patch.zero_grad()
            reconx = recon_x.detach().clone()
            w = patch_model(reconx)
            wx = w * reconx
            model.zero_grad()
            y0 = model(X)
            y1 = model(wx)
            # loss_patch = F.mse_loss(y0, y1) + w.norm(1) + torch.sum(w)
            loss_patch = F.mse_loss(y0, y1) + w.norm(1) 
            loss_patch.backward(retain_graph=True)
            optimizer_patch.step()

        VAE_cluster.train()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss_vae: {loss_vae.item() / len(data):.6f}"
            )
            print(f"Loss_patch: {loss_patch.item():.6f}")

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.4f}"
    )


# |%%--%%| <QSLOS9OOub|5agHvCgelf>


# Function to plot original and reconstructed images
def plot_reconstructed_images(m, data_loader, num_images=10):
    m.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # X = X.view(-1, X_dim).to(device)
            # print(f"X shape: {X.shape}")
            recon_x, mu, logvar = m(data.to(device))
            break

    data = data.cpu().numpy()
    X_sample = recon_x.cpu().numpy()
    # std = torch.exp(0.5 * logvar).reshape(logvar.shape[0], 10, 10)
    # std_interpolate = (
    #     nn.Upsample(size=(28, 28), mode="nearest")(std.unsqueeze(1))
    #     .squeeze(1)
    #     .cpu()
    #     .numpy()
    # )

    fig, axes = plt.subplots(2, num_images, figsize=(num_images, 2))
    for i in range(num_images):
        axes[0, i].imshow(data[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(X_sample[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
        # axes[2, i].imshow(std_interpolate[i], cmap="gray")
        # axes[2, i].axis("off")
        # axes[2, i].set_title("Std Dev")
    plt.show()


plot_reconstructed_images(VAE_cluster, testloader)

# |%%--%%| <5agHvCgelf|clFqBIZsYc>


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
# Check for MPS support
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE = ConvVAE().to(device)

# Create a dummy input tensor with the same shape as your MNIST data
dummy_input = torch.randn(1, 1, 28, 28).to(device)
output, mu, logvar = VAE(dummy_input)
print(
    f"Output shape: {output.shape}, mu shape: {mu.shape}, logvar shape: {logvar.shape}"
)

# |%%--%%| <clFqBIZsYc|k3kBTixKj4>


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
device = "mps"
VAE = ConvVAE().to(device)
optimizer = torch.optim.Adam(VAE.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    VAE.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = VAE(data)
        loss = loss_function(recon_x, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.4f}"
    )


# |%%--%%| <k3kBTixKj4|WAXlfo54qZ>


def plot_reconstructions(recon_x):
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        ax.imshow(recon_x[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


recon_x = recon_x.cpu().detach().numpy()
recon_x = recon_x.reshape(-1, 28, 28)
# Assuming recon_x is a list or array of reconstructed images
# recon_x = generate_reconstructions()  # Your function to generate recon_x
plot_reconstructions(recon_x)

plot_reconstructions(data.reshape(-1, 28, 28).cpu().detach().numpy())

# |%%--%%| <WAXlfo54qZ|VrJ3mG9ZR5>

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN


def plot_laten_valuable(VAE, testloader):
    zs = []
    labels = []

    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            mu, logvar = VAE.encode(data)
            z = VAE.reparameterize(mu, logvar)
            zs.append(z.cpu())
            labels.append(target)

    # Concatenate all zs and labels
    zs = torch.cat(zs).numpy()
    labels = torch.cat(labels).numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(zs)
    # Apply DBSCAN to find clusters
    dbscan = DBSCAN(eps=3, min_samples=5)
    clusters = dbscan.fit_predict(z_tsne)

    # Count the number of labels not within the main groups (noise points)
    num_noise_points = (clusters == -1).sum()

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f"t-SNE of Latent Space z (Noise points: {num_noise_points})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()


plot_laten_valuable(VAE, testloader)

# |%%--%%| <VrJ3mG9ZR5|QZZH4m03o5>


class ConvVAE_base(nn.Module):
    def __init__(self):
        super(ConvVAE_base, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc_mu = nn.Linear(128, 20)
        self.fc_logvar = nn.Linear(128, 20)

        # Decoder
        self.fc2 = nn.Linear(20, 128)
        self.fc3 = nn.Linear(128, 64 * 14 * 14)
        self.deconv1 = nn.ConvTranspose2d(
            64, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        h1 = h1.view(-1, 64 * 14 * 14)
        h2 = F.relu(self.fc1(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc2(z))
        h4 = F.relu(self.fc3(h3))
        h4 = h4.view(-1, 64, 14, 14)
        return torch.sigmoid(self.deconv1(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):

    BCE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Check for MPS support
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example usage
print(f"Using device: {device}")

VAE_base = ConvVAE_base().to(device)

optimizer = torch.optim.Adam(VAE_base.parameters(), lr=1e-3)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    VAE_base.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = VAE_base(data)
        loss = loss_function(recon_x, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.4f}"
    )

# |%%--%%| <QZZH4m03o5|2oWMB1XdPZ>

plot_reconstructions(recon_x.reshape(-1, 28, 28).cpu().detach().numpy())

plot_reconstructions(data.reshape(-1, 28, 28).cpu().detach().numpy())


# |%%--%%| <2oWMB1XdPZ|BJpHLcqfJI>

plot_laten_valuable(VAE_base, testloader)
# |%%--%%| <BJpHLcqfJI|j2HK62KiHr>

std = torch.exp(0.5 * logvar)


def plot_mu_std(VAE, testloader):
    # Collect all mu and std
    mus = []
    stds = []

    with torch.no_grad():
        for data, _ in testloader:
            data = data.to(device)
            mu, logvar = VAE.encode(data)
            std = torch.exp(0.5 * logvar)
            mus.append(mu.cpu())
            stds.append(std.cpu())

    # Concatenate all mus and stds
    mus = torch.cat(mus).numpy()
    stds = torch.cat(stds).numpy()

    # Plot mu
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(mus.flatten(), bins=50, color="blue", alpha=0.7)
    plt.title("Histogram of mu")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Plot std
    plt.subplot(1, 2, 2)
    plt.hist(stds.flatten(), bins=50, color="red", alpha=0.7)
    plt.title("Histogram of std")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


plot_mu_std(VAE_base, testloader)

# |%%--%%| <j2HK62KiHr|bEUM4lTUyi>


# Freeze all layers in the cnn model
for param in model.parameters():
    param.requires_grad = False
