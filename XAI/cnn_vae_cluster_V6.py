# SECTION: Define a CNN model for MNIST dataset and load the model weights

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

# Get the directory of the current file
# current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the file's directory
# os.chdir(current_file_directory)

# |%%--%%| <A0HJs55ICL|aOudbR11fI>

# SECTION: Import other necessary modules

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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


# |%%--%%| <aOudbR11fI|AscxxvLuFy>

"""## Load CNN Weights"""

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


# |%%--%%| <AscxxvLuFy|sxJgHvqiBP>


def loss_function(x, mu, log_var, phi, x_recon):
    phi = phi + 1e-10
    t1 = -0.5 * (log_var.exp() + mu**2)
    t1 = t1.sum()

    # NOTE: origine
    # x_flat = x.view(-1)
    # mu_flat = mu.view(-1)
    # t2 = torch.outer(x_flat, mu_flat) - 0.5 * x_flat.view(-1, 1) ** 2
    # t2 = -0.5 * (log_var.exp() + mu**2).view(1, -1) + t2
    # t2 = phi.view(1, -1) * t2
    # t2 = torch.sum(t2)

    # FIXME: this is not correct, but why?
    t2 = (x - x_recon) ** 2
    t2 = -torch.mean(t2)

    # NOTE:Basics
    t3 = phi * torch.log(phi)
    t3 = -torch.sum(t3)

    t4 = 0.5 * log_var.sum()
    # print(f't1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}')
    return -(t1 + t2 + t3 + t4)


def loss_function_2(x, x_recon, mu, log_var):
    # NOTE: original loss
    reproduction_loss = F.mse_loss(x_recon, x)

    # NOTE: original KLD
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # KLD = -0.5 * torch.sum((1 + log_var - mu.pow(2) - log_var.exp()) * phi)

    return reproduction_loss + KLD


# |%%--%%| <sxJgHvqiBP|A9EWgSfwGj>


# %%
# SECTION: Training

min_val = img.min()
max_val = img.max()

torch.autograd.set_detect_anomaly(True)
epochs = 1000
leaner_epochs = 10
predicted = true_y
channels_img = 1
latent_dim = 10

G = Generator(latent_dim, channels_img, min_val, max_val).to(device)
L = Learner(channels_img, latent_dim, min_val, max_val).to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=0.005)
opt_L = torch.optim.Adam(L.parameters(), lr=0.005)

for epoch in range(epochs + 1):
    for leaner_epoch in range(leaner_epochs + 1):
        x = img.clone().to(device)
        x = x.view(1, 1, 28, 28)
        z, mu, log_var, phi = L(x)
        x_recon = G(z)

        # train learner Get the index of the max log-probability
        model.eval()
        opt_L.zero_grad()
        critic_fake = F.softmax(model(x_recon), dim=1)[0][predicted]
        loss_L = -(torch.mean(critic_fake))
        loss_L.backward()
        opt_L.step()
    z, mu, log_var, phi = L(x)
    x_recon = G(z)

    # train generator
    model.eval()
    opt_G.zero_grad()
    with torch.no_grad():
        critic_fake = F.softmax(model(x_recon), dim=1)[0][predicted]
    t1 = -torch.sum(torch.log(critic_fake + 1e-5))
    # t2 = loss_function(x, mu, log_var, phi, x_recon)
    t2 = loss_function_2(x, x_recon, mu * phi, log_var * phi)
    loss_G = t1 + t2
    loss_G.backward(retain_graph=True)  # Retain graph for t3
    opt_G.step()

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss_L: {loss_L}, loss_G: {loss_G}")


# |%%--%%| <A9EWgSfwGj|EWF7msubuw>


# |%%--%%| <EWF7msubuw|fFhjaEm19i>
# %%


# %%


# %%


# %%
print(f"x.max(): {x.max()}, x.min(): {x.min()}")
print(f"x_recon.max(): {x_recon.max()}, x_recon.min(): {x_recon.min()}")
print(f"mu.max(): {mu.max()}, mu.min(): {mu.min()}")
print(f"log_var.max(): {log_var.max()}, log_var.min(): {log_var.min()}")
print(f"prob: {F.softmax(model(x_recon.view(1, 1, 28, 28)), dim=1)}")
print(f"phi: {phi}")

sums = []

# Iterate over the channels
for i in range(phi.shape[1]):
    # Compute the sum of elements greater than 0.5
    sum_greater_than_0_5 = torch.sum(phi[0, i, :, :] > 0.5).item()
    print(f"i:{i}, {sum_greater_than_0_5}")
    sums.append(sum_greater_than_0_5)

# Find the index of the maximum sum
argmax_i = torch.argmax(torch.tensor(sums)).item()

print(f"The index i with the maximum sum of elements > 0.5 is: {argmax_i}")

# %%
# SECTION: plot the reconstructed image

plot_recon_img(x_recon, model, true_y, img_id)

# %%
# SECTION: find the n_th patch of image
p_interpolate = phi[:, argmax_i, :, :].unsqueeze(0)  # Add batch dimension
p_interpolate = nn.Upsample(size=(28, 28), mode="nearest")(
    p_interpolate
)  # Apply upsampling
plot_patch_image(img, model, true_y, img_id, p_interpolate, device)


# %%
# SECTION: choice variance  and plot the reconstructed image


log_var_interpolated = F.interpolate(log_var.exp(), size=(28, 28), mode="nearest")

# Convert tensors to numpy arrays for plotting
log_var_np = log_var_interpolated.squeeze().detach().cpu().numpy()
original_image_np = img.squeeze().detach().cpu().numpy()

# Plotting
fig, axes = plt.subplots(5, 2, figsize=(10, 20))

for i in range(10):
    ax = axes[i // 2, i % 2]
    ax.imshow(original_image_np, cmap="gray", alpha=0.5)
    ax.imshow(log_var_np[i], cmap="hot", alpha=0.5)
    ax.set_title(f"Channel {i+1}")
    plt.colorbar(ax.imshow(log_var_np[i], cmap="hot", alpha=0.5), ax=ax)
    ax.axis("off")

plt.tight_layout()
plt.savefig("log_var_interpolated.png")
plt.clf()

# %%
# SECTION: train testloader

# torch.autograd.set_detect_anomaly(True)
epochs = 1000
leaner_epochs = 10
predicted = true_y
channels_img = 1
latent_dim = 10

G = Generator(latent_dim, channels_img).to(device)
L = Learner(channels_img, latent_dim).to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=0.005)
opt_L = torch.optim.Adam(L.parameters(), lr=0.005)

for batch_idx, (data, target) in enumerate(testloader):
    data = data.to(device)
    target = target.to(device)
    for epoch in range(epochs + 1):
        for leaner_epoch in range(leaner_epochs + 1):
            opt_L.zero_grad()
            x = data.clone().to(device)
            predicted = target

            z, mu, log_var, phi = L(x)
            x_recon = G(z)

            # train generator
            model.eval()
            opt_G.zero_grad()
            # critic_fake = F.softmax(model(x_recon), dim=1)[0][predicted]
            outputs = F.softmax(model(x), dim=1)
            predicted = torch.argmax(outputs, dim=1)
            critic_fake = outputs[torch.arange(len(predicted)), predicted]

            t1 = -torch.sum(torch.log(critic_fake + 1e-5))
            t2 = loss_function(x, mu, log_var, phi, x_recon)
            loss_G = t1 + t2
            loss_G.backward(retain_graph=True)  # Retain graph for t3
            opt_G.step()

        z, mu, log_var, phi = L(x)
        x_recon = G(z)
        # train learner Get the index of the max log-probability
        model.eval()
        opt_L.zero_grad()
        outputs = F.softmax(model(x), dim=1)
        critic_fake = outputs[torch.arange(len(predicted)), predicted]
        loss_L = -(torch.mean(critic_fake))
        loss_L.backward()
        opt_L.step()
        # print(f'mu.shape: {mu.shape}, log_var.shape: {log_var.shape}, phi.shape: {phi.shape}')
        if epoch % 500 == 0:
            print(f"epoch: {epoch}, loss_L: {loss_L}, loss_G: {loss_G}")


# %%
# SECTION: choice n random phto and plot the reconstructed image


plot_and_save_random_reconstructions_and_patches(
    testloader.dataset, G, L, n=5, device=device
)

# %%
# SECTION: train testloader2


def loss_function(x, mu, log_var, phi, x_recon):
    phi = phi + 1e-10
    t1 = -0.5 * (log_var.exp() + mu**2)
    t1 = t1.sum()

    # FIXME: alternative t2: swap leaner and generator
    t2 = (x - x_recon) ** 2
    t2 = -torch.mean(t2)

    # NOTE:Basics
    t3 = phi * torch.log(phi)
    t3 = -torch.sum(t3)

    t4 = 0.5 * log_var.sum()
    # print(f't1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}')
    return -(t1 + t2 + t3 + t4)


# torch.autograd.set_detect_anomaly(True)
epochs = 1000
leaner_epochs = 10
predicted = true_y
channels_img = 1
latent_dim = 10

G = Generator(latent_dim, channels_img, min_val, max_val).to(device)
L = Learner(channels_img, latent_dim, min_val, max_val).to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=0.005)
opt_L = torch.optim.Adam(L.parameters(), lr=0.005)

for batch_idx, (data, target) in enumerate(testloader):
    data = data.to(device)
    target = target.to(device)
    for epoch in range(epochs + 1):
        for leaner_epoch in range(leaner_epochs + 1):
            opt_L.zero_grad()
            x = data.clone().to(device)
            predicted = target

            z, mu, log_var, phi = L(x)
            x_recon = G(z)

            # train learner Get the index of the max log-probability
            model.eval()
            opt_L.zero_grad()
            outputs = F.softmax(model(x), dim=1)
            critic_fake = outputs[torch.arange(len(predicted)), predicted]
            loss_L = -(torch.mean(critic_fake))
            loss_L.backward()
            opt_L.step()

        z_0 = z
        z, mu, log_var, phi = L(x)
        x_recon = G(z)

        # train generator
        model.eval()
        opt_G.zero_grad()
        # critic_fake = F.softmax(model(x_recon), dim=1)[0][predicted]
        outputs = F.softmax(model(x), dim=1)
        predicted = torch.argmax(outputs, dim=1)
        critic_fake = outputs[torch.arange(len(predicted)), predicted]

        t1 = -torch.sum(torch.log(critic_fake + 1e-5))
        t2 = loss_function(x, mu, log_var, phi, x_recon)
        t3 = (z_0 - z) ** 2
        t3 = torch.mean(t3)
        loss_G = t1 + t2 + t3
        loss_G.backward(retain_graph=True)  # Retain graph for t3
        opt_G.step()
        # print(f'mu.shape: {mu.shape}, log_var.shape: {log_var.shape}, phi.shape: {phi.shape}')
        if epoch % 500 == 0:
            print(f"epoch: {epoch}, loss_L: {loss_L}, loss_G: {loss_G}")


# %%
# SECTION: choice n random phto and plot the reconstructed image


plot_and_save_random_reconstructions_and_patches(
    testloader.dataset, G, L, n=5, device=device
)

# %%


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap


# x = img.clone().to(device)
# x = x.view(1, 1, 28, 28)
z, mu, log_var, phi = L(x)
# Assuming phi is a tensor of shape (1000, 10, 14, 14)
phi_np = phi.cpu().detach().numpy()  # Convert to numpy array if it's a tensor
batch_size, channels, height, width = phi_np.shape

# Reshape phi to (batch_size, channels * height * width)
phi_flat = phi_np.reshape(batch_size, -1)

# Assuming phi is a tensor of shape (1000, 10, 14, 14)
phi_np = phi.cpu().detach().numpy()  # Convert to numpy array if it's a tensor
batch_size, channels, height, width = phi_np.shape

# Repeat the single sample to create a batch of size 1000
phi_np_repeated = np.repeat(phi_np, 1000, axis=0)

# Reshape phi to (batch_size, channels * height * width)
phi_flat = phi_np_repeated.reshape(1000, -1)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
phi_tsne = tsne.fit_transform(phi_flat)

# Apply UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
phi_umap = umap_reducer.fit_transform(phi_flat)

# Create a figure with subplots for t-SNE and UMAP
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Define a color map
cmap = plt.get_cmap("tab10")

# Plot t-SNE results
for i in range(10):
    labels = phi_np[:, i, :, :].reshape(batch_size, -1).mean(axis=1)
    mask = phi_np[:, i, :, :].reshape(batch_size, -1).mean(axis=1) > 0.05
    colors = np.where(mask, "red", "blue")
    axes[0].scatter(
        phi_tsne[:, 0], phi_tsne[:, 1], c=colors, label=f"Dimension {i}", alpha=0.6
    )
axes[0].set_title("t-SNE of phi")
axes[0].set_xlabel("Component 1")
axes[0].set_ylabel("Component 2")
axes[0].legend()

# Plot UMAP results
for i in range(10):
    labels = phi_np[:, i, :, :].reshape(batch_size, -1).mean(axis=1)
    mask = phi_np[:, i, :, :].reshape(batch_size, -1).mean(axis=1) > 0.5
    colors = np.where(mask, "red", "blue")
    axes[1].scatter(
        phi_umap[:, 0], phi_umap[:, 1], c=colors, label=f"Dimension {i}", alpha=0.6
    )
axes[1].set_title("UMAP of phi")
axes[1].set_xlabel("Component 1")
axes[1].set_ylabel("Component 2")
axes[1].legend()

plt.tight_layout()
plt.savefig("t-SNE_vs_UMAP_per_dimension.png")
plt.show()
("UMAP of phi")
axes[1].set_xlabel("Component 1")
axes[1].set_ylabel("Component 2")
axes[1].legend()

plt.tight_layout()
plt.savefig("t-SNE_vs_UMAP_per_dimension.png")
plt.show()


# %%


# Create a figure with subplots for t-SNE and UMAP
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Define a color map
cmap = plt.get_cmap("tab10")

# Plot t-SNE results
# for i in range(channels):
for i in range(3):
    labels = phi_np[0, i, :, :].reshape(-1)
    num_values_above_threshold = np.sum(labels > 0.5)
    axes[0].scatter(
        phi_tsne[:, 0],
        phi_tsne[:, 1],
        c=cmap(i),
        label=f"Dimension {i} (>0.5: {num_values_above_threshold})",
        alpha=0.6,
    )
axes[0].set_title("t-SNE of phi")
axes[0].set_xlabel("Component 1")
axes[0].set_ylabel("Component 2")
axes[0].legend()

# Plot UMAP results
# for i in range(channels):
for i in range(5):
    labels = phi_np[0, i, :, :].reshape(-1)
    num_values_above_threshold = np.sum(labels > 0.5)
    axes[1].scatter(
        phi_umap[:, 0],
        phi_umap[:, 1],
        c=cmap(i),
        label=f"Dimension {i} (>0.5: {num_values_above_threshold})",
        alpha=0.6,
    )
axes[1].set_title("UMAP of phi")
axes[1].set_xlabel("Component 1")
axes[1].set_ylabel("Component 2")
axes[1].legend()

plt.tight_layout()
plt.savefig("t-SNE_vs_UMAP_per_dimension.png")
plt.clf()
