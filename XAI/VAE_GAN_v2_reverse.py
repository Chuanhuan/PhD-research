import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import os


# |%%--%%| <0KiVc3DKq9|45FbLi3jh9>
r"""°°°
CNN model
°°°"""
#|%%--%%| <45FbLi3jh9|vdkIA87H1I>


os.chdir(os.path.abspath("/home/jack/Documents/PhD-research/XAI"))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Define transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # MNIST is grayscale
)

# Load datasets
train_dataset = torchvision.datasets.MNIST(
    root="~/Documents/data", train=True, download=True, transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="~/Documents/data", train=False, download=True, transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# |%%--%%| <vdkIA87H1I|4cYhBtzvcP>



# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


# Initialize model
model = CNN().to(device)

model.load_state_dict(torch.load("CNN_MNIST_-1_1.ckpt"))

# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # Training loop
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 100 == 0:
#             print(
#                 f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
#             )

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
# |%%--%%| <4cYhBtzvcP|QpPrzqZjIJ>
r"""°°°
Define parameters for the model
°°°"""
# |%%--%%| <QpPrzqZjIJ|k5YWyyqgo7>


class Config:
    batch_size = 100
    latent_dim = 20
    epochs = 10
    num_classes = 10
    img_dim = 28
    output_dim = 28 * 28
    initial_filters = 16
    intermediate_dim = 256
    lamb = 2.5  # 重构损失权重
    sample_std = 0.5  # 采样标准差
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = 1.0
    anneal_rate = 0.001


# |%%--%%| <k5YWyyqgo7|lMC32SbEnu>
r"""°°°
Define VAE model
°°°"""
# |%%--%%| <lMC32SbEnu|GLMDo7DbS2>



class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        # Convolutional layers to downsample the 28x28 image
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: (batch, 32, 14, 14)
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (batch, 64, 7, 7)
            nn.LeakyReLU(),
        )
        # Flatten the feature map
        self.flatten = nn.Flatten()  # Output: (batch, 64*7*7)
        # Fully connected layers to produce mu and logvar
        self.fc_mu = nn.Linear(64 * 7 * 7, z_dim)      # Output: (batch, z_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, z_dim)  # Output: (batch, z_dim)

    def forward(self, x):
        # Input shape: (batch, 1, 28, 28)
        x = self.conv(x)        # Shape: (batch, 64, 7, 7)
        x = self.flatten(x)     # Shape: (batch, 64*7*7)
        mu = self.fc_mu(x)      # Shape: (batch, z_dim)
        logvar = self.fc_logvar(x)  # Shape: (batch, z_dim)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        # Fully connected layer to map z_dim to a feature map size
        self.fc = nn.Linear(z_dim, 64 * 7 * 7)  # Output: (batch, 64*7*7)
        # Transposed convolutional layers to upsample to 28x28
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (batch, 32, 14, 14)
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # Output: (batch, 1, 28, 28)
            nn.Sigmoid()  # Output range [0,1], suitable for normalized images like MNIST
        )

    def forward(self, z):
        # Input shape: (batch, z_dim)
        x = self.fc(z)              # Shape: (batch, 64*7*7)
        x = x.view(-1, 64, 7, 7)    # Shape: (batch, 64, 7, 7)
        x = self.deconv(x)          # Shape: (batch, 1, 28, 28)
        return x

class ClusterVAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        return {
            "recon": self.decoder(z),
            "z": z,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
        }


# |%%--%%| <GLMDo7DbS2|zRr2XpaDJo>
r"""°°°
define Critic model
°°°"""
# |%%--%%| <zRr2XpaDJo|bKaOFAsJO1>


# Critic
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.main(x)
        x = x.view(-1, 1, 28, 28)
        return x


# |%%--%%| <bKaOFAsJO1|eqx0LfDGeH>
r"""°°°
loss function
°°°"""
# |%%--%%| <eqx0LfDGeH|83f6l2Dw3G>


def loss_fn_vae(recon_x,  z_mean, z_logvar, data, model):
    recon_loss = F.mse_loss(recon_x, data)

    model.eval()  # Ensure model is in eval mode (affects dropout, batchnorm, etc.)
    with torch.no_grad():  # Prevent gradients from flowing through the main model
        pred = model(data)

    gaussian_loss = -0.5 * torch.mean(1 + z_logvar - ( z_mean - pred)**2 - z_logvar.exp())

    total_loss =  Config.lamb *recon_loss + gaussian_loss
    return total_loss


def loss_fn_critic( recon_x, z_mean,c, data, model):

    recon_loss = F.mse_loss(recon_x *c , data *c)
    z_loss = F.mse_loss(z_mean, model(data))
    cat_sum = torch.sum(c, dim=(1, 2, 3)).mean()
    

    total_loss = recon_loss * Config.lamb * z_loss

    # total_loss = recon_loss +z_loss + cat_sum

    return total_loss


# |%%--%%| <83f6l2Dw3G|iRScmHTlMW>
r"""°°°
Training
°°°"""
# |%%--%%| <iRScmHTlMW|tQqyQelokQ>

c_vae = ClusterVAE(Config.num_classes).to(Config.device)
critic = Critic(Config.img_dim * Config.img_dim).to(Config.device)

vae_optimizer = optim.Adam(c_vae.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

model.eval()
model.requires_grad_(False)  # Freeze main model parameters

# c = torch.ones(batch_size, 1, Config.img_dim, Config.img_dim).to(Config.device)

for epoch in range(Config.epochs):
    total_loss_vae = 0
    total_loss_critic = 0
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")
    acc_list = []

    for data, labels in pbar:
        data = data.to(Config.device)
        labels = labels.to(Config.device)

        # train the VAE
        c_vae.train()
        critic.eval()

        vae_optimizer.zero_grad()

        with torch.no_grad():
            x = model(data)
            _, pred_base = x.max(1)

        output = c_vae(data)
        # output['z_mean'] = gumbel_softmax(output['z_mean'], Config.temperature, hard=True)
        # temperature = max(Config.temperature * (1 - Config.anneal_rate), Config.sample_std)
        recon_x, z, z_mean, z_logvar = (
            output["recon"],
            output["z"],
            output["z_mean"],
            output["z_logvar"],
        )
        loss_vae = loss_fn_vae(recon_x, z_mean, z_logvar, data, model)
        total_loss_vae += loss_vae.item()

        loss_vae.backward()
        vae_optimizer.step()

        # train the critic
        c_vae.eval()
        critic.train()

        critic_optimizer.zero_grad()
        # c: output of the critic from x (x was computed earlier using model(data))
        c = critic(data)

        # Detach the VAE output so that gradients do NOT flow into c_vae.
        with torch.no_grad():
            output = c_vae(c*data)
            recon_x, z, z_mean, z_logvar = (
                output["recon"],
                output["z"],
                output["z_mean"],
                output["z_logvar"],
            )

        # Compute the critic loss using the detached z_mean from c_vae.
        loss_critic = loss_fn_critic(recon_x, z_mean, c, data, model)

        loss_critic.backward()
        critic_optimizer.step()
        total_loss_critic += loss_critic.item()

        with torch.no_grad():
            _, pred_z = model(c * data).max(1)
            acc = (pred_base == pred_z).float().mean().item()
            acc_list.append(acc)

        pbar.set_postfix(loss=total_loss_critic / (pbar.n + 1))

    print(
        f"Epoch {epoch+1}/{Config.epochs}, Loss VAE: {total_loss_vae/len(train_loader):.4f}, Loss Critic: {total_loss_critic/len(train_loader):.4f}, Acc: {sum(acc_list)/len(acc_list):.4f}"
    )

# |%%--%%| <tQqyQelokQ|g1Uac4InPL>
r"""°°°
Plot images from the test set
°°°"""
# |%%--%%| <g1Uac4InPL|Pv1qVZYTpt>

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_results(test_loader, model, c_vae, critic, device):
    # Get test batch
    data, labels = next(iter(test_loader))
    data = data.to(device)

    with torch.no_grad():
        # Get model outputs
        x = model(data)
        output = c_vae(x)
        c = critic(x)

        # Convert to numpy arrays
        # Using "recon" or "z_mean" as your reconstructed image depends on your architecture.
        # Here we follow your original code:
        recon_imgs = output["z_mean"].cpu().numpy()
        true_imgs = data.cpu().numpy()
        # Reshape critic output to (batch, 28*28) then later to (28,28)
        c_values = c.cpu().numpy().reshape(-1, 28 * 28)
        labels = labels.cpu().numpy()

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot a grid of 3x3 images with the critic heatmap overlay
    for idx in range(9):
        ax = plt.subplot(3, 3, idx + 1)
        # Show the reconstructed image (grayscale)
        plt.imshow(recon_imgs[idx].reshape(28, 28), cmap="gray")
        # Overlay the critic's output as a heatmap (using 'jet' colormap)
        plt.imshow(c_values[idx].reshape(28, 28), cmap="jet", alpha=0.5)
        plt.colorbar()
        plt.title(f"True Label: {labels[idx]}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("reconstructed_images.png")
    plt.show()


def visualize_comparison_results(test_loader, model, c_vae, critic, device):
    # Get test batch
    data, labels = next(iter(test_loader))
    data = data.to(device)

    with torch.no_grad():
        # Get model outputs
        x = model(data)
        output = c_vae(x)
        c = critic(x)

        # Convert to numpy arrays
        recon_imgs = output["z_mean"].cpu().numpy()
        true_imgs = data.cpu().numpy()
        # Ensure critic output has shape (batch, 28, 28)
        c_values = c.cpu().numpy().reshape(-1, 28, 28)
        labels = labels.cpu().numpy()

    # Create figure with a grid of 3 rows x 9 columns
    plt.figure(figsize=(18, 6))
    
    num_samples = 9  # number of samples to display

    for idx in range(num_samples):
        # True image in row 1
        ax1 = plt.subplot(3, num_samples, idx + 1)
        plt.imshow(true_imgs[idx].reshape(28, 28), cmap="gray")
        plt.title(f"True: {labels[idx]}", fontsize=8)
        plt.axis("off")
        
        # Reconstructed image in row 2
        ax2 = plt.subplot(3, num_samples, idx + 1 + num_samples)
        plt.imshow(recon_imgs[idx].reshape(28, 28), cmap="gray")
        plt.title("Reconstructed", fontsize=8)
        plt.axis("off")
        
        # Critic heatmap in row 3
        ax3 = plt.subplot(3, num_samples, idx + 1 + 2*num_samples)
        im = plt.imshow(c_values[idx].reshape(28, 28), cmap="jet", alpha=0.5)
        plt.title("Critic Heatmap", fontsize=8)
        plt.axis("off")
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("comparison_images.png")
    plt.show()
# Usage
model.eval()
c_vae.eval()
critic.eval()
visualize_results(test_loader, model, c_vae, critic, Config.device)
visualize_comparison_results(train_loader, model, c_vae, critic, Config.device)
